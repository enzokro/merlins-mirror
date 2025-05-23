from PIL import Image
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetUnionPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetUnionModel,
    DiffusionPipeline,
    ControlNetModel,
    AutoencoderTiny,
)
from diffusers.models.attention_processor import AttnProcessor2_0
import cv2
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from mirror_ai.controlnet.controlnet_union import ControlNetModel_Union
from controlnet_aux import MidasDetector, OpenposeDetector, OpenposeDetector
from huggingface_hub import hf_hub_download
from diffusers.utils.torch_utils import randn_tensor
import torch_tensorrt
from safetensors.torch import load_file
from torchao.quantization import swap_conv2d_1x1_to_linear
from dotenv import load_dotenv
import os


from . import config
from .utils import dynamic_quant_filter_fn, conv_filter_fn

load_dotenv()
print(f"MODEL SETUP (device, dtype): {config.DEVICE}, {config.DTYPE}")
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

# set the pipeline class
pipe_class = StableDiffusionControlNetPipeline


class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self.generator = torch.Generator(device=torch.device(config.DEVICE)).manual_seed(config.SEED)


    def load(self, scheduler_name: str):
        """Loads all models, configures the pipeline, applies optimizations, and warms up."""
        print("--- Starting Pipeline Loading Process ---")

        # Load SDXL Lightning UNet
        print(f"Loading SD1.5 {config.N_STEPS}-Step Model...")

        # Load the tiny VAE
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=config.DEVICE, dtype=config.DTYPE)


        # # Load ControlNet Models
        self.depth_model = MidasDetector.from_pretrained("lllyasviel/Annotators").to(config.DEVICE)
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(config.DEVICE)
        controlnet_model_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=config.DTYPE,
        ).to(config.DEVICE)
        controlnet_model_pose = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(config.DEVICE)
        
        # for regular txt2img
        self.controlnet_model = controlnet_model = [
            controlnet_model_pose,
            controlnet_model_depth,
        ]

        # Create the Full Pipeline with Injected UNet, ControlNet, and VAE
        print("Creating the pipeline...")
        pipeline = pipe_class.from_pretrained(
            # 'Lykon/dreamshaper-xl-lightning',
            'lykon/dreamshaper-8',
            controlnet=controlnet_model,   # Inject the ControlNet
            vae=vae,                     # Inject the VAE
            torch_dtype=config.DTYPE,
            use_safetensors=True,
            variant="fp16"              
        ).to(config.DEVICE)

        # make sure we're using the new attention processor
        pipeline.unet.set_attn_processor(AttnProcessor2_0())

        # Configure the Scheduler (CRITICAL for SDXL Lightning)
        print(f"Configuring Scheduler ({scheduler_name}, spacing='{config.SCHEDULER_TIMESTEP_SPACING}')...")
        pipeline.scheduler = config.SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config,
            timestep_spacing=config.SCHEDULER_TIMESTEP_SPACING,
            use_karras_sigmas=True, 
            algorithm_type="sde-dpmsolver++",
        )
        print("Scheduler Configured.")

        # load the lightning lora weights
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_lora.safetensors"
        pipeline.load_lora_weights(
            hf_hub_download(repo, ckpt),
            adapter_name="lora_lightning_4step",
        )
        # pipeline.fuse_lora()

        # Set up Res-Adapter
        print("Loading Res-Adapter...")
        resadapter_model_name = "resadapter_v2_sd1.5"
        pipeline.load_lora_weights(
            hf_hub_download(
                repo_id="jiaxiangc/res-adapter", 
                subfolder=resadapter_model_name, 
                filename="pytorch_lora_weights.safetensors",
            ), 
            adapter_name="res_adapter",
            ) # load lora weights
        
        # set both loras
        pipeline.set_adapters(["res_adapter", "lora_lightning_4step"], adapter_weights=[0.7, 1.0])

        # set the pipeline
        self.pipeline = pipeline

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipeline.unet.config.in_channels
        self.latents_shape = (
            batch_size,
            num_channels_latents,
            config.RESA_HEIGHT // self.pipeline.vae_scale_factor,
            config.RESA_WIDTH // self.pipeline.vae_scale_factor,
        )
        self.refresh_latents()

        # Apply speedups and warmup the model
        self.pipeline.set_progress_bar_config(disable=True)

        # Speedup optimizations
        # # Fuse the QKV projections in the UNet.
        self.pipeline.fuse_qkv_projections()

        # ## Move the UNet and ControlNet to channels_last
        if config.DEVICE not in ("mps", "cpu"):
            self.pipeline.unet.to(memory_format=torch.channels_last)
            self.pipeline.controlnet.to(memory_format=torch.channels_last)

        # ## TODO: figure out dynamic quantization
        swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)

        print("Pipeline ready.") 


    def generate(
        self,
        prompt: str,
        camera_frame: Image.Image,
        num_inference_steps: int = None,
        controlnet_conditioning_scale: float = None,
    ) -> Image.Image:
        """
        Generates an image based on the prompt and conditioning image.

        Args:
            prompt: The text prompt to guide image generation.
            camera_frame: The current frame from the camera.
            num_inference_steps: Overrides the default N_STEPS from config. Must match model if overridden.
            guidance_scale: Overrides the default GUIDANCE_SCALE from config. Should likely remain 0.0.
            controlnet_conditioning_scale: Overrides the default CONTROLNET_CONDITIONING_SCALE from config.

        Returns:
            The generated PIL Image.

        Raises:
            ValueError: If the pipeline is not loaded.
            RuntimeError: If there's an error during inference.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not loaded. Call load() first.")

        # Use configured defaults if not overridden
        steps = num_inference_steps if num_inference_steps is not None else config.N_STEPS

        # Process control images based on active ControlNets
        camera_frame.save(f'{IMAGE_DEBUG_PATH}/image_orig_shape_{camera_frame.size}.jpg')
        camera_frame = camera_frame.resize((config.RESA_WIDTH, config.RESA_HEIGHT))

        ## manual controlnet params
        cn_scale = [0.5, 0.5]
        control_start = [0., 0.]
        control_end = [1.0, 1.0]
        size = (config.RESA_WIDTH, config.RESA_HEIGHT)
        # size=(config.SDXL_WIDTH, config.SDXL_HEIGHT)
    
        # get depth image
        # depth_image = self.get_depth_map(camera_frame).resize(size)
        depth_image = self.depth_model(camera_frame, output_type="pil").resize(size)

        # # get canny image
        # canny_image = np.array(camera_frame)
        # canny_image = cv2.Canny(canny_image, 100, 200)
        # canny_image = canny_image[:, :, None]
        # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        # canny_image = Image.fromarray(canny_image).resize(size)

        pose_image = self.openpose(camera_frame, hand_and_face=False).resize(size)

        control_image = [
            pose_image,
            depth_image,
            # canny_image,
        ]

        # save the images
        pose_image.save(f'{IMAGE_DEBUG_PATH}/image_depth_shape_{pose_image.size}.jpg')
        depth_image.save(f'{IMAGE_DEBUG_PATH}/image_pose_shape_{depth_image.size}.jpg')
        # canny_image.save(f'{IMAGE_DEBUG_PATH}/image_pose_shape_{canny_image.size}.jpg')

        try:
            with torch.no_grad():
                output_image = self.pipeline(
                    prompt=prompt,
                    # negative_prompt=config.NEGATIVE_PROMPT,

                    num_inference_steps=steps,

                    ## TODO: fix
                    # fixed generator and latents for consistent results between frames
                    generator=self.generator,
                    latents=self.latents, 

                    # for regular controlnet pipeline
                    image=control_image,
                    controlnet_conditioning_scale=cn_scale,
                    controlnet_guidance_start=control_start,
                    controlnet_guidance_end=control_end,

                    # output image setup
                    width=config.RESA_WIDTH, #config.SDXL_WIDTH,
                    height=config.RESA_HEIGHT, #SDXL_HEIGHT,
                    output_type="pil",

                ).images[0]
            # output_image.save(f'{IMAGE_DEBUG_PATH}/output_image_shape_{output_image.size}.jpg')

            return output_image

        except Exception as e:
            print(f"Error during pipeline inference: {e}")
            raise RuntimeError("Failed to generate image.") from e

    def refresh_latents(self):
        latents = randn_tensor(
            self.latents_shape,
            generator=self.generator,
            device=torch.device(config.DEVICE),
            dtype=config.DTYPE,
        )
        latents *= self.pipeline.scheduler.init_noise_sigma.to(config.DTYPE)
        print(f'Latents shape: {latents.shape}')
        self.latents = latents
