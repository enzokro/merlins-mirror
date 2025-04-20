from PIL import Image
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetUnionPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetUnionModel,
    DiffusionPipeline,
    ControlNetModel,
    AutoencoderTiny,
)
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

# small torch speedups for compilation
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
# # for the linear swaps
# torch._inductor.config.force_fuse_int_mm_with_mul = True
# torch._inductor.config.use_mixed_mm = True

load_dotenv()
print(f"MODEL SETUP (device, dtype): {config.DEVICE}, {config.DTYPE}")
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

# set the pipeline class
pipe_class = StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler


class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self.generator = torch.Generator(device=torch.device(config.DEVICE)).manual_seed(config.SEED)


    def load(self, scheduler_name: str):
        """Loads all models, configures the pipeline, applies optimizations, and warms up."""
        print("--- Starting Pipeline Loading Process ---")

        # # Load ControlNet Models
        self.depth_model = MidasDetector.from_pretrained("lllyasviel/Annotators").to(config.DEVICE)
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(config.DEVICE)

        controlnet_model_depth = ControlNetModel.from_pretrained(
            "xinsir/controlnet-depth-sdxl-1.0",
            # variant="fp16",
            # use_safetensors=True,
            torch_dtype=config.DTYPE,
        ).to(config.DEVICE)

        controlnet_model_pose = ControlNetModel.from_pretrained(
            "xinsir/controlnet-openpose-sdxl-1.0",
            # variant="fp16",
            # use_safetensors=True,
            torch_dtype=config.DTYPE,
        ).to(config.DEVICE)
        
        # for regular txt2img
        self.controlnet_model = controlnet_model = [
            controlnet_model_pose,
            controlnet_model_depth,
        ]

       # Load the tiny VAE
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(config.DEVICE, dtype=config.DTYPE)


        # Create the Full Pipeline with Injected UNet, ControlNet, and VAE
        print("Creating the pipeline...")
        pipeline = pipe_class.from_pretrained(
            # 'Lykon/dreamshaper-xl-1-0',
            # 'Lykon/dreamshaper-xl-v2-turbo',
            # 'Lykon/dreamshaper-xl-lightning',
            # 'fluently/Fluently-XL-Final',
            "sd-community/sdxl-flash",
            controlnet=controlnet_model,   # Inject the ControlNet
            torch_dtype=config.DTYPE,
            vae=vae,
            use_safetensors=True,
            # variant="fp16"              
        ).to(config.DEVICE)

        # Configure the Scheduler (CRITICAL for SDXL Lightning)
        # scheduler_name = "DDIM"
        scheduler_name = "TCD"
        print(f"Configuring Scheduler ({scheduler_name}, spacing='{config.SCHEDULER_TIMESTEP_SPACING}')...")
        # scheduler_name = "DPMSolverMultistep"
        pipeline.scheduler = config.SCHEDULERS[scheduler_name].from_config(
        # pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipeline.scheduler.config,
            # num_train_timesteps=1000,

            # # For DDIM
            # timestep_spacing="trailing", 
            # clip_sample=False, 
            # set_alpha_to_one=False,

            # # For TCD
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
            
            # # For DPMSolverMultistep
            # use_karras_sigmas=True, 
            # algorithm_type="sde-dpmsolver++",
            # # Optional (often beneficial for PCM consistency models):
            # timestep_spacing="trailing",
            # solver_order=2,               # optimal solver order for speed/quality balance
            # lower_order_final=True,       # helps improve stability at very low steps (e.g., 2 steps)
        )
        print("Scheduler Configured.")

        # Set up Res-Adapter
        print("Loading Res-Adapter...")
        resadapter_model_name = "resadapter_v2_sdxl"
        pipeline.load_lora_weights(
            hf_hub_download(
                repo_id="jiaxiangc/res-adapter", 
                subfolder=resadapter_model_name, 
                filename="pytorch_lora_weights.safetensors",
            ), 
            adapter_name="res_adapter",
            )

        # load PCM weights
        checkpoint = f"pcm_sdxl_smallcfg_{config.N_STEPS}step_converted.safetensors"
        # checkpoint = f"pcm_sdxl_normalcfg_{config.N_STEPS}step_converted.safetensors"
        mode = "sdxl"
        pipeline.load_lora_weights(
            "wangfuyun/PCM_Weights", weight_name=checkpoint, subfolder=mode,
            adapter_name="pcm",
        )

        # Set the adapters and weights
        pipeline.set_adapters(
            [
                "res_adapter", 
                "pcm",
            ], 
            adapter_weights=
            [
                0.7, 
                1.0,
            ])

        # set the pipeline
        self.pipeline = pipeline

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipeline.unet.config.in_channels
        self.latents_shape = (
            batch_size,
            num_channels_latents,
            config.RESA_HEIGHT // self.pipeline.vae_scale_factor,
            config.RESA_WIDTH // self.pipeline.vae_scale_factor)
        self.refresh_latents()

        # Apply speedups and warmup the model
        self.pipeline.set_progress_bar_config(disable=True)

        # # Speedup optimizations
        # # # Fuse the QKV projections in the UNet.
        # self.pipeline.fuse_qkv_projections()

        # ## Move the UNet and ControlNet to channels_last
        if config.DEVICE not in ("mps", "cpu"):
            self.pipeline.unet.to(memory_format=torch.channels_last)
            self.pipeline.controlnet.to(memory_format=torch.channels_last)

        # ## TODO: figure out dynamic quantization
        swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)

        # Compile the UNet, VAE, and ControlNet
        # backend="tensorrt"
        # print(f"Compiling the models with backend: {backend}")
        # self.pipeline.unet = torch.compile(self.pipeline.unet, backend=backend, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, backend=backend, mode="max-autotune", fullgraph=True)
        # # Attempt to compile the ControlNet
        # self.pipeline.controlnet[0] = torch.compile(self.pipeline.controlnet[0], backend=backend, mode="max-autotune", fullgraph=True)
        # self.pipeline.controlnet[1] = torch.compile(self.pipeline.controlnet[1], backend=backend, mode="max-autotune", fullgraph=True)

        # warmup the model
        print("Warming up the model...")
        img = randn_tensor((1, 3, 512, 512), generator=self.generator, device=torch.device(config.DEVICE), dtype=config.DTYPE)
        self.pipeline(
            prompt="TEST",
            image=[img, img],
            num_inference_steps=1,
            width=config.RESA_WIDTH,
            height=config.RESA_HEIGHT,
        )

        print("Pipeline loaded.") 


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
        size = (config.RESA_WIDTH, config.RESA_HEIGHT)
        # size=(config.SDXL_WIDTH, config.SDXL_HEIGHT)

        # Process control images based on active ControlNets
        # camera_frame.save(f'{IMAGE_DEBUG_PATH}/image_orig_shape_{camera_frame.size}.jpg')
        camera_frame = camera_frame.resize((config.RESA_WIDTH, config.RESA_HEIGHT))

        # print(f"Running Inference: steps={steps}, guidance={g_scale}, cn_scale={cn_scale}...")

        # set the image size
        size = (config.RESA_WIDTH, config.RESA_HEIGHT)
        # size=(config.SDXL_WIDTH, config.SDXL_HEIGHT)
    
        # get depth image
        depth_image = self.depth_model(camera_frame, output_type="pil").resize(size)

        pose_image = self.openpose(camera_frame, hand_and_face=True, output_type="pil").resize(size)

        control_image = [
            pose_image,
            depth_image,
        ]

        # save the images
        # pose_image.save(f'{IMAGE_DEBUG_PATH}/image_depth_shape_{pose_image.size}.jpg')
        # depth_image.save(f'{IMAGE_DEBUG_PATH}/image_pose_shape_{depth_image.size}.jpg')

        ## manual controlnet params
        cn_scale = [1.0, 1.0]
        control_start = [0., 0.]
        control_end = [1.0, 1.0]

        guidance_scale = 1.0

        try:
            with torch.no_grad():
                output_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=config.NEGATIVE_PROMPT,

                    num_inference_steps=steps,

                    # set the guidance scale
                    guidance_scale=guidance_scale,

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
        latents *= self.pipeline.scheduler.init_noise_sigma
        print(f'Latents shape: {latents.shape}')
        self.latents = latents
