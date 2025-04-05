import time
import cv2
from PIL import Image
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetUnionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)
from mirror_ai.controlnet.controlnet_union import ControlNetModel_Union
from controlnet_aux import MidasDetector, OpenposeDetector
from huggingface_hub import hf_hub_download
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from torchao.quantization import swap_conv2d_1x1_to_linear
from dotenv import load_dotenv
import os

# from torchao.quantization import apply_dynamic_quant

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
pipe_class = StableDiffusionXLControlNetUnionPipeline

class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self.generator = torch.Generator(device="cpu").manual_seed(config.SEED)


    def load(self, scheduler_name: str):
        """Loads all models, configures the pipeline, applies optimizations, and warms up."""
        print("--- Starting Pipeline Loading Process ---")

        # Load SDXL Lightning UNet
        print(f"Loading SDXL Lightning {config.N_STEPS}-Step UNet...")

        # Instantiate UNet from config, then load the specific Lightning weights
        unet_config = UNet2DConditionModel.load_config(config.SDXL_BASE_MODEL_ID, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to(config.DEVICE, dtype=config.DTYPE)
        lightning_ckpt_file = config.LIGHTNING_CKPT_TEMPLATE.format(n_steps=config.N_STEPS)
        unet.load_state_dict(
            load_file(
                hf_hub_download(config.SDXL_LIGHTNING_REPO_ID, lightning_ckpt_file),
                device=config.DEVICE
            )
        )
        unet.to(dtype=config.DTYPE)

        # Load ControlNet Models
        print("Loading ControlNet models...")
        controlnet_model = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)
        self.depth_model = MidasDetector.from_pretrained("lllyasviel/Annotators").to(config.DEVICE)
        self.pose_model  = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(config.DEVICE)

        # Load the fixed vae
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        vae.to(config.DEVICE)

        # Create the Full Pipeline with Injected UNet, ControlNet, and VAE
        print("Creating the pipeline...")
        pipeline = pipe_class.from_pretrained(
            config.SDXL_BASE_MODEL_ID,
            unet=unet,                     # Inject the Lightning UNet
            controlnet=controlnet_model,   # Inject the ControlNet
            vae=vae,                       # Inject the VAE
            torch_dtype=config.DTYPE,
            use_safetensors=True,
            safety_checker=None,
            variant="fp16"              
        ).to(config.DEVICE)

        # Configure the Scheduler (CRITICAL for SDXL Lightning)
        print(f"Configuring Scheduler ({scheduler_name}, spacing='{config.SCHEDULER_TIMESTEP_SPACING}')...")
        pipeline.scheduler = config.SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config,
            timestep_spacing=config.SCHEDULER_TIMESTEP_SPACING
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
            ) # load lora weights
        pipeline.set_adapters(["res_adapter"], adapter_weights=[1.0])

        # set the pipeline
        self.pipeline = pipeline

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipeline.unet.config.in_channels
        self.latents_shape = (batch_size, num_channels_latents, config.DEFAULT_IMAGE_HEIGHT // self.pipeline.vae_scale_factor, config.DEFAULT_IMAGE_WIDTH // self.pipeline.vae_scale_factor)
        self.refresh_latents()

        # Apply speedups and warmup the model
        self.pipeline.set_progress_bar_config(disable=True)

        # Speedup optimizations
        ## Fuse the QKV projections in the UNet.
        self.pipeline.fuse_qkv_projections()

        ## Move the UNet and ControlNet to channels_last
        if config.DEVICE not in ("mps", "cpu"):
            self.pipeline.unet.to(memory_format=torch.channels_last)
            if isinstance(self.pipeline.controlnet, list):
                for cn in self.pipeline.controlnet:
                    cn.to(memory_format=torch.channels_last)
            else:
                self.pipeline.controlnet.to(memory_format=torch.channels_last)

        ## TODO: figure out dynamic quantization
        swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)
        # apply_dynamic_quant(self.pipeline.unet, dynamic_quant_filter_fn)
        # apply_dynamic_quant(self.pipeline.vae, dynamic_quant_filter_fn)

        ## Compile the UNet, VAE, and ControlNet
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="max-autotune", fullgraph=True)
        # self.pipeline.controlnet = torch.compile(self.pipeline.controlnet, mode="max-autotune", fullgraph=True)

        print("Pipeline loaded.") 


    def generate(
        self,
        prompt: str,
        camera_frame: Image.Image,
        num_inference_steps: int = None,
        guidance_scale: float = None,
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
        g_scale = guidance_scale if guidance_scale is not None else config.GUIDANCE_SCALE
        cn_scale = controlnet_conditioning_scale if controlnet_conditioning_scale is not None else config.CONTROLNET_CONDITIONING_SCALE


        # Process control images based on active ControlNets
        cv_img = np.asarray(camera_frame)
        image_depth = self.depth_model(cv_img, output_type="pil")
        image_pose = self.pose_model(cv_img, hand_and_face=False, output_type="pil")

        # image_depth.save(f'{IMAGE_DEBUG_PATH}/image_depth_shape_{image_depth.size}.jpg')
        # image_pose.save(f'{IMAGE_DEBUG_PATH}/image_pose_shape_{image_pose.size}.jpg')

        # print(f"Running Inference: steps={steps}, guidance={g_scale}, cn_scale={cn_scale}...")
        # start_time = time.time()

        control_start = config.CONTROL_GUIDANCE_START
        control_end = config.CONTROL_GUIDANCE_END

        control_start = [control_start, control_start]
        control_end = [control_end, control_end]
        cn_scale = [cn_scale, cn_scale]

        try:
            with torch.no_grad():
                output_image = self.pipeline(
                    prompt=[prompt]*1,
                    negative_prompt=[config.NEGATIVE_PROMPT]*1,

                    guidance_scale=g_scale,
                    num_inference_steps=steps,
                    generator=self.generator,

                    # fixed latents for consistent results between frames
                    latents=self.get_latents(), 

                    # ControlNet parameters
                    image_list=[image_pose, image_depth, 0, 0, 0, 0], 
                    controlnet_conditioning_scale=cn_scale,
                    control_guidance_start=control_start,
                    control_guidance_end=control_end,
                    union_control=True,
                    union_control_type=torch.Tensor([1, 1, 0, 0, 0, 0]),
                    crops_coords_top_left=(0, 0),

                    # output image setup
                    width=config.DEFAULT_IMAGE_WIDTH,
                    height=config.DEFAULT_IMAGE_HEIGHT,
                    output_type="pil",

                ).images[0]
            # output_image.save(f'{IMAGE_DEBUG_PATH}/output_image_shape_{output_image.size}.jpg')

            # end_time = time.time()
            # print(f"Inference finished in {end_time - start_time:.3f} seconds.")
            return output_image

        except Exception as e:
            print(f"Error during pipeline inference: {e}")
            raise RuntimeError("Failed to generate image.") from e
        
    def get_latents(self):
        return self.latents.clone()

    def refresh_latents(self):
        self.latents = randn_tensor(self.latents_shape, generator=self.generator, device=torch.device(config.DEVICE), dtype=config.DTYPE)
