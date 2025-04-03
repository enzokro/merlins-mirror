import time
import cv2
from PIL import Image
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    # AutoPipelineForImage2Image,
)
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from huggingface_hub import hf_hub_download
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from torchao.quantization import swap_conv2d_1x1_to_linear
from dotenv import load_dotenv
import os

# from torchao.quantization import apply_dynamic_quant

from . import config
from .utils import dynamic_quant_filter_fn, conv_filter_fn
from .controlnet import ControlNetPreprocessor, ControlNetModels

# small torch speedups for compilation
# torch.backends.cuda.matmul.allow_tf32 = True
# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# # for the linear swaps
# torch._inductor.config.force_fuse_int_mm_with_mul = True
# torch._inductor.config.use_mixed_mm = True

load_dotenv()
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")


print(f"MODEL SETUP (device, dtype): {config.DEVICE}, {config.DTYPE}")

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self._is_compiled = False
        self._is_warmed_up = False
        self.generator = torch.Generator(device="cpu").manual_seed(config.SEED)
        
        # Initialize ControlNet preprocessor and models
        self.controlnet_preprocessor = None
        self.controlnet_models = None

    def load(self, scheduler_name: str):
        """Loads all models, configures the pipeline, applies optimizations, and warms up."""
        print("--- Starting Pipeline Loading Process ---")

        # --- 1. Load SDXL Lightning UNet  ---
        print(f"Loading SDXL Lightning {config.N_STEPS}-Step UNet")

        lightning_ckpt_file = config.LIGHTNING_CKPT_TEMPLATE.format(n_steps=config.N_STEPS)

        # Instantiate UNet from config, then load the specific Lightning weights
        # unet = UNet2DConditionModel.from_config(config.SDXL_BASE_MODEL_ID, subfolder="unet").to(config.DEVICE, dtype=config.DTYPE)
        unet_config = UNet2DConditionModel.load_config(config.SDXL_BASE_MODEL_ID, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to(config.DEVICE, dtype=config.DTYPE)
        unet.load_state_dict(
            load_file(
                hf_hub_download(config.SDXL_LIGHTNING_REPO_ID, lightning_ckpt_file),
                device=config.DEVICE
            )
        )
        unet.to(dtype=config.DTYPE)
        print("UNet Loaded and moved to device.")

        # --- 3. Load ControlNet Models ---
        print("Loading ControlNet models...")
        # self.depth_processor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        # self.processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        self.controlnet_models = ControlNetModels()
        controlnet = self.controlnet_models.get_active_models()[0]
        # controlnet = ControlNetModel.from_pretrained(
        #     # "xinsir/controlnet-canny-sdxl-1.0",
        #     "xinsir/controlnet-depth-sdxl-1.0",
        #     torch_dtype=torch.float16
        # )

        # load the vae
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        # Initialize the preprocessor
        self.controlnet_preprocessor = ControlNetPreprocessor()

        # --- 4. Create the Full Pipeline with Injected UNet and ControlNet ---
        print("Instantiating StableDiffusionXLPipeline...")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            config.SDXL_BASE_MODEL_ID,
            unet=unet,                  # Inject the loaded Lightning UNet
            controlnet=controlnet,      # Inject the loaded ControlNet
            vae=vae, # Inject the loaded VAE
            torch_dtype=config.DTYPE,
            use_safetensors=True,
            safety_checker=None,
            variant="fp16"              # Standard practice for SDXL
        ).to(config.DEVICE)
        print("Pipeline Instantiated and moved to device.")

        # --- 5. Configure the Scheduler (CRITICAL for SDXL Lightning) ---
        print(f"Configuring Scheduler ({scheduler_name}, spacing='{config.SCHEDULER_TIMESTEP_SPACING}')...")
        pipeline.scheduler = config.SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config,
            timestep_spacing=config.SCHEDULER_TIMESTEP_SPACING
            # If using 1-step model, add: prediction_type=config.SCHEDULER_PREDICTION_TYPE
        )
        print("Scheduler Configured.")

        self.pipeline = pipeline

        # --- 6. Initialize latents ---
        batch_size = 1
        num_channels_latents = self.pipeline.unet.config.in_channels
        self.latents_shape = (batch_size, num_channels_latents, config.DEFAULT_IMAGE_HEIGHT // self.pipeline.vae_scale_factor, config.DEFAULT_IMAGE_WIDTH // self.pipeline.vae_scale_factor)
        self.refresh_latents()

        # --- 7. Apply speedups and warmup the model ---
        self.pipeline.set_progress_bar_config(disable=True)

        # # Fuse the QKV projections in the UNet.
        # self.pipeline.fuse_qkv_projections()

        if config.DEVICE != "mps":
            self.pipeline.unet.to(memory_format=torch.channels_last)
            if isinstance(self.pipeline.controlnet, list):
                for cn in self.pipeline.controlnet:
                    cn.to(memory_format=torch.channels_last)
            else:
                self.pipeline.controlnet.to(memory_format=torch.channels_last)

        # TODO: figure out dynamic quantization
        # swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        # swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)
        # apply_dynamic_quant(self.pipeline.unet, dynamic_quant_filter_fn)
        # apply_dynamic_quant(self.pipeline.vae, dynamic_quant_filter_fn)

        # Compile the UNet and VAE.
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="max-autotune", fullgraph=True)

        print("--- Pipeline Loading Process Complete ---") 

    def _process_control_images(self, camera_frame):
        """
        Process control images for each active ControlNet.
        
        Args:
            camera_frame (PIL.Image): Input camera frame
            
        Returns:
            list or PIL.Image: Processed control images
        """
        # Get active control types
        control_types = []
        for model_path in config.CONTROLNETS:
            if model_path == config.CONTROLNET_DEPTH:
                control_types.append("depth")
            elif model_path == config.CONTROLNET_POSE:
                control_types.append("pose")
        
        # Process control images
        if len(control_types) == 1:
            return self.controlnet_preprocessor.process_control_image(camera_frame, control_types[0])
        else:
            return [self.controlnet_preprocessor.process_control_image(camera_frame, ct) for ct in control_types]

    def generate(
        self,
        prompt: str,
        camera_frame: Image.Image,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        controlnet_conditioning_scale: float = None,
        strength: float = None,
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
        strength = strength if strength is not None else config.STRENGTH

        # Ensure the conditioning image is appropriately sized (optional, but good practice)
        # SDXL typically expects 1024x1024.
        if camera_frame.size != (config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT):
            print(f"Warning: Image size {camera_frame.size} differs from default ({config.DEFAULT_IMAGE_WIDTH}, {config.DEFAULT_IMAGE_HEIGHT}). Resizing...")
            camera_frame = camera_frame.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT))

        # width, height = camera_frame.size
        # ratio = np.sqrt(config.DEFAULT_IMAGE_WIDTH * config.DEFAULT_IMAGE_HEIGHT / (width * height))
        # new_width, new_height = int(width * ratio), int(height * ratio)
        # camera_frame = camera_frame.resize((new_width, new_height))
        # camera_frame.save(f'{IMAGE_DEBUG_PATH}/reshaped_camera_frame_shape_{camera_frame.size}.jpg')

        # control_img = self.processor_midas(camera_frame, image_resolution=config.DEFAULT_IMAGE_WIDTH, output_type='pil')
        # # need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
        # height, width, _  = camera_frame.shape
        # ratio = np.sqrt(1024. * 1024. / (width * height))
        # new_width, new_height = int(width * ratio), int(height * ratio)
        # control_img = cv2.resize(control_img, (new_width, new_height))
        # control_img = Image.fromarray(control_img)
        # control_img = control_img.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT), resample=Image.Resampling.NEAREST)

        # for canny
        # control_img = cv2.Canny(np.asarray(camera_frame), config.CANNY_EDGE_LOW, config.CANNY_EDGE_HIGH)
        # control_img = HWC3(control_img)
        # control_img = Image.fromarray(control_img)
        # control_img.save(f'{IMAGE_DEBUG_PATH}/control_image_shape_{control_img.size}.jpg')

        # # Process control images based on active ControlNets
        control_img = self._process_control_images(camera_frame)
        control_img = control_img.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT), resample=Image.Resampling.NEAREST)
        # print(f"Control image type: {type(control_image)}")

        control_img.save(f'{IMAGE_DEBUG_PATH}/control_image_shape_{control_img.size}.jpg')

        print(f"Running Inference: steps={steps}, guidance={g_scale}, cn_scale={cn_scale}...")
        start_time = time.time()

        try:
            with torch.no_grad():
                output_image = self.pipeline(
                    prompt=prompt,

                    # For ControlNetPipeline
                    image=control_img,
                    # control_image=control_img,

                    # # For ControlNetImg2ImgPipeline
                    # image=camera_frame,
                    # control_image=control_img,

                    num_inference_steps=steps,

                    # ControlNet parameters
                    guidance_scale=g_scale,
                    controlnet_conditioning_scale=cn_scale,
                    strength=strength,
                    control_guidance_start=config.CONTROL_GUIDANCE_START,
                    control_guidance_end=config.CONTROL_GUIDANCE_END,

                    negative_prompt=config.NEGATIVE_PROMPT,
                    generator=self.generator,

                    # fixed latents for consistent results between frames
                    latents=self.get_latents(), 

                    # output image setup
                    width=config.DEFAULT_IMAGE_WIDTH,
                    height=config.DEFAULT_IMAGE_HEIGHT,
                    output_type="pil",
                ).images[0]
            output_image.save(f'{IMAGE_DEBUG_PATH}/output_image_shape_{output_image.size}.jpg')

            end_time = time.time()
            print(f"Inference finished in {end_time - start_time:.3f} seconds.")
            return output_image

        except Exception as e:
            print(f"Error during pipeline inference: {e}")
            raise RuntimeError("Failed to generate image.") from e
        
    def get_latents(self):
        return self.latents.clone()

    def refresh_latents(self):
        # self.latents = torch.randn(self.latents_shape, generator=self.generator, device=config.DEVICE, dtype=config.DTYPE)
        self.latents = randn_tensor(self.latents_shape, generator=self.generator, device=torch.device(config.DEVICE), dtype=config.DTYPE)
