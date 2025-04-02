import time
from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    AutoencoderKL,
)
from huggingface_hub import hf_hub_download
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from torchao.quantization import swap_conv2d_1x1_to_linear
# from torchao.quantization import apply_dynamic_quant

from . import config
from .utils import get_depth_map, get_pose_map, dynamic_quant_filter_fn, conv_filter_fn

# small torch speedups for compilation
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
# # for the linear swaps
# torch._inductor.config.force_fuse_int_mm_with_mul = True
# torch._inductor.config.use_mixed_mm = True

### STABLE-FAST SECTION ###
#######################################################
# # Optional Stable-Fast import
# try:
#     from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
#     is_sfast_available = True
# except ImportError:
#     print("Stable-Fast library not found. Pipeline will run without optimization.")
#     is_sfast_available = False

# # Optional xformers/triton check for logging
# try:
#     import xformers
#     is_xformers_available = True
# except ImportError:
#     is_xformers_available = False
# try:
#     import triton
#     is_triton_available = True
# except ImportError:
#     is_triton_available = False
### /STABLE-FAST SECTION ###
#######################################################

print(f"MODEL SETUP (device, dtype): {config.DEVICE}, {config.DTYPE}")

class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self._is_compiled = False
        self._is_warmed_up = False
        self.generator = torch.Generator(device="cpu").manual_seed(config.SEED)

    def load(self):
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
        print("UNet Loaded and moved to device.")

        # --- 2a. Load the fixed VAE ---
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )

        # --- 3. Load ControlNet++ ---
        controlnet = []
        for cnet in config.CONTROLNETS:
            print(f"Loading ControlNet ({cnet})...")
            controlnet_model = ControlNetModel.from_pretrained(
                cnet,
                torch_dtype=config.DTYPE
            ).to(config.DEVICE)
            controlnet.append(controlnet_model)
        if len(controlnet) == 1:
            controlnet = controlnet[0]

        # --- 4. Create the Full Pipeline with Injected UNet and ControlNet ---
        print("Instantiating StableDiffusionXLPipeline...")
        # pipeline = StableDiffusionXLPipeline.from_pretrained(
        pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            config.SDXL_BASE_MODEL_ID,
            unet=unet,                  # Inject the loaded Lightning UNet
            controlnet=controlnet,  # Inject the loaded ControlNet
            vae=vae,                    # Inject the fixed VAE
            torch_dtype=config.DTYPE,
            use_safetensors=True,
            variant="fp16"              # Standard practice for SDXL
        ).to(config.DEVICE)
        print("Pipeline Instantiated and moved to device.")

        # --- 5. Configure the Scheduler (CRITICAL for SDXL Lightning) ---
        print(f"Configuring Scheduler (EulerDiscrete, spacing='{config.SCHEDULER_TIMESTEP_SPACING}')...")
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
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

        # Fuse the QKV projections in the UNet.
        self.pipeline.fuse_qkv_projections()

        if config.DEVICE != "mps":
            self.pipeline.unet.to(memory_format=torch.channels_last)
            self.pipeline.controlnet.to(memory_format=torch.channels_last)

        # TODO: figure out dynamic quantization
        # swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        # swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)
        # apply_dynamic_quant(self.pipeline.unet, dynamic_quant_filter_fn)
        # apply_dynamic_quant(self.pipeline.vae, dynamic_quant_filter_fn)

        # Compile the UNet and VAE.
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="max-autotune", fullgraph=True)

        # self._apply_stable_fast()
        self._warmup()
        print("--- Pipeline Loading Process Complete ---")


    def _warmup(self):
        """Performs warmup runs, essential after Stable-Fast compilation."""
        if not self._is_compiled or config.WARMUP_RUNS <= 0:
            if config.WARMUP_RUNS > 0:
                print("Skipping warmup: Pipeline not compiled with Stable-Fast.")
            self._is_warmed_up = True # Mark as 'warmed up' even if skipped
            return

        print(f"Performing {config.WARMUP_RUNS} Warmup Runs...")
        # Use a dummy prompt and a blank conditioning image for warmup
        dummy_prompt = "warmup"
        dummy_conditioning_image = Image.new("RGB", (config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT))

        kwarg_inputs_warmup = dict(
            prompt=dummy_prompt,
            image=dummy_conditioning_image,
            num_inference_steps=config.N_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            output_type="latent" # Faster warmup, don't need VAE decode
        )
        try:
            for i in range(config.WARMUP_RUNS):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.pipeline(**kwarg_inputs_warmup)
                end_time = time.time()
                print(f"Warmup run {i+1}/{config.WARMUP_RUNS} completed in {end_time - start_time:.3f} seconds.")
            self._is_warmed_up = True
            print("Warmup Complete.")
        except Exception as e:
            print(f"Error during warmup: {e}. Performance may be suboptimal.")
            # proceed anyway, but flag that warmup didn't fully complete
            self._is_warmed_up = False 

    def generate(
        self,
        prompt: str,
        camera_frame: Image.Image,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        controlnet_conditioning_scale: float = None
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

        # Ensure the conditioning image is appropriately sized (optional, but good practice)
        # SDXL typically expects 1024x1024.
        if camera_frame.size != (config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT):
            print(f"Warning: Conditioning image size {camera_frame.size} differs from default ({config.DEFAULT_IMAGE_WIDTH}, {config.DEFAULT_IMAGE_HEIGHT}). Resizing...")
            camera_frame = camera_frame.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT))

        # get the depth map and pose map
        depth_map = get_depth_map(camera_frame)
        control_image = depth_map

        # # for both open pose and depth
        # pose_map = get_pose_map(camera_frame)
        # control_image = [depth_map, pose_map]

        print(f"Running Inference: steps={steps}, guidance={g_scale}, cn_scale={cn_scale}...")
        start_time = time.time()

        try:
            with torch.no_grad():
                output_image = self.pipeline(
                    prompt=prompt,

                    # ## For ControlNetImg2ImgPipeline
                    image=camera_frame,
                    control_image=control_image,

                    # ## For StableDiffusionXLPipeline
                    # image=control_image,

                    num_inference_steps=steps,

                    # MOVING OVER VALUES FROM BEFORE
                    controlnet_conditioning_scale=0.5,
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                    strength=0.75,
                    guidance_scale=0,


                    negative_prompt=config.NEGATIVE_PROMPT,
                    generator=self.generator,

                    # fixed latents for consistent results between frames
                    latents=self.get_latents(), 

                    # output image setup
                    width=config.DEFAULT_IMAGE_WIDTH,
                    height=config.DEFAULT_IMAGE_HEIGHT,
                    output_type="pil",
                ).images[0]

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
    
    # def _apply_stable_fast(self):
    #     """Applies Stable-Fast compilation if available and configured."""
    #     if not is_sfast_available:
    #         print("Skipping Stable-Fast: Library not available.")
    #         return

    #     print("Applying Stable-Fast Compilation...")
    #     sfast_config = CompilationConfig.Default()
    #     compile_success = False
    #     try:
    #         if config.SFAST_ENABLE_XFORMERS:
    #             if is_xformers_available:
    #                 sfast_config.enable_xformers = True
    #                 print("Stable-Fast: xformers enabled.")
    #             else:
    #                 print("Stable-Fast: xformers requested but not installed, disabling.")
    #                 sfast_config.enable_xformers = False
    #         else:
    #              sfast_config.enable_xformers = False
    #              print("Stable-Fast: xformers disabled by config.")

    #         if config.SFAST_ENABLE_TRITON:
    #             if is_triton_available:
    #                 sfast_config.enable_triton = True
    #                 print("Stable-Fast: Triton enabled.")
    #             else:
    #                 print("Stable-Fast: Triton requested but not installed, disabling.")
    #                 sfast_config.enable_triton = False
    #         else:
    #              sfast_config.enable_triton = False
    #              print("Stable-Fast: Triton disabled by config.")

    #         if config.SFAST_ENABLE_CUDA_GRAPH:
    #             sfast_config.enable_cuda_graph = True
    #             print("Stable-Fast: CUDA Graph enabled.")
    #         else:
    #             sfast_config.enable_cuda_graph = False
    #             print("Stable-Fast: CUDA Graph disabled by config.")

    #         self.pipeline = compile(self.pipeline, sfast_config)
    #         self._is_compiled = True
    #         compile_success = True
    #         print("Stable-Fast Compilation Complete.")

        # except Exception as e:
        #     print(f"Error during Stable-Fast compilation: {e}. Pipeline will run unoptimized.")
        #     # Ensure pipeline is still the original if compilation fails mid-way
        #     # (compile might return partially modified object on error, safer to reload or just use original)
        #     # For simplicity here, we assume the original self.pipeline reference is usable.
        #     self._is_compiled = False

        # if compile_success:
        #      print("Pipeline successfully compiled with Stable-Fast.")
        # else:
        #      print("Pipeline running without Stable-Fast optimization due to compilation issues or config.")

