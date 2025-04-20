from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderTiny,
    LCMScheduler,
    StableDiffusionXLAdapterPipeline
)
import PIL
import torch
from diffusers import StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter
from controlnet_aux import MidasDetector, OpenposeDetector, ZoeDetector, CannyDetector
from huggingface_hub import hf_hub_download
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from torchao.quantization import swap_conv2d_1x1_to_linear
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
import torchvision.transforms.functional as F


from . import config
from .utils import dynamic_quant_filter_fn, conv_filter_fn

# SET DTYPE
config.DTYPE = torch.bfloat16
# SET STEPS
config.N_STEPS = 1


# small torch speedups for compilation
torch.backends.cuda.matmul.allow_tf32 = True
# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# # for the linear swaps
# torch._inductor.config.force_fuse_int_mm_with_mul = True
# torch._inductor.config.use_mixed_mm = True


load_dotenv()
print(f"MODEL SETUP (device, dtype): {config.DEVICE}, {config.DTYPE}")
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

ADAPTER_REPO_IDS = {
    "canny": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    # "sketch": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    # "lineart": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
    "openpose": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
    # "recolor": "TencentARC/t2i-adapter-recolor-sdxl-1.0",
}

DEPTH_MODEL = "canny"
if DEPTH_MODEL == "depth-zoe":
    ADAPTER_REPO_IDS['depth-zoe'] = "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0"
elif DEPTH_MODEL == "depth-midas":
    ADAPTER_REPO_IDS['depth-midas'] = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
elif DEPTH_MODEL == "canny":
    ADAPTER_REPO_IDS['canny'] = "TencentARC/t2i-adapter-canny-sdxl-1.0"

ADAPTER_NAMES = list(ADAPTER_REPO_IDS.keys())

class Preprocessor(ABC):
    @abstractmethod
    def to(self, device: torch.device | str) -> "Preprocessor":
        pass

    @abstractmethod
    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        pass

class MidasPreprocessor(Preprocessor):
    def __init__(self):
        self.model = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
        )

    def to(self, device: torch.device | str) -> Preprocessor:
        self.model.to(device)
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=512, image_resolution=1024)


class OpenposePreprocessor(Preprocessor):
    def __init__(self):
        self.model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    def to(self, device: torch.device | str) -> Preprocessor:
        self.model.to(device)
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        out = self.model(image, detect_resolution=512, image_resolution=1024)
        out = np.array(out)[:, :, ::-1]
        out = PIL.Image.fromarray(np.uint8(out))
        return out
    
class ZoePreprocessor(Preprocessor):
    def __init__(self):
        self.model = ZoeDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="zoed_nk.pth", model_type="zoedepth_nk"
        )

    def to(self, device: torch.device | str) -> Preprocessor:
        self.model.to(device)
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, gamma_corrected=True, image_resolution=1024)
    
class CannyPreprocessor(Preprocessor):
    def __init__(self):
        self.model = CannyDetector()

    def to(self, device: torch.device | str) -> Preprocessor:
        return self

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return self.model(image, detect_resolution=384, image_resolution=1024)

device = config.DEVICE
PRELOAD_PREPROCESSORS_IN_GPU_MEMORY = True
preprocessors_gpu: dict[str, Preprocessor] = {
    # "canny": CannyPreprocessor().to(device),
    # "sketch": PidiNetPreprocessor().to(device),
    # "lineart": LineartPreprocessor().to(device),
    "openpose": OpenposePreprocessor().to(device),
    # "recolor": RecolorPreprocessor().to(device),
}

if DEPTH_MODEL == "depth-zoe":
    preprocessors_gpu['depth-zoe'] = ZoePreprocessor().to(device)
elif DEPTH_MODEL == "depth-midas":
    preprocessors_gpu['depth-midas'] = MidasPreprocessor().to(device)
elif DEPTH_MODEL == "canny":
    preprocessors_gpu['canny'] = CannyPreprocessor().to(device)

def get_preprocessor(adapter_name: str) -> Preprocessor:
    return preprocessors_gpu[adapter_name]

adapters = MultiAdapter(
    [
        T2IAdapter.from_pretrained(ADAPTER_REPO_IDS['openpose']),
        T2IAdapter.from_pretrained(ADAPTER_REPO_IDS[DEPTH_MODEL]),
    ]
)
adapters = adapters.to(config.DEVICE, dtype=config.DTYPE)

# set the pipeline class
pipe_class = StableDiffusionAdapterPipeline


class ImagePipeline:
    """Wraps the SDXL Lightning and ControlNet++ pipeline setup."""
    def __init__(self):
        self.pipeline = None
        self.generator = torch.Generator(device=torch.device(config.DEVICE)).manual_seed(config.SEED)


    def load(self, scheduler_name: str):
        """Loads all models, configures the pipeline, applies optimizations, and warms up."""
        print("--- Starting Pipeline Loading Process ---")

        # Load SDXL Hyper UNet
        print(f"Loading SDXL Hyper {config.N_STEPS}-Step UNet...")

        # Load the tiny VAE
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(config.DEVICE, dtype=config.DTYPE)

        pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
            # 'Lykon/dreamshaper-xl-1-0',
            # 'Lykon/dreamshaper-xl-v2-turbo',
            # 'Lykon/dreamshaper-xl-lightning',
            "stabilityai/stable-diffusion-xl-base-1.0",

            # pass in the adapters
            adapter=adapters,

            vae=vae,

            # torch_dtype=config.DTYPE,
            use_safetensors=True,
            torch_dtype=config.DTYPE,
            )
        pipeline.to(device=config.DEVICE, dtype=config.DTYPE)
        unet_state = load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors"), device=config.DEVICE)
        pipeline.unet.load_state_dict(unet_state)

        # Configure the Scheduler (CRITICAL for SDXL Lightning)
        # scheduler_name = "DDIM"
        # scheduler_name = "DPMSolverMultistep"
        # scheduler_name = "LCM"
        scheduler_name = "TCD"
        pipeline.scheduler = config.SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config, 
            # timestep_spacing ="trailing",
        )

        pipeline.load_lora_weights(
            "stabilityai/stable-diffusion-xl-base-1.0", weight_name="sd_xl_offset_example-lora_1.0.safetensors",
            adapter_name="offset"
        )
        # pipeline.fuse_lora(lora_scale=0.4)


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


        # Set the adapters and weights
        pipeline.set_adapters(
            [
                "offset", 
                "res_adapter",
            ], 
            adapter_weights=
            [
                0.4, 
                1.0,
            ])
        pipeline.fuse_lora(adapter_names=["offset", "res_adapter"], lora_scale=1.0)


        # set the pipeline
        self.pipeline = pipeline

        # Apply speedups and warmup the model
        self.pipeline.set_progress_bar_config(disable=True)

        # Speedup optimizations
        # # Fuse the QKV projections in the UNet.
        # self.pipeline.fuse_qkv_projections()

        # ## Move the UNet and ControlNet to channels_last
        if config.DEVICE not in ("mps", "cpu"):
            self.pipeline.unet.to(memory_format=torch.channels_last)
            self.pipeline.adapter.to(memory_format=torch.channels_last)

        # ## TODO: figure out dynamic quantization
        swap_conv2d_1x1_to_linear(self.pipeline.unet, conv_filter_fn)
        swap_conv2d_1x1_to_linear(self.pipeline.vae, conv_filter_fn)

        # Compile the UNet, VAE, and ControlNet
        backend="inductor"
        # print(f"Compiling the models with backend: {backend}")
        self.pipeline.unet = torch.compile(self.pipeline.unet, backend=backend, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, backend=backend, mode="max-autotune", fullgraph=True)
        # # Attempt to compile the ControlNet
        # self.pipeline.controlnet[0] = torch.compile(self.pipeline.controlnet[0], backend=backend, mode="max-autotune", fullgraph=True)
        # self.pipeline.controlnet[1] = torch.compile(self.pipeline.controlnet[1], backend=backend, mode="max-autotune", fullgraph=True)

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipeline.unet.config.in_channels
        self.latents_shape = (
            batch_size,
            num_channels_latents,
            config.LATENTS_HEIGHT // self.pipeline.vae_scale_factor,
            config.LATENTS_WIDTH // self.pipeline.vae_scale_factor)
        self.refresh_latents()

        print("Pipeline loaded.") 
        return self.pipeline


    def generate(
        self,
        prompt: str,
        camera_frame: Image.Image,
        num_inference_steps: int = None,
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
        
        # set the image size
        size = (config.RESA_WIDTH, config.RESA_HEIGHT)

        # Use configured defaults if not overridden
        steps = num_inference_steps if num_inference_steps is not None else config.N_STEPS

        # resize the camera frame
        # camera_frame = camera_frame.resize((config.RESA_WIDTH, config.RESA_HEIGHT))

        # get depth image
        depth_image = get_preprocessor(DEPTH_MODEL)(camera_frame)#.resize(size)
        pose_image = get_preprocessor('openpose')(camera_frame)#.resize(size)

        control_image = [
            pose_image,
            depth_image,
        ]

        # save the images
        # pose_image.save(f'{IMAGE_DEBUG_PATH}/image_depth_shape_{pose_image.size}.jpg')
        # depth_image.save(f'{IMAGE_DEBUG_PATH}/image_pose_shape_{depth_image.size}.jpg')

        ## manual controlnet params
        adapter_conditioning_scale = [1.0, 1.0]
        adapter_conditioning_factor = 1.0

        guidance_scale = 0.7

        try:
            with torch.inference_mode():
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

                    # for t2i pipeline
                    image=control_image,
                    adapter_conditioning_scale=adapter_conditioning_scale,
                    adapter_conditioning_factor=adapter_conditioning_factor,

                    # output image setup
                    # width=config.RESA_WIDTH, #config.SDXL_WIDTH,
                    # height=config.RESA_HEIGHT, #SDXL_HEIGHT,
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
