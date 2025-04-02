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
from safetensors.torch import load_file
from dotenv import load_dotenv
from mirror_ai import config

# load the env vars that set the HF_HOME directory
load_dotenv()

def download_models():
    """Downloads all models, configures the pipeline, applies optimizations, and warms up."""

    """Loads all models, configures the pipeline, applies optimizations, and warms up."""
    print("--- Starting Model Download Process ---")

    # --- 1. Load SDXL Lightning UNet  ---
    print(f"Loading SDXL Lightning {config.N_STEPS}-Step UNet")

    lightning_ckpt_file = config.LIGHTNING_CKPT_TEMPLATE.format(n_steps=config.N_STEPS)

    # Instantiate UNet from config, then load the specific Lightning weights
    unet = UNet2DConditionModel.from_config(config.SDXL_BASE_MODEL_ID, subfolder="unet").to(config.DEVICE, dtype=config.DTYPE)
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
    pipeline = StableDiffusionXLPipeline.from_pretrained(
    # pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
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

    print("Models downloaded.")

if __name__ == "__main__":
    download_models()