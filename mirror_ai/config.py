import torch

# --- Core Model Identifiers ---
# Base SDXL model used for components like VAE and text encoders
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# Repository containing the SDXL Lightning UNet checkpoints
SDXL_LIGHTNING_REPO_ID = "ByteDance/SDXL-Lightning"
# ControlNet++ model repository
CONTROLNET_MODEL_ID = "xinsir/controlnet-union-sdxl-1.0"

# Random seed for reproducibility
SEED = 12297829382473034410 

# --- Pipeline Configuration ---
# Number of inference steps. MUST match the loaded UNet checkpoint (e.g., 2, 4, 8).
N_STEPS = 2
# Template for UNet checkpoint filenames. Use .format(n_steps=N_STEPS).
LIGHTNING_CKPT_TEMPLATE = "sdxl_lightning_{n_steps}step_unet.safetensors"
# Device to run the pipeline on ("cuda" or "cpu"). CUDA is highly recommended.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Data type for model parameters and inference. float16 is recommended for speed and memory.
DTYPE = torch.float16

# --- Inference Parameters ---
# Guidance scale. MUST be 0.0 for SDXL Lightning/Turbo models.
GUIDANCE_SCALE = 0.0
# Controlnet setup
CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_POSE = "thibaud/controlnet-openpose-sdxl-1.0"

CONTROLNETS = [
    CONTROLNET_DEPTH,
    # CONTROLNET_POSE,
]
# Conditioning scale for ControlNet. Balances prompt vs. control image influence. (Tunable)
CONTROLNET_CONDITIONING_SCALE = 0.75 # Default value, can be adjusted

# --- Scheduler Configuration ---
# Timestep spacing strategy. MUST be "trailing" for SDXL Lightning UNets (except 1-step).
SCHEDULER_TIMESTEP_SPACING = "trailing"
# Prediction type (only needed for experimental 1-step UNet: "sample").
# SCHEDULER_PREDICTION_TYPE = None # Keep None for N_STEPS=2, 4, 8

# --- Stable-Fast Optimization Configuration ---
# Attempt to enable xformers memory-efficient attention. Requires xformers installation.
SFAST_ENABLE_XFORMERS = True
# Attempt to enable Triton kernels. Requires Triton installation.
SFAST_ENABLE_TRITON = True
# Enable CUDA Graph optimization. Reduces CPU overhead but increases VRAM. (Tunable)
SFAST_ENABLE_CUDA_GRAPH = True
# Whether to use 4-bit quantization for lower VRAM usage
USE_QUANTIZATION = True  

# --- Warmup Configuration ---
# Number of initial inference runs to perform after compilation for Stable-Fast optimization.
WARMUP_RUNS = 3
# Default image size for SDXL. Conditioning images should ideally match this.
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024

DISPLAY_WIDTH = 640  # Width for web display
DISPLAY_HEIGHT = 480  # Height for web display

# Negative prompt for SDXL
NEGATIVE_PROMPT = "low quality, blurry, distorted, bad anatomy"

# Frame processing
FRAME_BLEND = 0.75

# message types for request queue
REQUEST_SET_PROMPT = "set_prompt"
REQUEST_REFRESH_LATENTS = "refresh_latents"
REQUEST_SHUTDOWN = "shutdown"

# message types for result queue
RESULT_FRAME = "frame"
RESULT_ERROR = "error"
RESULT_STATUS = "status"

# JPEG quality for the web app
JPEG_QUALITY = 75

    
# Video settings
VIDEO_PATH = 0  # Webcam index (0 for default camera)
FRAME_BLEND = 0.7  # Blending factor for frame interpolation (higher = smoother but more latency)
FRAME_SKIP = 2  # Only send every nth frame to reduce bandwidth

# todo: play around with capture settings for speed/quality

# Image dimensions
CAMERA_WIDTH = DISPLAY_WIDTH #DEFAULT_IMAGE_WIDTH  # Width for model input/output
CAMERA_HEIGHT = DISPLAY_HEIGHT #DEFAULT_IMAGE_HEIGHT  # Height for model input/output

# FPS for the video stream
CAMERA_FPS = 30

