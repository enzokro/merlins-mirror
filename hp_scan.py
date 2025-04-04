"""Merlin looks into the mirror..."""
import os
from dotenv import load_dotenv
# import what we need here to avoid heavy imports in the web process
import itertools
from PIL import Image
from mirror_ai.pipeline import ImagePipeline
from mirror_ai import config

load_dotenv()
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

# params to sweep
cfgs = [
    0,
    0.1,
    0.5,
    1.,
    2.0,
    3.5,
]

# strengths = [
#     0.1,
#     0.5,
#     0.75,
#     0.8,
#     0.99,
# ]

control_scales = [
    0.1,
    0.5,
    0.75,
    0.8,
    0.99,
]

schedulers = [
    "K_EULER_ANCESTRAL",
    "K_EULER",
    "KarrasDPM",
    "HeunDiscrete",
    "DPMSolverMultistep",
]


def run():
    """Turns webcam frames into AI-generated images."""

    print("Initializing Merlin...")
    
    # set up the image transformer and the webcam stream
    for scheduler in schedulers:
        pipeline = ImagePipeline()
        pipeline.load(scheduler)
        
        # start with no prompt, and default running
        current_prompt = "A fractal psychedelic living room"
        camera_frame = Image.open(f'empty_room.jpg')
        camera_frame = camera_frame.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT))

        for idx, (cfg, strength, control_scale) in enumerate(itertools.product(cfgs, control_scales)):
            # transform the webcam image
            pil_frame = pipeline.generate(
                current_prompt,
                camera_frame,
                guidance_scale=cfg,
                strength=strength,
                controlnet_conditioning_scale=control_scale
            )
            pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_{idx}_generated_CFG_{cfg}_STR_{strength}_CNSCALE_{control_scale}_SCHEDULER_{scheduler}.jpg')

            # resize keeping aspect ratio
            pil_frame = pil_frame.resize((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_final_CFG_{cfg}_STR_{strength}_CNSCALE_{control_scale}_SCHEDULER_{scheduler}_resized.jpg')
                