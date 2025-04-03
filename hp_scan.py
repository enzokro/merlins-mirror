"""Merlin looks into the mirror..."""
import os
from dotenv import load_dotenv
# import what we need here to avoid heavy imports in the web process
import traceback
import time
import itertools
import base64
from io import BytesIO
from PIL import Image
import torch
from mirror_ai.pipeline import ImagePipeline
from mirror_ai.video import VideoStreamer
from mirror_ai.utils import convert_to_pil_image, resize
from mirror_ai import config

load_dotenv()
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

# params to swwp
cfgs = [
    0,
    0.1,
    0.5,
    1.,
    2.0,
    3.5,
]

strengths = [
    0.1,
    0.5,
    0.75,
    0.8,
    0.99,
]

control_scales = [
    0.1,
    0.5,
    0.75,
    0.8,
    0.99,
]



def run():
    """Turns webcam frames into AI-generated images."""

    print("Initializing Merlin...")
    
    # set up the image transformer and the webcam stream
    pipeline = ImagePipeline()
    pipeline.load()
    video_streamer = VideoStreamer(
        camera_id=config.CAMERA_ID,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS
    )
    
    # start with no prompt, and default running
    current_prompt = "An astronaut sitting in space"
    camera_frame = Image.open(f'{IMAGE_DEBUG_PATH}/camera_01_frame.jpg')

    for idx, (cfg, strength, control_scale) in enumerate(itertools.product(cfgs, strengths, control_scales)):
        # transform the webcam image
        processed_frame = pipeline.generate(
            current_prompt,
            camera_frame,
            guidance_scale=cfg,
            strength=strength,
            controlnet_conditioning_scale=control_scale
        )
        processed_frame.save(f'{IMAGE_DEBUG_PATH}/camera_{idx}_generated_CFG_{cfg}_STR_{strength}_CNSCALE_{control_scale}.jpg')
        
        # convert to storable format
        pil_frame = convert_to_pil_image(processed_frame)
        pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_06_pil.jpg')

        # resize keeping aspect ratio
        pil_frame = pil_frame.resize((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_07_resized.jpg')

        # turn into base64 (easier to send through queue)
        img_byte_arr = BytesIO()
        pil_frame.save(img_byte_arr, format='JPEG', quality=config.JPEG_QUALITY)
        img_byte_arr.seek(0)
        encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                