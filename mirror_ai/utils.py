from PIL import Image
import cv2
import numpy as np
import torch
from . import config

def interpolate_images(image1, image2, alpha=config.FRAME_BLEND):
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    return Image.blend(image1, image2, alpha)

def convert_to_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    elif torch.is_tensor(image):
        if image.device != torch.device('cpu'):
            image = image.to('cpu')
        image = image.numpy() if image.requires_grad else image.detach().numpy()
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        if image.shape[2] == 1:
            image = image[:, :, 0]
        return Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float32 else image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return Image.fromarray(image)
    else:
        raise TypeError("Unsupported image type")
    
def dynamic_quant_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (1280, 640),
            (1920, 1280),
            (1920, 640),
            (2048, 1280),
            (2048, 2560),
            (2560, 1280),
            (256, 128),
            (2816, 1280),
            (320, 640),
            (512, 1536),
            (512, 256),
            (512, 512),
            (640, 1280),
            (640, 1920),
            (640, 320),
            (640, 5120),
            (640, 640),
            (960, 320),
            (960, 640),
        ]
    )


def conv_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )
