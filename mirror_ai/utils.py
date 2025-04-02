from PIL import Image
import cv2
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from controlnet_aux import OpenposeDetector
from . import config


# Depth map utils
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(config.DEVICE)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(config.DEVICE)
    with torch.no_grad(), torch.autocast(config.DEVICE):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(config.DEFAULT_IMAGE_HEIGHT, config.DEFAULT_IMAGE_WIDTH),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

# Pose estimation
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(config.DEVICE)

def get_pose_map(image):
    pose_image = openpose(image)
    pose_image = pose_image.resize((config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT))
    return pose_image

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

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    "FROM: https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L65"
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
