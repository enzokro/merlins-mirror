import torch
import os
from PIL import Image
from controlnet_aux import (
    MidasDetector,
    OpenposeDetector,
)

from .. import config

class ControlNetPreprocessor:
    """
    Handles preprocessing images for ControlNet inputs.
    Initializes and manages different detectors (MiDaS, OpenPose, etc.)
    """

    def __init__(self):
        # Ensure cache directory exists
        os.makedirs(config.CONTROLNET_PREPROCESSOR_CACHE, exist_ok=True)
        
        # Initialize empty annotators dictionary to lazy-load as needed
        self.annotators = {}
        
        # Set device for preprocessing
        self.device = config.DEVICE

    def initialize_depth_detector(self):
        """Initialize the MiDaS depth detector."""
        print("Initializing MiDaS depth detector...")
        return MidasDetector.from_pretrained(
            "lllyasviel/Annotators",
            cache_dir=config.CONTROLNET_PREPROCESSOR_CACHE,
        )

    def initialize_pose_detector(self):
        """Initialize the OpenPose detector."""
        print("Initializing OpenPose detector...")
        return OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators",
            cache_dir=config.CONTROLNET_PREPROCESSOR_CACHE,
        )

    def get_depth_map(self, image):
        """
        Generate depth map from input image using MiDaS.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Processed depth map
        """
        if "depth" not in self.annotators:
            self.annotators["depth"] = self.initialize_depth_detector()
        
        return self.annotators["depth"](image)
    
    def get_pose_map(self, image):
        """
        Generate pose map from input image using OpenPose.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Processed pose map
        """
        if "pose" not in self.annotators:
            self.annotators["pose"] = self.initialize_pose_detector()
        
        return self.annotators["pose"](image)
    
    def process_control_image(self, image, control_type):
        """
        Process an image for the specified ControlNet type.
        
        Args:
            image (PIL.Image): Input image
            control_type (str): Type of control ('depth', 'pose')
            
        Returns:
            PIL.Image: Processed control image
        """
        if control_type == "depth":
            return self.get_depth_map(image)
        elif control_type == "pose":
            return self.get_pose_map(image)
        else:
            # For unsupported types, return the original image
            print(f"Warning: Control type '{control_type}' not supported. Using original image.")
            return image 