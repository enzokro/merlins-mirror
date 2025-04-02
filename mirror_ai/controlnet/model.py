import torch
import os
from diffusers import ControlNetModel

from .. import config

class ControlNetModels:
    """
    Handles loading and management of ControlNet models.
    Provides access to specific model types (depth, pose, etc.)
    """
    
    def __init__(self):
        # Ensure cache directory exists
        os.makedirs(config.CONTROLNET_CACHE_DIR, exist_ok=True)
        
        # Initialize model containers (lazy-loading)
        self._depth_model = None
        self._pose_model = None
    
    @property
    def depth_model(self):
        """Lazy-loaded depth ControlNet model."""
        if self._depth_model is None:
            print(f"Loading depth ControlNet model from {config.CONTROLNET_DEPTH}...")
            self._depth_model = ControlNetModel.from_pretrained(
                config.CONTROLNET_DEPTH,
                torch_dtype=config.DTYPE,
                cache_dir=config.CONTROLNET_CACHE_DIR
            ).to(config.DEVICE)
        return self._depth_model
    
    @property
    def pose_model(self):
        """Lazy-loaded pose ControlNet model."""
        if self._pose_model is None:
            print(f"Loading pose ControlNet model from {config.CONTROLNET_POSE}...")
            self._pose_model = ControlNetModel.from_pretrained(
                config.CONTROLNET_POSE,
                torch_dtype=config.DTYPE,
                cache_dir=config.CONTROLNET_CACHE_DIR
            ).to(config.DEVICE)
        return self._pose_model
    
    def get_active_models(self):
        """
        Returns a list of active ControlNet models based on config.
        
        Returns:
            list: List of initialized ControlNet models
        """
        models = []
        for model_path in config.CONTROLNETS:
            if model_path == config.CONTROLNET_DEPTH:
                models.append(self.depth_model)
            elif model_path == config.CONTROLNET_POSE:
                models.append(self.pose_model)
        
        return models
    
    def get_single_model(self):
        """
        Returns the first active ControlNet model if only one is configured,
        otherwise returns None.
        
        Returns:
            ControlNetModel or None: The first active model or None
        """
        active_models = self.get_active_models()
        if len(active_models) == 1:
            return active_models[0]
        return None 