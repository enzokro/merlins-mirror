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
        
        models = []
        for model_path in config.CONTROLNETS:
            models.append(self.get_model(model_path))

        self.models = models

    
    def get_model(self, model_path):
        print(f"Loading depth ControlNet model from {model_path}...")
        model = ControlNetModel.from_pretrained(
            model_path,
            torch_dtype=config.DTYPE,
            cache_dir=config.CONTROLNET_CACHE_DIR
        ).to(config.DEVICE)
        return model
    
    
    def get_active_models(self):
        """
        Returns a list of active ControlNet models based on config.
        
        Returns:
            list: List of initialized ControlNet models
        """
        return self.models

    
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