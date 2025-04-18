from typing import *

import torch
import tensorrt as trt
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        # Store input names from the engine for dynamic handling
        self.input_names = []
        self.down_residual_input_names = [] # Store sorted down_block_res_* names
        self.mid_residual_input_name = None # Store mid_block_res name if present
        self.base_input_names = {"sample", "timestep", "encoder_hidden_states"} # Known base inputs

        # Discover and categorize input names from the loaded engine
        for i in range(self.engine.engine.num_io_tensors):
             name = self.engine.engine.get_tensor_name(i)
             if self.engine.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                 self.input_names.append(name)
                 if name.startswith("down_block_res_"):
                     self.down_residual_input_names.append(name)
                 elif name == "mid_block_res":
                     self.mid_residual_input_name = name
                 elif name not in self.base_input_names:
                     print(f"Warning: Unknown input tensor '{name}' found in engine.")

        # Sort residual names for consistent mapping (e.g., down_block_res_0, _1, ...)
        self.down_residual_input_names.sort()

        print(f"UNet Engine Initialized.")
        print(f"  Total Inputs: {self.input_names}")
        print(f"  Base Inputs Detected: {{name for name in self.input_names if name in self.base_input_names}}")
        print(f"  Down Residual Inputs Detected ({len(self.down_residual_input_names)}): {self.down_residual_input_names}")
        print(f"  Mid Residual Input Detected: {self.mid_residual_input_name}")


    def __call__(
        self,
        latent_model_input: torch.Tensor, # Corresponds to "sample" input in ONNX/TRT
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        # ControlNet residuals passed from the pipeline
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        **kwargs, # Allow passing other potential kwargs if needed
    ) -> UNet2DConditionOutput:

        # --- Input Validation & Preparation ---
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        # Ensure timestep is 1D (handle potential 0D input)
        if timestep.ndim == 0:
            batch_size = latent_model_input.shape[0] # Use latent input batch size
            timestep = timestep.repeat(batch_size)
        elif timestep.ndim != 1:
             raise ValueError(f"Expected timestep to be 0D or 1D, but got {timestep.ndim}D shape: {timestep.shape}")

        # Prepare base shapes and feed dict (mapping pipeline args to engine input names)
        shape_dict = {
            "sample": latent_model_input.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
        }
        feed_dict = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        # --- Dynamically Add Residual Shapes & Feed Dict Entries ---

        # Add down block residuals
        num_expected_down_residuals = len(self.down_residual_input_names)
        if num_expected_down_residuals > 0:
            if down_block_additional_residuals is None:
                raise ValueError(f"Engine expects {num_expected_down_residuals} down block residuals, but 'down_block_additional_residuals' is None.")
            if len(down_block_additional_residuals) != num_expected_down_residuals:
                raise ValueError(
                    f"Mismatch in number of down block residuals. Engine expects {num_expected_down_residuals} ({self.down_residual_input_names}), "
                    f"but received tuple has {len(down_block_additional_residuals)} elements."
                )

            # Map the tuple elements to the named engine inputs
            for i, name in enumerate(self.down_residual_input_names):
                current_residual = down_block_additional_residuals[i]
                shape_dict[name] = current_residual.shape
                feed_dict[name] = current_residual

        # Add mid block residual
        if self.mid_residual_input_name is not None:
            if mid_block_additional_residual is None:
                raise ValueError(f"Engine expects mid block residual ('{self.mid_residual_input_name}'), but 'mid_block_additional_residual' is None.")
            shape_dict[self.mid_residual_input_name] = mid_block_additional_residual.shape
            feed_dict[self.mid_residual_input_name] = mid_block_additional_residual

        # --- Allocate Buffers & Infer --- #

        # Allocate buffers based on the complete shape_dict
        # The engine wrapper's allocate_buffers method handles determining output shapes now.
        self.engine.allocate_buffers(
            shape_dict=shape_dict,
            device=latent_model_input.device, # Use device from a primary input
        )

        # Run inference
        outputs = self.engine.infer(
            feed_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )

        # Assume the primary output is named "latent" as defined in models.py
        output_name = "latent"
        if output_name not in outputs:
             raise RuntimeError(f"Expected output tensor '{output_name}' not found in engine outputs: {list(outputs.keys())}")

        # Return wrapped output
        return UNet2DConditionOutput(sample=outputs[output_name])

    # --- Compatibility Methods --- #
    def to(self, *args, **kwargs):
        # Make this compatible with HF pipeline's .to() calls
        # The engine is already loaded to a device (implicitly CUDA via Polygraphy/TRT)
        # This method can be a no-op.
        pass

    def forward(self, *args, **kwargs):
        # Allow calling the engine via .forward() as pipelines might expect
        return self.__call__(*args, **kwargs)


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor, **kwargs):
        self.encoder.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            },
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        self.decoder.allocate_buffers(
            shape_dict={
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            },
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
