#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants
from typing import List, Tuple, Optional
from collections import OrderedDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            # TODO: Try saving external data. Requires ONNX opset >= 16.
            print("WARNING: model size exceeds supported 2GB limit. Trying saving external data.")
            # raise TypeError("ERROR: model size exceeds supported 2GB limit")
            #shape_inference.infer_shapes_path(onnx_graph, external_data=True)
            # TODO: Fix this to allow external data saving/loading
            onnx_graph = shape_inference.infer_shapes(onnx_graph) # Fallback for now
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(
        self,
        fp16=True,
        device="cuda",
        verbose=True,
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class TraceableUNetWrapper(torch.nn.Module):
    """
    Wrapper for HuggingFace Diffusers UNet model to make it traceable for ONNX export
    by accepting flattened residual inputs as positional arguments and passing them
    as keyword arguments to the original model's forward method.
    This version expects 12 down-block residuals and 1 mid-block residual.
    """
    def __init__(self, original_unet, all_input_names: list[str]):
        super().__init__()
        self.original_unet = original_unet
        # Store the full list of input names in the order they will be received in forward
        self.all_input_names = all_input_names # Should be 16 names now

        # Validate expected input count
        if len(all_input_names) != 16:
            raise ValueError(
                f"TraceableUNetWrapper expected 16 input names (sample, ts, hs, 12 down, 1 mid), "
                f"but received {len(all_input_names)}: {all_input_names}"
            )

        # Identify the names expected for down block and mid block residuals
        # based on the corrected UNet spec naming convention (down_res_0 to down_res_11)
        def sort_key(name):
            if name.startswith("down_res_"):
                try:
                    return int(name.split('_')[-1])
                except (ValueError, IndexError):
                    return float('inf')
            return float('inf')

        # Extract and sort the 12 expected down residual names
        self.down_res_names = sorted(
            [name for name in all_input_names if name.startswith("down_res_")],
            key=sort_key
        )
        if len(self.down_res_names) != 12:
            raise ValueError(f"Wrapper expected 12 'down_res_' names, found {len(self.down_res_names)}: {self.down_res_names}")

        # Identify the mid residual name
        self.mid_res_name = "mid_res_0"
        if self.mid_res_name not in all_input_names:
            # Check for alternative common names if needed, but stick to convention for now
             raise ValueError(f"Expected mid residual name '{self.mid_res_name}' not found in input names: {all_input_names}")

        # Store the indices corresponding to each type of input *from the all_input_names list*
        try:
            self.sample_idx = all_input_names.index("sample")
            self.timestep_idx = all_input_names.index("timestep")
            self.encoder_hidden_states_idx = all_input_names.index("encoder_hidden_states")
            # Get indices for the *correctly sorted* down residual names
            self.down_res_indices = [all_input_names.index(name) for name in self.down_res_names]
            self.mid_res_idx = all_input_names.index(self.mid_res_name)
        except ValueError as e:
            raise ValueError(f"Critical input name missing from all_input_names list provided to TraceableUNetWrapper: {e}. Provided list: {all_input_names}")

        print(f"TraceableUNetWrapper initialized (12+1 residuals):")
        print(f"  Expecting {len(all_input_names)} total inputs in order: {self.all_input_names}")
        print(f"  Reconstructing 12 down residuals from indices: {self.down_res_indices} (Names: {self.down_res_names})")
        print(f"  Expecting mid residual at index: {self.mid_res_idx} (Name: {self.mid_res_name})")


    def forward(self, *args):
        # Check if the number of received arguments matches expectation (should be 16)
        if len(args) != len(self.all_input_names):
             raise ValueError(f"Wrapper forward received {len(args)} arguments, but expected {len(self.all_input_names)} based on initialization ({self.all_input_names}).")

        # Extract arguments based on stored indices
        sample = args[self.sample_idx]
        timestep = args[self.timestep_idx]
        encoder_hidden_states = args[self.encoder_hidden_states_idx]

        # Reconstruct the down_residuals tuple using the *correctly sorted* indices
        down_residuals_list = [args[i] for i in self.down_res_indices]
        down_residuals_tuple = tuple(down_residuals_list) if down_residuals_list else None

        # Extract the mid_residual
        mid_residual = args[self.mid_res_idx]

        # --- START HACK: Force tracer to see ALL residuals --- #
        # Perform a trivial operation that depends on the residuals.
        # Summing ensures dependency without changing values significantly.
        _tracer_trick_sum = 0.0
        if down_residuals_tuple is not None: # Check if tuple exists and is not empty
            for res in down_residuals_tuple:
                if res is not None: # Ensure individual residuals are not None
                     _tracer_trick_sum = _tracer_trick_sum + torch.sum(res) * 0.0
        if mid_residual is not None:
             _tracer_trick_sum = _tracer_trick_sum + torch.sum(mid_residual) * 0.0
        # We will add this _tracer_trick_sum to the output later.
        # --- END HACK --- #

        # Call the original UNet's forward, passing residuals as keyword arguments
        # return_dict=False is crucial for getting a tensor output for ONNX
        output = self.original_unet.forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            # --- PASS RESIDUALS AS KEYWORD ARGS --- #
            down_block_additional_residuals=down_residuals_tuple,
            mid_block_additional_residual=mid_residual,
            # --- END KEYWORD ARGS --- #
            return_dict=False, # Necessary for ONNX export
            # Set other optional args explicitly to None if they exist in the original signature
            # to avoid potential issues during tracing if defaults change.
            class_labels=None,
            timestep_cond=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            down_intrablock_additional_residuals=None,
            encoder_attention_mask=None,
        )

        # The original forward with return_dict=False returns a tuple, usually (sample,)
        if isinstance(output, tuple):
            if not output:
                 raise ValueError("Original UNet forward returned an empty tuple.")
            # Assume the first element is the main output tensor
            sample_output = output[0]
        elif torch.is_tensor(output): # Handle case where it might just return the tensor directly
            sample_output = output
        elif hasattr(output, 'sample'): # Handle object output just in case
             sample_output = output.sample
        else:
             raise TypeError(f"Unexpected output type from original_unet.forward: {type(output)}")

        # --- Apply the tracer trick sum to the output --- #
        modified_sample_output = sample_output + _tracer_trick_sum
        modified_sample_output = UNet2DConditionOutput(sample=modified_sample_output)
        # --- End Apply tracer trick --- #

        # Return only the modified sample tensor
        return modified_sample_output


class UNet(BaseModel):
    def __init__(
        self,
        fp16=False,
        device="cuda",
        hf_token=None, # Keep hf_token if UNet needs to load weights, otherwise remove
        max_batch_size=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        # Remove outdated/incorrect architecture assumptions
        # down_block_types=(...), block_out_channels=(...), layers_per_block=...,
        # num_controlnet_down_residuals=..., has_controlnet_mid_residual=...,
        verbose=False,
        out_channels=4, # Keep out_channels
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
            verbose=verbose,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.out_channels = out_channels # Store out_channels

        # --- Define residual shapes based on diffusers ControlNetModel (12 down + 1 mid) ---
        B = 2 * self.max_batch # Use 2B for classifier-free guidance shapes
        self.residual_shapes = OrderedDict({
            # Down Block 0 (H, W) - 4 residuals (Res0, Res1, Trans0, Trans1)
            "down_res_0": (B, 320, "H", "W"),
            "down_res_1": (B, 320, "H", "W"),
            "down_res_2": (B, 320, "H", "W"),
            "down_res_3": (B, 320, "H", "W"),
            # Down Block 1 (H/2, W/2) - 3 residuals (Res0, Trans0, Trans1)
            "down_res_4": (B, 640, "H/2", "W/2"),
            "down_res_5": (B, 640, "H/2", "W/2"),
            "down_res_6": (B, 640, "H/2", "W/2"),
            # Down Block 2 (H/4, W/4) - 3 residuals (Res0, Trans0, Trans1)
            "down_res_7": (B, 1280, "H/4", "W/4"),
            "down_res_8": (B, 1280, "H/4", "W/4"),
            "down_res_9": (B, 1280, "H/4", "W/4"),
            # Down Block 3 (H/8, W/8) - 2 residuals (Res0, Res1)
            "down_res_10": (B, 1280, "H/8", "W/8"),
            "down_res_11": (B, 1280, "H/8", "W/8"),
            # Mid Block (H/8, W/8) - 1 residual
            "mid_res_0":  (B, 1280, "H/8", "W/8")
        })

        self.down_residual_indices = list(range(12)) # Indices 0-11 for down residuals
        self.mid_residual_index = 0 # Only one mid residual

        print(f"UNet helper configured for 12 down residuals and 1 mid residual (diffusers ControlNet structure)")


    def get_input_names(self):
        # --- Corrected Input Names (12 down + 1 mid) ---
        names = ["sample", "timestep", "encoder_hidden_states"]
        # Add the 12 down residual names
        names.extend([f"down_res_{i}" for i in self.down_residual_indices])
        # Add the mid residual name
        names.append("mid_res_0")
        return names

    def get_output_names(self):
        return ["latent"]

    def _get_spatial_dims(self, latent_height, latent_width, block_level):
         # Helper calculates spatial dims based on block_level (0=input, 1=after first downsample, etc.)
         scale = 2**block_level
         # Ensure dimensions are at least 1
         h = max(1, latent_height // scale)
         w = max(1, latent_width // scale)
         return (h, w)

    # --- Corrected mapping from residual index (0-11) to block level (0-3) ---
    def _get_block_level_from_res_index(self, res_index):
         if res_index < 0 or res_index > 11:
              raise ValueError(f"Invalid residual index {res_index} for 12 down residuals.")
         if res_index < 4: return 0 # Indices 0-3 are in Block 0 (H, W)
         if res_index < 7: return 1 # Indices 4-6 are in Block 1 (H/2, W/2)
         if res_index < 10: return 2 # Indices 7-9 are in Block 2 (H/4, W/4)
         # Indices 10-11 are in Block 3 (H/8, W/8)
         return 3


    def get_dynamic_axes(self):
         axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B", 1: "S"},
            "latent": {0: "2B", 2: "H", 3: "W"},
         }
         # Add axes for the 12 down residuals + 1 mid residual
         for name, shape_tmpl in self.residual_shapes.items():
             # Use the correct dimension indices (0=Batch, 1=Channel, 2=Height, 3=Width)
             # Handle potential dynamic batch size ('2B') and sequence length ('S')
             h_expr = shape_tmpl[2] # e.g., "H", "H/8"
             w_expr = shape_tmpl[3] # e.g., "W", "W/8"
             axes[name] = {0: "2B", 2: h_expr, 3: w_expr} # Corrected: use index 2 for H, 3 for W
         return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch, _, _, _, _,
            min_latent_height, max_latent_height, min_latent_width, max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        # Use 2*B for classifier-free guidance
        minB, optB, maxB = min_batch * 2, batch_size * 2, max_batch * 2

        profile = OrderedDict({
            "sample": [
                (minB, self.unet_dim, min_latent_height, min_latent_width),
                (optB, self.unet_dim, latent_height, latent_width),
                (maxB, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(minB,), (optB,), (maxB,)], # Timestep is 1D
            "encoder_hidden_states": [
                (minB, self.text_maxlen, self.embedding_dim),
                (optB, self.text_maxlen, self.embedding_dim),
                (maxB, self.text_maxlen, self.embedding_dim),
            ],
        })

        # Add profiles for residuals using the corrected 12+1 structure
        res_names = list(self.residual_shapes.keys())
        # Iterate through all 13 residual names (12 down + 1 mid)
        for name in res_names:
            shape_tmpl = self.residual_shapes[name]
            channels = shape_tmpl[1]

            if name == "mid_res_0":
                block_idx = 3 # Mid block operates at H/8, W/8
            else:
                # Use the corrected mapping from residual index (0-11) to block level
                res_index = int(name.split('_')[-1]) # Get the number suffix
                block_idx = self._get_block_level_from_res_index(res_index)

            min_h, min_w = self._get_spatial_dims(min_latent_height, min_latent_width, block_idx)
            opt_h, opt_w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            max_h, max_w = self._get_spatial_dims(max_latent_height, max_latent_width, block_idx)

            profile[name] = [
                (minB, channels, min_h, min_w),
                (optB, channels, opt_h, opt_w),
                (maxB, channels, max_h, max_w),
            ]

        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        B = batch_size * 2 # Factor of 2 for classifier-free guidance

        shape_dict = OrderedDict({
            "sample": (B, self.unet_dim, latent_height, latent_width),
            "timestep": (B,), # Timestep is 1D
            "encoder_hidden_states": (B, self.text_maxlen, self.embedding_dim),
            "latent": (B, self.out_channels, latent_height, latent_width), # Output shape
        })

        # Add residual shapes using corrected 12+1 structure
        res_names = list(self.residual_shapes.keys())
        # Iterate through all 13 residual names (12 down + 1 mid)
        for name in res_names:
            shape_tmpl = self.residual_shapes[name]
            channels = shape_tmpl[1]

            if name == "mid_res_0":
                 block_idx = 3 # Mid block operates at H/8, W/8
            else:
                 # Use the corrected mapping from residual index (0-11) to block level
                 res_index = int(name.split('_')[-1]) # Get the number suffix
                 block_idx = self._get_block_level_from_res_index(res_index)

            h, w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            shape_dict[name] = (B, channels, h, w)

        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        B = batch_size * 2 # Factor of 2 for classifier-free guidance

        inputs = OrderedDict()
        inputs["sample"] = torch.randn(
            B, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
        )
        # Ensure timestep is float32 as expected by embedding layers
        inputs["timestep"] = torch.ones((B,), dtype=torch.float32, device=self.device)
        inputs["encoder_hidden_states"] = torch.randn(
            B, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device
        )

        # Generate sample residuals using corrected 12+1 structure
        res_names = list(self.residual_shapes.keys())
        # Iterate through all 13 residual names (12 down + 1 mid)
        for name in res_names:
            shape_tmpl = self.residual_shapes[name]
            channels = shape_tmpl[1]

            if name == "mid_res_0":
                 block_idx = 3 # Mid block operates at H/8, W/8
            else:
                 # Use the corrected mapping from residual index (0-11) to block level
                 res_index = int(name.split('_')[-1]) # Get the number suffix
                 block_idx = self._get_block_level_from_res_index(res_index)

            h, w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            inputs[name] = torch.randn(B, channels, h, w, dtype=dtype, device=self.device)

        # Return OrderedDict matching the order from get_input_names()
        # This ensures the tuple passed to torch.onnx.export is correctly ordered
        return OrderedDict([(name, inputs[name]) for name in self.get_input_names()])

    # Optimize method can remain as it operates on the graph structure
    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        latents = torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float16, #torch.float32, # VAE typically uses float32
            device=self.device,
        )
        return {'latent': latents}

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        images = torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32, #torch.float32, # VAE typically uses float32
            device=self.device,
        )
        return {'images': images}
    
    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph
