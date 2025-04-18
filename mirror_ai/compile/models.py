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
    with flattened ControlNet residual inputs provided as positional arguments.
    """
    def __init__(self, original_unet, all_input_names: list[str]):
        super().__init__()
        self.original_unet = original_unet
        # Store the full list of input names in the order they will be received in forward
        self.all_input_names = all_input_names

        # Identify the names expected for down block and mid block residuals
        # --- FIX: Sort numerically ---
        def sort_key(name):
            if name.startswith("down_block_res_"):
                try:
                    return int(name.split('_')[-1])
                except ValueError:
                    return float('inf') # Place non-numeric suffixes last
            return float('inf') # Place other names last

        self.down_res_names = sorted(
            [name for name in all_input_names if name.startswith("down_block_res_")],
            key=sort_key
        )
        # --- END FIX ---
        self.mid_res_name = "mid_block_res" if "mid_block_res" in all_input_names else None

        # Store the indices corresponding to each type of input *from the original all_input_names list*
        try:
            self.sample_idx = all_input_names.index("sample")
            self.timestep_idx = all_input_names.index("timestep")
            self.encoder_hidden_states_idx = all_input_names.index("encoder_hidden_states")
            # Get indices for the *correctly sorted* residual names
            self.down_res_indices = [all_input_names.index(name) for name in self.down_res_names]
            self.mid_res_idx = all_input_names.index(self.mid_res_name) if self.mid_res_name else -1
        except ValueError as e:
            raise ValueError(f"Critical input name missing from all_input_names list provided to TraceableUNetWrapper: {e}. Provided list: {all_input_names}")


        print(f"TraceableUNetWrapper initialized:")
        print(f"  Expecting {len(all_input_names)} total inputs in order: {self.all_input_names}")
        print(f"  Reconstructing down residuals from indices: {self.down_res_indices} (Names: {self.down_res_names})") # Will now show correct order
        print(f"  Expecting mid residual at index: {self.mid_res_idx} (Name: {self.mid_res_name})")


    def forward(self, *args):
        # Check if the number of received arguments matches expectation
        if len(args) != len(self.all_input_names):
             raise ValueError(f"Wrapper forward received {len(args)} arguments, but expected {len(self.all_input_names)} based on initialization.")

        # Extract arguments based on stored indices
        sample = args[self.sample_idx]
        timestep = args[self.timestep_idx]
        encoder_hidden_states = args[self.encoder_hidden_states_idx]

        # Reconstruct the down_residuals tuple using the *correctly sorted* indices
        down_residuals = [args[i] for i in self.down_res_indices]

        # Extract the mid_residual
        mid_residual = args[self.mid_res_idx] if self.mid_res_idx != -1 else None

        # --- START HACK: Force tracer to see residuals ---
        # Perform a trivial operation that depends on the residuals
        # Summing ensures dependency without changing values significantly if inputs are small
        _tracer_trick_sum = 0.0
        if down_residuals: # Check if list is not empty
            _tracer_trick_sum += sum(torch.sum(res) for res in down_residuals) * 0.0
        if mid_residual is not None:
             _tracer_trick_sum += torch.sum(mid_residual) * 0.0
        # You might need to incorporate this sum into the final output calculation
        # in a trivial way if the above isn't sufficient, e.g., output + _tracer_trick_sum
        # --- END HACK ---

        # Convert the potentially modified list back to a tuple for the original UNet
        down_residuals_tuple = tuple(down_residuals) if self.down_res_names else None
        print("NUMBER OF DOWN RESIDUALS: ", len(down_residuals_tuple))

        # Call the original UNet's forward, passing residuals in the expected format
        # return_dict=False is crucial for getting a tensor output for ONNX
        output = self.original_unet.forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_residuals_tuple,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
            # Explicitly set others to None if they exist in the signature
            # class_labels=None, timestep_cond=None, attention_mask=None,
            # cross_attention_kwargs=None, added_cond_kwargs=None,
            # down_intrablock_additional_residuals=None, # Keep this None unless T2I adapter used
            # encoder_attention_mask=None,
        )

        # --- START FIX: Handle tuple output and apply tracer trick ---
        # The original forward with return_dict=False returns a tuple, usually (sample,)
        if isinstance(output, tuple):
            if not output:
                 raise ValueError("Original UNet forward returned an empty tuple.")
            sample_output = output[0]
        elif hasattr(output, 'sample'): # Handle case where it might return an object despite return_dict=False (less likely but safe)
             sample_output = output.sample
        else:
             # If it's neither a tuple nor has a .sample attribute, raise an error or handle appropriately
             raise TypeError(f"Unexpected output type from original_unet.forward: {type(output)}")

        # Apply the tracer trick sum to the extracted sample tensor
        modified_sample_output = sample_output + _tracer_trick_sum
        # --- END FIX ---

        # Return the modified sample tensor
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
        # Provide actual architecture details for SD 1.5
        down_block_types=(
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
        ),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        # ControlNet residual specifics (Assuming standard SD 1.5 ControlNet output structure)
        num_controlnet_down_residuals=12,
        has_controlnet_mid_residual=True,
        verbose=False,
        out_channels=4, # Add out_channels explicitly
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
        # Store necessary architecture details if needed elsewhere, but remove from input definitions
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.down_block_types = down_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block # Needed for calculating residual shapes potentially
        self.num_down_blocks = len(down_block_types)
        self.num_controlnet_down_residuals = num_controlnet_down_residuals
        self.has_controlnet_mid_residual = has_controlnet_mid_residual
        self.out_channels = out_channels # Store out_channels

        # --- Calculate expected residual shapes based on SD1.5 architecture ---
        # This mapping assumes the ControlNet residuals match the UNet's internal skip connection shapes
        self.residual_shapes = OrderedDict()
        current_height, current_width = "H", "W" # Placeholders for dynamic axes
        res_idx = 0
        # Down blocks
        # Hardcode the typical 12 residual shapes for SD 1.5 ControlNet structure
        # Block 0 out: 320 -> 3 residuals
        self.residual_shapes["down_block_res_0"] = (2 * self.max_batch, 320, "H", "W")
        self.residual_shapes["down_block_res_1"] = (2 * self.max_batch, 320, "H", "W")
        self.residual_shapes["down_block_res_2"] = (2 * self.max_batch, 320, "H", "W")
        # Block 1 out: 640 -> 3 residuals
        self.residual_shapes["down_block_res_3"] = (2 * self.max_batch, 640, "H/2", "W/2")
        self.residual_shapes["down_block_res_4"] = (2 * self.max_batch, 640, "H/2", "W/2")
        self.residual_shapes["down_block_res_5"] = (2 * self.max_batch, 640, "H/2", "W/2")
        # Block 2 out: 1280 -> 3 residuals
        self.residual_shapes["down_block_res_6"] = (2 * self.max_batch, 1280, "H/4", "W/4")
        self.residual_shapes["down_block_res_7"] = (2 * self.max_batch, 1280, "H/4", "W/4")
        self.residual_shapes["down_block_res_8"] = (2 * self.max_batch, 1280, "H/4", "W/4")
        # Block 3 out: 1280 -> 3 residuals
        self.residual_shapes["down_block_res_9"] = (2 * self.max_batch, 1280, "H/8", "W/8")
        self.residual_shapes["down_block_res_10"] = (2 * self.max_batch, 1280, "H/8", "W/8")
        self.residual_shapes["down_block_res_11"] = (2 * self.max_batch, 1280, "H/8", "W/8")

        # Mid block
        if self.has_controlnet_mid_residual:
            mid_channels = self.block_out_channels[-1]
            self.residual_shapes["mid_block_res"] = (2 * self.max_batch, mid_channels, "H/8", "W/8")

        print(f"UNet helper configured for {self.num_controlnet_down_residuals} down residuals and mid_residual={self.has_controlnet_mid_residual}")
        # print(f"Calculated residual shapes (max batch): {self.residual_shapes}") # Reduce verbosity

    def get_input_names(self):
        names = ["sample", "timestep", "encoder_hidden_states"]
        # Add down residuals numerically
        names.extend([f"down_block_res_{i}" for i in range(self.num_controlnet_down_residuals)])
        # Add mid residual if expected
        if self.has_controlnet_mid_residual:
            names.append("mid_block_res")
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

    def get_dynamic_axes(self):
         axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B", 1: "S"},
            "latent": {0: "2B", 2: "H", 3: "W"},
         }
         # Add axes for residuals based on the pre-calculated shapes
         for name, shape_tmpl in self.residual_shapes.items():
             # shape_tmpl is like (2*max_batch, channels, H_scaled, W_scaled)
             h_expr = shape_tmpl[2] # e.g., "H", "H/8"
             w_expr = shape_tmpl[3] # e.g., "W", "W/8"
             axes[name] = {0: "2B", 2: h_expr, 3: w_expr}

         return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch, _, _, _, _,
            min_latent_height, max_latent_height, min_latent_width, max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        profile = OrderedDict({
            "sample": [
                (min_batch * 2, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size * 2, self.unet_dim, latent_height, latent_width),
                (max_batch * 2, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch * 2,), (batch_size * 2,), (max_batch * 2,)],
            "encoder_hidden_states": [
                (min_batch * 2, self.text_maxlen, self.embedding_dim),
                (batch_size * 2, self.text_maxlen, self.embedding_dim),
                (max_batch * 2, self.text_maxlen, self.embedding_dim),
            ],
        })

        # Add profiles for residuals using the pre-calculated structure
        for name, shape_tmpl in self.residual_shapes.items():
            channels = shape_tmpl[1]
            # Determine block index based on name (approximate)
            if name == "mid_block_res":
                 # --- FIX: Mid block input is after num_down_blocks-1 downsampling stages ---
                 block_idx = self.num_down_blocks - 1 # e.g., 3 for SD1.5 (H/8, W/8)
            else:
                 # Approximation: Assume 3 residuals per down block for spatial scaling
                 res_num = int(name.split('_')[-1])
                 block_idx = res_num // 3 # Block level (0, 1, 2, 3)
                 # Down block 0 (res 0,1,2) -> level 0 (H, W)
                 # Down block 1 (res 3,4,5) -> level 1 (H/2, W/2)
                 # Down block 2 (res 6,7,8) -> level 2 (H/4, W/4)
                 # Down block 3 (res 9,10,11) -> level 3 (H/8, W/8)

            # --- Use _get_spatial_dims with the corrected block_idx ---
            min_h, min_w = self._get_spatial_dims(min_latent_height, min_latent_width, block_idx)
            opt_h, opt_w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            max_h, max_w = self._get_spatial_dims(max_latent_height, max_latent_width, block_idx)
            # --- END FIX ---

            profile[name] = [
                (min_batch * 2, channels, min_h, min_w),
                (batch_size * 2, channels, opt_h, opt_w),
                (max_batch * 2, channels, max_h, max_w),
            ]

        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        B = batch_size * 2 # Factor of 2 for classifier-free guidance

        shape_dict = OrderedDict({
            "sample": (B, self.unet_dim, latent_height, latent_width),
            "timestep": (B,),
            "encoder_hidden_states": (B, self.text_maxlen, self.embedding_dim),
            "latent": (B, self.out_channels, latent_height, latent_width),
        })

        # Add residual shapes using pre-calculated structure
        for name, shape_tmpl in self.residual_shapes.items():
            channels = shape_tmpl[1]
            # Determine block index based on name (approximate)
            if name == "mid_block_res":
                 # --- FIX: Mid block input is after num_down_blocks-1 downsampling stages ---
                 block_idx = self.num_down_blocks - 1 # e.g., 3 for SD1.5 (H/8, W/8)
            else:
                 # Approximation: Assume 3 residuals per down block for spatial scaling
                 res_num = int(name.split('_')[-1])
                 block_idx = res_num // 3 # Block level (0, 1, 2, 3)

            # --- Use _get_spatial_dims with the corrected block_idx ---
            h, w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            # --- END FIX ---
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
        # Ensure timestep is float32
        inputs["timestep"] = torch.ones((B,), dtype=torch.float32, device=self.device)
        inputs["encoder_hidden_states"] = torch.randn(
            B, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device
        )

        # Generate sample residuals using pre-calculated structure
        for name, shape_tmpl in self.residual_shapes.items():
            channels = shape_tmpl[1]
            # Determine block index based on name (approximate)
            if name == "mid_block_res":
                 # --- FIX: Mid block input is after num_down_blocks-1 downsampling stages ---
                 block_idx = self.num_down_blocks - 1 # e.g., 3 for SD1.5 (H/8, W/8)
            else:
                 # Approximation: Assume 3 residuals per down block for spatial scaling
                 res_num = int(name.split('_')[-1])
                 block_idx = res_num // 3 # Block level (0, 1, 2, 3)

            # --- Use _get_spatial_dims with the corrected block_idx ---
            h, w = self._get_spatial_dims(latent_height, latent_width, block_idx)
            # --- END FIX ---
            inputs[name] = torch.randn(B, channels, h, w, dtype=dtype, device=self.device)

        return inputs

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
