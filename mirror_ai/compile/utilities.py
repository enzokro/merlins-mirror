#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
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

import os
import gc
import onnx # Import onnx here
import traceback # Keep traceback for error reporting
from collections import OrderedDict
from typing import *

import numpy as np
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util


# Import BaseModel definition relative to this file's location
from .models import BaseModel, CLIP, VAE, UNet, VAEEncoder, TraceableUNetWrapper


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        # Attempt to free buffers safely
        try:
             if hasattr(self, 'buffers') and self.buffers:
                 for buf in self.buffers.values():
                     if isinstance(buf, cuda.DeviceView): # Check if it's DeviceView first
                          # DeviceViews don't own memory, don't free
                          pass
                     elif hasattr(buf, 'free'):
                          buf.free()
        except Exception as e:
             print(f"Error during Engine buffer cleanup: {e}")
        # Delete other attributes
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        config_kwargs = {}

        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        # Ensure input_profile is a list, as expected by CreateConfig profiles arg
        profiles_list = [input_profile] if input_profile else []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(
                fp16=fp16,
                refittable=enable_refit,
                profiles=profiles_list,
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Clear previous tensors
        self.tensors = OrderedDict()
        input_info = {} # Store info for inputs needed in second pass

        # --- Pass 1: Set Input Shapes ---
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                if shape_dict is None or tensor_name not in shape_dict:
                    raise ValueError(f"Missing shape for input tensor '{tensor_name}' in shape_dict.")
                
                shape = shape_dict[tensor_name]
                
                # Validate shape from shape_dict
                if not isinstance(shape, (tuple, list)) or not all(isinstance(d, int) and d > 0 for d in shape):
                    raise ValueError(f"Input shape for '{tensor_name}' must be a tuple/list of positive integers, but got: {shape}")

                # Check dimension count against engine's expectation
                engine_shape = self.engine.get_tensor_shape(tensor_name)
                if len(shape) != len(engine_shape):
                    raise ValueError(
                        f"Dimension mismatch for input '{tensor_name}'. Provided shape {shape} has {len(shape)} dims, "
                        f"but engine expects {len(engine_shape)} dims (engine shape: {engine_shape})."
                    )

                # Set the input shape in the context
                if not self.context.set_input_shape(tensor_name, shape):
                     # This check might be redundant if TRT throws internally, but good practice.
                     raise RuntimeError(f"Failed to set input shape for tensor '{tensor_name}' to {shape}.")

                # Store info for allocation pass
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                input_info[tensor_name] = {"shape": shape, "dtype": dtype}

        # --- Pass 2: Allocate All Buffers ---
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            
            if tensor_name in input_info:
                # Allocate input tensor using stored info
                shape = input_info[tensor_name]["shape"]
                dtype = input_info[tensor_name]["dtype"]
            else: # Output tensor
                # Get output shape from context *after* inputs are set
                shape = self.context.get_tensor_shape(tensor_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                # Validate output shape
                if not isinstance(shape, tuple) or not all(d > 0 for d in shape):
                     # If dynamic shapes were used, output shapes might still be dynamic (-1) if not properly resolved.
                     # Or context might be in an error state.
                     raise RuntimeError(f"Output shape for '{tensor_name}' is not concrete after setting inputs: {shape}. "
                                        "Check model/engine profile or potential context errors.")

            # Allocate the tensor
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[tensor_name] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    # More direct error message + CUDA error check
                    stream.synchronize() 
                    cuda_error = cudart.cudaGetLastError()
                    error_str = cudart.cudaGetErrorString(cuda_error[0]) if cuda_error[0] != cudart.cudaError_t.cudaSuccess else "No pending CUDA error"
                    raise ValueError(f"ERROR: TensorRT execute_async_v3 failed during CUDA graph capture pre-run. CUDA status: {cuda_error[0]} ({error_str})")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                 # More direct error message + CUDA error check
                stream.synchronize() 
                cuda_error = cudart.cudaGetLastError()
                error_str = cudart.cudaGetErrorString(cuda_error[0]) if cuda_error[0] != cudart.cudaError_t.cudaSuccess else "No pending CUDA error"
                raise ValueError(f"ERROR: TensorRT execute_async_v3 failed. CUDA status: {cuda_error[0]} ({error_str})")

        return self.tensors


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
):
    print(f"Building TensorRT engine for {model_data.name} from ONNX: {onnx_opt_path}")
    engine = Engine(engine_path) # Keep Engine instance creation

    # --- Dynamic Profile Generation ---
    # 1. Load the ONNX graph to find its actual inputs
    try:
        onnx_graph = onnx.load(onnx_opt_path, load_external_data=True)
        network_input_names = {inp.name for inp in onnx_graph.graph.input}
        print(f"  ONNX graph inputs found: {network_input_names}")
        del onnx_graph # Free memory
    except Exception as e:
        print(f"  Error loading ONNX graph {onnx_opt_path} to determine inputs: {e}")
        raise

    # 2. Get the FULL potential profile from the model_data helper
    full_profile_dict = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )
    if not full_profile_dict:
         print("Warning: model_data.get_input_profile returned empty or None.")
         full_profile_dict = {} # Prevent error below

    # 3. Create the TRT Profile, adding ONLY the inputs present in the ONNX graph
    profile = Profile()
    missing_in_profile = []
    added_to_profile = []
    for name in network_input_names:
        if name in full_profile_dict:
            dims = full_profile_dict[name]
            assert len(dims) == 3
            profile.add(name, min=dims[0], opt=dims[1], max=dims[2])
            added_to_profile.append(name)
        else:
            # This case should ideally not happen if model_data is correct,
            # but good to know if an input exists in ONNX but not the helper profile.
            missing_in_profile.append(name)
            print(f"  Warning: Input '{name}' found in ONNX graph but not in model_data profile dict. Cannot set profile shape.")

    print(f"  Created TRT profile for inputs: {added_to_profile}")
    if missing_in_profile:
         print(f"  Inputs missing profile data: {missing_in_profile}")

    # --- Original Build Logic (using the filtered profile) ---
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        activation_carveout = 4 * GiB
        max_workspace_size = free_mem - activation_carveout
    else:
        max_workspace_size = 0 # Default to 0, TRT might use a default minimum

    # Pass the dynamically created profile here
    engine.build(
        onnx_path=onnx_opt_path, # Use the correct parameter name for build
        fp16=True, # Assuming FP16 build
        input_profile=profile, # Pass the filtered profile
        enable_refit=build_enable_refit,
        enable_all_tactics=build_all_tactics,
        # timing_cache=None, # Add if using timing cache
        workspace_size=max_workspace_size,
    )
    # --- End Original Build Logic ---

    print(f"TensorRT engine build successful: {engine_path}") # Confirmation message
    return engine # Return the engine instance


def export_onnx(
    model, # Can be original model or TraceableUNetWrapper
    onnx_path: str,
    model_data: BaseModel, # This is our helper class instance (e.g., UNet from models.py)
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int = 17, # Default to a recent opset
):
    print(f"Exporting {model_data.name} to ONNX: {onnx_path}")
    if not hasattr(model_data, 'get_input_names') or not hasattr(model_data, 'get_sample_input'):
         raise AttributeError(f"{type(model_data)} must implement get_input_names() and get_sample_input()")

    # --- Device and Dtype Handling ---
    export_dtype = torch.float16 if model_data.fp16 else torch.float32 # Use model_data.fp16
    original_model_dtype = None
    # Check the dtype of the potential *original_unet* if it's a wrapper
    unet_to_check = model # Check the passed-in model directly first
    if hasattr(unet_to_check, 'dtype'):
         original_model_dtype = unet_to_check.dtype
         # Always ensure the model is float32 for ONNX export, regardless of original dtype
         if original_model_dtype != export_dtype:
             print(f"  Original model dtype is {original_model_dtype}, ensuring model is {export_dtype} for export.")
             try:
                 # Apply conversion to the main model object first
                 if model.dtype != export_dtype:
                     print(f"  Converting model ({model.dtype}) to {export_dtype} using .to()")
                     model = model.to(export_dtype)

             except Exception as e:
                 print(f"Warning: Could not change model dtype ({e}). Cloning model for export.")
                 import copy # Use copy if needed
                 model = copy.deepcopy(model).to(export_dtype) # Clone and convert

    export_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Prefer CUDA if available
    # Use a safer way to get device, checking parameters first
    try:
        original_device = next(iter(model.parameters())).device
    except StopIteration:
        # Handle models with no parameters, e.g., use buffer device or default
        try:
            original_device = next(model.buffers()).device
        except StopIteration:
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Fallback device
            print(f"Warning: Model has no parameters or buffers, assuming original device is {original_device}")

    if original_device != export_device:
        print(f"  Moving model from {original_device} to {export_device} for ONNX export.")
        # Move after potential dtype conversion
        try:
            model.to(export_device)
            # Explicitly move inner model too if wrapped
            if isinstance(model, UNet) and hasattr(model, 'original_unet'):
                 inner_device = next(model.original_unet.parameters(), None)
                 inner_device = inner_device.device if inner_device is not None else None
                 if inner_device != export_device:
                     print(f"  Moving wrapped original_unet to {export_device}")
                     model.original_unet.to(export_device)
        except Exception as e:
             print(f"Warning: Failed to move model parts to CPU for export: {e}. Exporting from {original_device}.")
             export_device = original_device # Fallback to original device if move fails

    model.eval() # Set to eval mode *after* potential modifications

    # --- Reinstate Conditional Wrapper Instantiation for UNet --- #
    export_model = model # Default to the original model
    if isinstance(model_data, UNet):
        print("  Detected UNet model type. Instantiating corrected TraceableUNetWrapper (12+1 residuals).")
        all_input_names_for_wrapper = model_data.get_input_names() # Should be 16 names
        if not all_input_names_for_wrapper or not isinstance(all_input_names_for_wrapper, list) or len(all_input_names_for_wrapper) != 16:
            raise ValueError(
                f"model_data.get_input_names() did not return a valid list of 16 names for UNet wrapper. Got: {all_input_names_for_wrapper}"
            )
        try:
            # Pass the original (potentially dtype/device converted) model to the corrected wrapper
            export_model = TraceableUNetWrapper(model, all_input_names_for_wrapper)
            export_model.to(device=export_device, dtype=export_dtype) # Ensure wrapper is on correct device/dtype
            export_model.eval()
            print(f"  Using TraceableUNetWrapper for ONNX export.")
        except Exception as wrapper_e:
            print(f"ERROR: Failed to instantiate or prepare TraceableUNetWrapper: {wrapper_e}")
            raise
    else:
        print(f"  Model type {type(model)} is not UNet, exporting directly.")
    # --- End Conditional Wrapper Instantiation ---

    # --- Prepare Inputs for Tracing --- #
    with torch.inference_mode():
        # Get the canonical list of input names for the ONNX graph
        onnx_input_names = model_data.get_input_names() # Should be 16 for UNet
        if not isinstance(onnx_input_names, list):
            raise TypeError(f"{type(model_data)}.get_input_names() must return a list.")

        # Get all sample inputs from the model_data helper
        # This should now return the correct 12+1 residuals for UNet
        sample_inputs_dict = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        if not isinstance(sample_inputs_dict, dict):
            raise TypeError(f"{type(model_data)}.get_sample_input() must return a dictionary.")

        # --- Construct the ordered input tuple for torch.onnx.export ---
        onnx_export_args_list = []
        print(f"Debug: Constructing ordered input tuple based on onnx_input_names: {onnx_input_names}")
        for name in onnx_input_names:
            if name not in sample_inputs_dict:
                # This check should pass now if models.py is correct
                raise ValueError(f"Input name '{name}' from get_input_names() not found in sample_inputs_dict keys: {list(sample_inputs_dict.keys())}")

            tensor = sample_inputs_dict[name]

            # Ensure correct dtype and device for export inputs
            expected_input_dtype = export_dtype
            if name == "input_ids": # CLIP specific
                 expected_input_dtype = torch.int32
            elif name == "timestep": # UNet specific
                 expected_input_dtype = torch.float32

            if tensor.dtype != expected_input_dtype:
                 tensor = tensor.to(expected_input_dtype)
            if tensor.device != export_device:
                 tensor = tensor.to(export_device)

            onnx_export_args_list.append(tensor)

        # Convert the final list to the required tuple format
        onnx_export_args_tuple = tuple(onnx_export_args_list)

        # --- Define ONNX Graph Outputs & Dynamic Axes (unchanged) --- #
        onnx_output_names = model_data.get_output_names()
        onnx_dynamic_axes = model_data.get_dynamic_axes()

        # Debug: Print final arguments being passed to torch.onnx.export
        print("\n--- Debug: Final arguments for torch.onnx.export ---")
        print(f"  Exporting Model Type: {type(export_model)}") # Should show Wrapper for UNet
        if hasattr(export_model, 'dtype'):
            print(f"    Export Model Dtype: {export_model.dtype}")
        # Check wrapped model if applicable
        if isinstance(export_model, TraceableUNetWrapper) and hasattr(export_model, 'original_unet'):
             print(f"    Wrapped UNet Dtype: {export_model.original_unet.dtype}")
        print(f"  Args Tuple len={len(onnx_export_args_tuple)} (Input Dtypes: {[arg.dtype for arg in onnx_export_args_tuple]}) ") # Should be 16 args
        print(f"  Input Names (ONNX Graph): {onnx_input_names}") # Should be 16 names
        print(f"  Output Names (ONNX Graph): {onnx_output_names}")
        print("-------------------------------------------------")


        # Temporary export path for pre-optimization model
        tmp_dir = os.path.abspath("onnx_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_onnx_path = os.path.join(tmp_dir, os.path.basename(onnx_path))

        # --- Export --- #
        try:
            print("Debug: Attempting torch.onnx.export...")
            torch.onnx.export(
                export_model, # Use the *wrapper* for UNet, original model otherwise
                onnx_export_args_tuple, # The single tuple with all 16 inputs for UNet
                tmp_onnx_path, # Export to temporary path first
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=onnx_input_names,
                output_names=onnx_output_names,
                dynamic_axes=onnx_dynamic_axes,
            )
            print(f"Debug: torch.onnx.export successful to temporary path: {tmp_onnx_path}")

            # --- Conditional Optimization (Applies to non-UNet models) --- #
            if not isinstance(model_data, UNet):
                print(f"Optimizing ONNX model ({type(model_data)}): {tmp_onnx_path} -> {onnx_path}")
                try:
                     onnx_graph = onnx.load(tmp_onnx_path, load_external_data=True)
                except FileNotFoundError:
                     print(f"Error: Temporary ONNX file not found at {tmp_onnx_path} after export. Cannot optimize.")
                     raise
                except Exception as e:
                     print(f"Error loading temporary ONNX {tmp_onnx_path}: {e}")
                     raise
                # Use the model_data's optimize method (e.g., for VAE, CLIP)
                onnx_opt_graph = model_data.optimize(onnx_graph)
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                # Save the *optimized* graph to the final onnx_path
                onnx.save_model(onnx_opt_graph, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{os.path.basename(onnx_path)}.data", size_threshold=1024)
                print(f"ONNX optimization and final save complete: {onnx_path}")
                del onnx_graph, onnx_opt_graph
            else:
                # --- UNet: Skip Optimization, Copy Directly --- #
                print(f"Skipping ONNX optimization for UNet. Copying wrapped+exported graph directly to: {onnx_path}")
                import shutil
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                # Copy the unoptimized graph from temp path to final path
                shutil.copyfile(tmp_onnx_path, onnx_path)
                # Copy external data file if it exists
                tmp_data_path = tmp_onnx_path + ".data"
                final_data_path = onnx_path + ".data"
                if os.path.exists(tmp_data_path):
                     try:
                         shutil.copyfile(tmp_data_path, final_data_path)
                         print(f"Copied external data file: {final_data_path}")
                     except Exception as e:
                         print(f"Warning: Failed to copy external data file {tmp_data_path} to {final_data_path}: {e}")

        except Exception as e:
             print(f"ERROR: ONNX export failed for {model_data.name}!")
             print(traceback.format_exc())
             print("\n--- Export Args Debug (on failure) ---")
             print(f"  Args Tuple (len={len(onnx_export_args_tuple)}):")
             for i, arg in enumerate(onnx_export_args_tuple):
                 input_name = onnx_input_names[i] if i < len(onnx_input_names) else f"unknown_{i}"
                 print(f"    [{i}] name: '{input_name}', type: Tensor, shape: {arg.shape}, dtype: {arg.dtype}")
             print("------------------------------------\n")
             raise

    # Move model back to original device and dtype *if* it was moved/changed
    # Check the original model's final dtype for restoration logic
    final_exported_model_dtype = export_model.dtype if hasattr(export_model, 'dtype') else None
    # Handle wrapper case for dtype check
    if isinstance(export_model, TraceableUNetWrapper):
        final_exported_model_dtype = export_model.original_unet.dtype if hasattr(export_model.original_unet, 'dtype') else None


    if original_device != export_device or (original_model_dtype is not None and final_exported_model_dtype != original_model_dtype):
        try:
            # Restore the original model passed in, not necessarily export_model
            model.to(original_device)
            if original_model_dtype is not None and model.dtype != original_model_dtype:
                 model.to(original_model_dtype)

            # Verify final state
            final_dtype = model.dtype if hasattr(model, 'dtype') else 'N/A'
            final_device = next(iter(model.parameters()), torch.tensor(0)).device # Safer device check
            print(f"  Restored original model to device {final_device} (Restored dtype: {final_dtype}).")

        except Exception as e:
            print(f"Warning: Failed to move model back to {original_device} or restore dtype: {e}")

    # Clean up references
    # del model_data # Keep model_data if needed elsewhere in the calling scope
    del sample_inputs_dict
    del onnx_export_args_list
    del onnx_export_args_tuple
    # del model # Be cautious deleting 'model' if it was passed in from outside

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if os.path.exists(tmp_dir):
        import shutil
        try:
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temporary ONNX export directory: {tmp_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory {tmp_dir}: {e}")


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel, # Need model_data for its optimize method
): # This function is now unused but kept for reference if needed
    """Optimizes an ONNX model using the model-specific optimize method."""
    print(f"Optimizing ONNX model: {onnx_path} -> {onnx_opt_path}")
    # Load the unoptimized ONNX graph
    try:
         # Consider increasing size threshold if models are large
         onnx_graph = onnx.load(onnx_path, load_external_data=True) # Assume external data might be needed
    except Exception as e:
         print(f"Error loading ONNX file {onnx_path}: {e}")
         raise

    # Use the model_data's specific optimize method
    onnx_opt_graph = model_data.optimize(onnx_graph)

    # Save the optimized graph
    try:
        # Save with external data if graph size is large
        onnx.save_model(onnx_opt_graph, onnx_opt_path, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{os.path.basename(onnx_opt_path)}.data", size_threshold=1024) # Adjust threshold as needed
    except ValueError as ve:
         if "larger than 2GB" in str(ve):
              print("Optimized model still > 2GB after external data attempt. Saving might fail elsewhere.")
              # Save without external data as fallback (might error)
              onnx.save(onnx_opt_graph, onnx_opt_path)
         else: raise # Re-raise other ValueErrors
    except Exception as e:
        print(f"Error saving optimized ONNX file {onnx_opt_path}: {e}")
        raise

    print(f"ONNX optimization complete: {onnx_opt_path}")
    del onnx_opt_graph
    del model_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
