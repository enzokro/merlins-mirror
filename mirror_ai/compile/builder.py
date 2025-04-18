import gc
import os
from typing import *

import torch

from .models import BaseModel, UNet, TraceableUNetWrapper
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


# class EngineBuilder:
#     def __init__(
#         self,
#         model: BaseModel,
#         network: Any,
#         device=torch.device("cuda"),
#     ):
#         self.device = device

#         self.model = model
#         self.network = network

#     def build(
#         self,
#         onnx_path: str,
#         onnx_opt_path: str,
#         engine_path: str,
#         opt_image_height: int = 512,
#         opt_image_width: int = 512,
#         opt_batch_size: int = 1,
#         min_image_resolution: int = 256,
#         max_image_resolution: int = 1024,
#         build_enable_refit: bool = False,
#         build_static_batch: bool = False,
#         build_dynamic_shape: bool = False,
#         build_all_tactics: bool = False,
#         onnx_opset: int = 17,
#         force_engine_build: bool = False,
#         force_onnx_export: bool = False,
#         force_onnx_optimize: bool = False,
#     ):
#         if not force_onnx_export and os.path.exists(onnx_path):
#             print(f"Found cached model: {onnx_path}")
#         else:
#             print(f"Exporting model: {onnx_path}")

#             export_model = self.network
#             if isinstance(self.model, UNet):
#                 print("UNet model detected, using TraceableUNetWrapper for export.")
#                 all_input_names = self.model.get_input_names()
#                 export_model = TraceableUNetWrapper(self.network, all_input_names)
#                 export_model.to(self.device)
#                 export_model.eval()
#             else:
#                 print(f"Model type {type(self.network)} detected, exporting directly.")

#             export_onnx(
#                 export_model,
#                 onnx_path=onnx_path,
#                 model_data=self.model,
#                 opt_image_height=opt_image_height,
#                 opt_image_width=opt_image_width,
#                 opt_batch_size=opt_batch_size,
#                 onnx_opset=onnx_opset,
#             )
#             del self.network
#             gc.collect()
#             torch.cuda.empty_cache()
#         if not force_onnx_optimize and os.path.exists(onnx_opt_path):
#             print(f"Found cached model: {onnx_opt_path}")
#         else:
#             print(f"Skipping separate optimize_onnx step. Using {onnx_path} for engine build.")
#             onnx_opt_path = onnx_path # Point build_engine to the unoptimized ONNX

#         self.model.min_latent_shape = min_image_resolution // 8
#         self.model.max_latent_shape = max_image_resolution // 8
#         if not force_engine_build and os.path.exists(engine_path):
#             print(f"Found cached engine: {engine_path}")
#         else:
#             build_engine(
#                 engine_path=engine_path,
#                 onnx_opt_path=onnx_opt_path,
#                 model_data=self.model,
#                 opt_image_height=opt_image_height,
#                 opt_image_width=opt_image_width,
#                 opt_batch_size=opt_batch_size,
#                 build_static_batch=build_static_batch,
#                 build_dynamic_shape=build_dynamic_shape,
#                 build_all_tactics=build_all_tactics,
#                 build_enable_refit=build_enable_refit,
#             )

#         gc.collect()
#         torch.cuda.empty_cache()


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network # This is the actual torch model (e.g., Diffusers UNet)

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str, # Keep param, but might just point to onnx_path
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = False,
        build_all_tactics: bool = False,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False, # Keep for potential future use
    ):
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")

            # REMOVED TraceableUNetWrapper logic
            # Export the raw network directly
            export_model = self.network
            print(f"Model type {type(self.network)} detected, exporting directly.")

            export_onnx(
                export_model,
                onnx_path=onnx_path,
                model_data=self.model, # Use the simplified UNet model spec
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            # Consider deleting self.network ONLY if VAE/other parts are also compiled/handled
            # For now, keep it to avoid deleting prematurely
            # del self.network
            gc.collect()
            torch.cuda.empty_cache()

        # Check for optimized path; if optimize step is skipped, use base path
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached optimized model: {onnx_opt_path}")
            onnx_build_path = onnx_opt_path
        else:
            # Logic within export_onnx now handles optional optimization.
            # If optimization was skipped, onnx_path contains the graph to build from.
            # If optimization happened, onnx_path might point to the optimized graph OR
            # the optimized graph might be saved separately depending on export_onnx logic.
            # Assume export_onnx saves the final graph (opt or not) to onnx_path
            print(f"Using ONNX graph at {onnx_path} for engine build.")
            onnx_build_path = onnx_path


        # Update min/max latent shapes based on image resolution limits
        self.model.min_latent_shape = min_image_resolution // 8
        self.model.max_latent_shape = max_image_resolution // 8

        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_build_path, # Use the path determined above
                model_data=self.model, # Use the simplified UNet model spec
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
            )

        gc.collect()
        torch.cuda.empty_cache()