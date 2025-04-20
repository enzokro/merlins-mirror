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


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
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
        force_onnx_optimize: bool = False,
    ):
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")

            export_model = self.network
            print(f"Model type {type(self.network)} detected, exporting directly.")

            export_onnx(
                export_model,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            gc.collect()
            torch.cuda.empty_cache()

        onnx_build_path = onnx_path

        if os.path.exists(onnx_opt_path) and onnx_opt_path != onnx_path:
             if force_onnx_optimize:
                 print(f"Using externally optimized ONNX graph: {onnx_opt_path}")
                 onnx_build_path = onnx_opt_path
             else:
                 print(f"Found optimized path {onnx_opt_path}, but using base {onnx_path} as optimization not forced/relevant.")
        else:
             print(f"Using ONNX graph at {onnx_build_path} for engine build.")

        if hasattr(self.model, 'min_latent_shape'):
             self.model.min_latent_shape = min_image_resolution // 8
             self.model.max_latent_shape = max_image_resolution // 8

        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            print(f"Building engine from: {onnx_build_path}")
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_build_path,
                model_data=self.model,
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