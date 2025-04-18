import os
import gc
import torch
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline
from polygraphy import cuda
from . import (
    TorchVAEEncoder,
    compile_unet,
    compile_vae_decoder,
    compile_vae_encoder,
)
from .engine import (
    AutoencoderKLEngine,
    UNet2DConditionModelEngine,
)
from .models import (
    VAE,
    UNet,
    VAEEncoder,
)

# Define target dtype for compilation/export explicitly
TARGET_EXPORT_DTYPE = torch.float16

def optimize_controlnet_pipeline_with_tensorrt(
    pipeline,
    name,
    engine_dir="./trt_engines",
    max_batch_size=2,
    min_batch_size=1,
    use_cuda_graph=False,
):
    engine_dir = Path(engine_dir) / name
    # Create directory for engines
    os.makedirs(engine_dir, exist_ok=True)
    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    print(f"Ensuring pipeline components are on device: {pipeline.device} and dtype: {TARGET_EXPORT_DTYPE}")
    pipeline_components_converted = {}
    try:
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
             if pipeline.unet.dtype != TARGET_EXPORT_DTYPE:
                 print(f"  Converting UNet from {pipeline.unet.dtype} to {TARGET_EXPORT_DTYPE}")
                 pipeline.unet.to(dtype=TARGET_EXPORT_DTYPE)
                 pipeline_components_converted['unet'] = True
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
             if pipeline.vae.dtype != TARGET_EXPORT_DTYPE:
                 print(f"  Converting VAE from {pipeline.vae.dtype} to {TARGET_EXPORT_DTYPE}")
                 pipeline.vae.to(dtype=TARGET_EXPORT_DTYPE)
                 pipeline_components_converted['vae'] = True
        # ControlNet conversion (if applicable and needed)
        if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            if isinstance(pipeline.controlnet, torch.nn.ModuleList): # Handle multiple controlnets
                for i, cn in enumerate(pipeline.controlnet):
                    if cn.dtype != TARGET_EXPORT_DTYPE:
                         print(f"  Converting ControlNet {i} from {cn.dtype} to {TARGET_EXPORT_DTYPE}")
                         cn.to(dtype=TARGET_EXPORT_DTYPE)
                         pipeline_components_converted[f'controlnet_{i}'] = True
            elif isinstance(pipeline.controlnet, torch.nn.Module): # Handle single controlnet
                 if pipeline.controlnet.dtype != TARGET_EXPORT_DTYPE:
                     print(f"  Converting ControlNet from {pipeline.controlnet.dtype} to {TARGET_EXPORT_DTYPE}")
                     pipeline.controlnet.to(dtype=TARGET_EXPORT_DTYPE)
                     pipeline_components_converted['controlnet'] = True

        if not pipeline_components_converted:
            print("  Pipeline components already in target export dtype.")

    except Exception as e:
        print(f"Warning: Failed during explicit dtype conversion: {e}")
        # Proceed cautiously, the dtype issue might persist

    # Extract the models (now hopefully in TARGET_EXPORT_DTYPE)
    text_encoder = pipeline.text_encoder # Usually float16 is fine for text encoder
    unet = pipeline.unet
    vae = pipeline.vae
    # ControlNet might be a single model or a list
    controlnet = pipeline.controlnet if hasattr(pipeline, 'controlnet') else None

    # Create engine paths
    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"
    
    # Save configs for later reattachment
    unet_config = unet.config # Save unet config
    unet_dtype = unet.dtype
    vae_config = vae.config
    # Store the dtype it *should* be after conversion for VAE restoration
    vae_dtype_final = vae.dtype

    # --- Move models to CPU to free GPU memory - Handled within export_onnx ---
    # print("Moving models to CPU temporarily...")
    # original_devices = {}
    # if unet: original_devices['unet'] = unet.device; unet.to('cpu')
    # if vae: original_devices['vae'] = vae.device; vae.to('cpu')
    # if controlnet:
    #      if isinstance(controlnet, torch.nn.ModuleList):
    #          original_devices['controlnet'] = [cn.device for cn in controlnet]
    #          for cn in controlnet: cn.to('cpu')
    #      elif isinstance(controlnet, torch.nn.Module):
    #          original_devices['controlnet'] = controlnet.device
    #          controlnet.to('cpu')
    # gc.collect()
    # torch.cuda.empty_cache()

    # sanity logging
    print(f"Compiling {name} with TensorRT...")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create model specifications
    print(f"Creating unet model spec for {name} (Engine Target: FP16)")
    unet_model_spec = UNet(
        fp16=True, # Target TensorRT engine precision
        device=pipeline.device, # Target device for spec (doesn't affect export)
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet_config.in_channels, # Use saved config
    )
    print(f"Creating vae decoder model spec for {name}...")
    vae_decoder_model_spec = VAE(
        device=pipeline.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    print(f"Creating vae encoder model spec for {name}...")
    vae_encoder_model_spec = VAEEncoder(
        device=pipeline.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    
    # Compile UNet if not already compiled
    if not os.path.exists(unet_engine_path):
        print(f"Compiling UNet (source dtype: {unet.dtype}) to TensorRT FP16 Engine...")
        # Pass the unet here
        compile_unet(
            unet,
            unet_model_spec, # The spec defining engine properties
            f"{onnx_dir}/unet.onnx",
            f"{onnx_dir}/unet.opt.onnx", # This path might not be used anymore if optimize_onnx is skipped
            unet_engine_path,
            opt_batch_size=max_batch_size,
        )
        # Delete the PyTorch model to free memory after compilation if it's no longer needed
        # Keep it if we might load pre-compiled engines later and need its config/dtype
        if os.path.exists(vae_decoder_engine_path) and os.path.exists(vae_encoder_engine_path):
             if 'unet' in locals(): del unet; gc.collect(); torch.cuda.empty_cache()
    else:
         print(f"UNet engine found at {unet_engine_path}, skipping compilation.")

    # Compile VAE decoder if not already compiled
    if not os.path.exists(vae_decoder_engine_path):
        print(f"Compiling VAE decoder (source dtype: {vae.dtype}) to TensorRT...")
        # VAE should already be float32 from the top conversion
        vae.forward = vae.decode # Set the forward method for export
        compile_vae_decoder(
            vae,
            vae_decoder_model_spec,
            f"{onnx_dir}/vae_decoder.onnx",
            f"{onnx_dir}/vae_decoder.opt.onnx", # Opt path may be unused
            vae_decoder_engine_path,
            opt_batch_size=max_batch_size,
        )
        gc.collect()
        torch.cuda.empty_cache()
    else:
         print(f"VAE Decoder engine found at {vae_decoder_engine_path}, skipping compilation.")

    # Compile VAE encoder if not already compiled
    if not os.path.exists(vae_encoder_engine_path):
        print(f"Compiling VAE encoder (source dtype: {vae.dtype}) to TensorRT...")
        # Wrap the VAE
        vae_encoder_torch = TorchVAEEncoder(vae).to(pipeline.device) # Keep on target device
        compile_vae_encoder(
            vae_encoder_torch, # Pass the wrapper
            vae_encoder_model_spec,
            f"{onnx_dir}/vae_encoder.onnx",
            f"{onnx_dir}/vae_encoder.opt.onnx", # Opt path may be unused
            vae_encoder_engine_path,
            opt_batch_size=max_batch_size,
        )
        del vae_encoder_torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
         print(f"VAE Encoder engine found at {vae_encoder_engine_path}, skipping compilation.")

    # Clean up original VAE only if we compiled both parts or don't need it later
    if not os.path.exists(vae_decoder_engine_path) or not os.path.exists(vae_encoder_engine_path):
         print("Keeping VAE reference as not all parts were compiled/found.")
    else:
         # Delete VAE only if UNet was also handled (compiled or found)
         if os.path.exists(unet_engine_path):
             if 'vae' in locals():
                 del vae
                 gc.collect()
                 torch.cuda.empty_cache()

    # --- Move models back to original device --- Handled by export_onnx cleanup

    # Create CUDA stream for inference
    cuda_stream = cuda.Stream()
    
    # --- Create TensorRT engine wrappers and replace original models ---
    print("Replacing pipeline components with TensorRT Engines.")
    # Load UNet engine wrapper
    pipeline.unet = UNet2DConditionModelEngine(
        unet_engine_path,
        cuda_stream,
        use_cuda_graph=use_cuda_graph
    )
    # Restore UNet config
    setattr(pipeline.unet, "dtype", unet_dtype)
    setattr(pipeline.unet, "config", unet_config)
    # Note: We don't set dtype on the engine wrapper, it operates internally

    # Load VAE engine wrapper
    pipeline.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        pipeline.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    # Restore VAE config and the dtype it had *after* the explicit conversion
    setattr(pipeline.vae, "config", vae_config)
    setattr(pipeline.vae, "dtype", vae_dtype_final) # Restore the dtype it should have now


    # Keep ControlNet as PyTorch for now (TensorRT compilation not implemented here)
    # Ensure ControlNet is on the correct device if it wasn't deleted
    if controlnet:
        print("Keeping ControlNet as PyTorch model.")
        pipeline.controlnet = controlnet # Reassign potentially modified (dtype) controlnet
        if isinstance(pipeline.controlnet, torch.nn.ModuleList):
            for i, cn in enumerate(pipeline.controlnet):
                 # Move back to original device if tracked, otherwise pipeline.device
                 # device = original_devices['controlnet'][i] if 'controlnet' in original_devices else pipeline.device
                 cn.to(pipeline.device) # Simplified: move to pipeline device
                 # Ensure correct dtype (should be TARGET_EXPORT_DTYPE)
                 if cn.dtype != TARGET_EXPORT_DTYPE: cn.to(dtype=TARGET_EXPORT_DTYPE)

        elif isinstance(pipeline.controlnet, torch.nn.Module):
            # device = original_devices['controlnet'] if 'controlnet' in original_devices else pipeline.device
            pipeline.controlnet.to(pipeline.device) # Simplified: move to pipeline device
            if pipeline.controlnet.dtype != TARGET_EXPORT_DTYPE: pipeline.controlnet.to(dtype=TARGET_EXPORT_DTYPE)

    # Free remaining memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print("TensorRT acceleration enabled for UNet and VAE.")
    return pipeline

# Usage example:
def load_optimized_pipeline(pipeline=None, name="", engine_dir="./trt_engines"):
    if pipeline is None:
        # Load the base pipeline - Use DTYPE from config if possible
        try:
            from mirror_ai.config import DTYPE as CONFIG_DTYPE
        except ImportError:
            print("Warning: Could not import DTYPE from config, defaulting to float16 for initial load.")
            CONFIG_DTYPE = torch.float16 # Default if config isn't available

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", # Adjust base model if needed
            torch_dtype=CONFIG_DTYPE
        )
        pipeline = pipeline.to("cuda") # Adjust device if needed

    # Define engine paths
    engine_base_path = Path(engine_dir) / name
    unet_engine_file = engine_base_path / "unet.engine"
    vae_encoder_engine_file = engine_base_path / "vae_encoder.engine"
    vae_decoder_engine_file = engine_base_path / "vae_decoder.engine"

    # Check if engines exist
    if (unet_engine_file.exists() and
        vae_encoder_engine_file.exists() and
        vae_decoder_engine_file.exists()):

        print(f"Loading pre-compiled TensorRT engines from {engine_base_path}...")

        # Save configs/dtype for later reattachment
        unet_config = pipeline.unet.config
        unet_dtype = pipeline.unet.dtype
        vae_config = pipeline.vae.config
        vae_dtype_original = pipeline.vae.dtype # Dtype before replacement
        vae_scale_factor = pipeline.vae_scale_factor

        # Free PyTorch models before loading engines
        del pipeline.unet
        del pipeline.vae
        gc.collect()
        torch.cuda.empty_cache()

        # Create CUDA stream for inference
        cuda_stream = cuda.Stream()

        # Create TensorRT engine wrappers and replace original models
        pipeline.unet = UNet2DConditionModelEngine(
            str(unet_engine_file),
            cuda_stream,
            use_cuda_graph=False # Default use_cuda_graph to False for loading
        )

        pipeline.vae = AutoencoderKLEngine(
            str(vae_encoder_engine_file),
            str(vae_decoder_engine_file),
            cuda_stream,
            vae_scale_factor,
            use_cuda_graph=False, # Default use_cuda_graph to False for loading
        )

        # Restore needed attributes
        setattr(pipeline.unet, "config", unet_config)
        setattr(pipeline.unet, "dtype", unet_dtype)
        setattr(pipeline.vae, "config", vae_config)
        setattr(pipeline.vae, "dtype", vae_dtype_original)

        print("TensorRT engines loaded successfully.")
    else:
        # Compile the engines as they don't exist
        print(f"TensorRT engines not found in {engine_base_path}, compiling...")
        pipeline = optimize_controlnet_pipeline_with_tensorrt(pipeline, name=name, engine_dir=engine_dir)

    return pipeline