import os
import gc
import torch
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline
from polygraphy import cuda
from streamdiffusion.acceleration.tensorrt import (
    TorchVAEEncoder,
    compile_unet,
    compile_vae_decoder,
    compile_vae_encoder,
)
from streamdiffusion.acceleration.tensorrt.engine import (
    AutoencoderKLEngine,
    UNet2DConditionModelEngine,
)
from streamdiffusion.acceleration.tensorrt.models import (
    VAE,
    UNet,
    VAEEncoder,
)

def optimize_controlnet_pipeline_with_tensorrt(
    pipeline,
    engine_dir="./trt_engines",
    max_batch_size=2,
    min_batch_size=1,
    use_cuda_graph=False,
):
    # Create directory for engines
    os.makedirs(engine_dir, exist_ok=True)
    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Extract the models
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    vae = pipeline.vae
    
    # Create engine paths
    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"
    
    # Save configs for later reattachment
    vae_config = vae.config
    vae_dtype = vae.dtype
    
    # Move models to CPU to free GPU memory
    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create model specifications
    unet_model = UNet(
        fp16=True,
        device=pipeline.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
    vae_decoder_model = VAE(
        device=pipeline.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=pipeline.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    
    # Compile UNet if not already compiled
    if not os.path.exists(unet_engine_path):
        print(f"Compiling UNet to TensorRT... This may take a while.")
        compile_unet(
            unet,
            unet_model,
            f"{onnx_dir}/unet.onnx",
            f"{onnx_dir}/unet.opt.onnx",
            unet_engine_path,
            opt_batch_size=max_batch_size,
        )
        # Delete the PyTorch model to free memory
        del unet

    # Compile VAE decoder if not already compiled
    if not os.path.exists(vae_decoder_engine_path):
        print(f"Compiling VAE decoder to TensorRT...")
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            f"{onnx_dir}/vae_decoder.onnx",
            f"{onnx_dir}/vae_decoder.opt.onnx",
            vae_decoder_engine_path,
            opt_batch_size=max_batch_size,
        )

    # Compile VAE encoder if not already compiled
    if not os.path.exists(vae_encoder_engine_path):
        print(f"Compiling VAE encoder to TensorRT...")
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            f"{onnx_dir}/vae_encoder.onnx",
            f"{onnx_dir}/vae_encoder.opt.onnx",
            vae_encoder_engine_path,
            opt_batch_size=max_batch_size,
        )
        del vae_encoder

    # Clean up original VAE
    del vae
    
    # Create CUDA stream for inference
    cuda_stream = cuda.Stream()
    
    # Create TensorRT engine wrappers and replace original models
    pipeline.unet = UNet2DConditionModelEngine(
        unet_engine_path, 
        cuda_stream, 
        use_cuda_graph=use_cuda_graph
    )
    
    pipeline.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        pipeline.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    
    # Restore needed attributes
    setattr(pipeline.vae, "config", vae_config)
    setattr(pipeline.vae, "dtype", vae_dtype)
    
    # Free remaining memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print("TensorRT acceleration enabled.")
    return pipeline

# Usage example:
def load_optimized_pipeline():
    # Load the base pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Create engine directory path
    engine_dir = "./trt_engines"
    
    # Check if engines exist - if so, load them directly
    if (os.path.exists(f"{engine_dir}/unet.engine") and 
        os.path.exists(f"{engine_dir}/vae_encoder.engine") and 
        os.path.exists(f"{engine_dir}/vae_decoder.engine")):
        
        print("Loading pre-compiled TensorRT engines...")
        
        # Save configs for later reattachment
        vae_config = pipeline.vae.config
        vae_dtype = pipeline.vae.dtype
        
        # Create CUDA stream for inference
        cuda_stream = cuda.Stream()
        
        # Create TensorRT engine wrappers and replace original models
        pipeline.unet = UNet2DConditionModelEngine(
            f"{engine_dir}/unet.engine", 
            cuda_stream, 
            use_cuda_graph=False
        )
        
        pipeline.vae = AutoencoderKLEngine(
            f"{engine_dir}/vae_encoder.engine",
            f"{engine_dir}/vae_decoder.engine",
            cuda_stream,
            pipeline.vae_scale_factor,
            use_cuda_graph=False,
        )
        
        # Restore needed attributes
        setattr(pipeline.vae, "config", vae_config)
        setattr(pipeline.vae, "dtype", vae_dtype)
        
        print("TensorRT engines loaded successfully.")
    else:
        # Compile the engines as they don't exist
        pipeline = optimize_controlnet_pipeline_with_tensorrt(pipeline)
    
    return pipeline