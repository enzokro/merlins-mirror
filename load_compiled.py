import os
from mirror_ai.pipeline_dream_pcm_sd15 import ImagePipeline
from mirror_ai.compile.go_compile import load_optimized_pipeline
from dotenv import load_dotenv

# load the environment variables
load_dotenv()

print("Loading compiled Merlin...")
engine_dir = os.getenv("COMPILED_MODELS_DIR")
print(f"Engine directory: {engine_dir}")

# set up the image transformer and the webcam stream
image_pipeline = ImagePipeline()
pipeline = image_pipeline.load("ddim")

# # compile the pipeline
compiled_name = "dream_pcm_sd15"
# print(f"Compiling {ImagePipeline} as {compiled_name} in {engine_dir}")
# pipeline = optimize_controlnet_pipeline_with_tensorrt(pipeline, name=compiled_name, engine_dir=engine_dir)

# NOTE: name and engine_dir must be set to the same values as the compile_pipeline call above
optimized_pipeline = load_optimized_pipeline(pipeline=pipeline, name=compiled_name, engine_dir=engine_dir)

print("Loaded compiled Merlin!")