Currently need to decide between using the StableDiffusionXLControlNetImg2ImgPipeline or the StableDiffusionXLPipeline.

The StableDiffusionXLControlNetImg2ImgPipeline will work best, since we're generating a frame both from the incoming video stream, the controlnet conditioning image, and the prompt.

The StableDiffusionXLPipeline will work best for generating a frame from just the prompt and the conditioning image.

Need to experiment to find out which looks better, and which is actually realtime.

# Command that works on a linux machine, as long as the nvidia-toolkit is installed:
sudo docker build -t localhost:5000/mmm-ai . && sudo docker run --rm -it -p 10295:10295 -v ./models:/app/models --device /dev/video0 --gpus all localhost:5000/mmm-ai

The above was ran after we downloaded the models and mounted them to the docker container.

> IMPORTANT: must pass --gpus all to docker run to get the GPU working.