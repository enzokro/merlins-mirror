Currently need to decide between using the StableDiffusionXLControlNetImg2ImgPipeline or the StableDiffusionXLPipeline.

The StableDiffusionXLControlNetImg2ImgPipeline will work best, since we're generating a frame both from the incoming video stream, the controlnet conditioning image, and the prompt.

The StableDiffusionXLPipeline will work best for generating a frame from just the prompt and the conditioning image.

Need to experiment to find out which looks better, and which is actually realtime.
