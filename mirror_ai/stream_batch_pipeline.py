import time
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Tuple, Optional, Union, List
# from diffusers.models import AutoencoderKL, CLIPTextModel, CLIPTokenizer, UNet2DConditionModel, ControlNetModel, MultiControlNetModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

class StreamBatchPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        # Standard StableDiffusionControlNetPipeline arguments first
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        # Custom arguments are removed from signature, will be set later
        # num_inference_steps: int = 4,
        # use_denoising_batch=True,
        # frame_buffer_size=1,
    ):
        # --- DEBUG PRINTS START ---
        print(f"[DEBUG StreamBatchPipeline.__init__] Received args types:")
        print(f"  - unet: {type(unet)}")
        print(f"  - feature_extractor: {type(feature_extractor)}")
        print(f"  - text_encoder: {type(text_encoder)}")
        print(f"  - scheduler: {type(scheduler)}")
        print(f"  - safety_checker: {type(safety_checker)}")
        print(f"  - image_encoder: {type(image_encoder)}")
        print(f"  - controlnet: {type(controlnet)}")
        print(f"  - vae: {type(vae)}")
        print(f"  - tokenizer: {type(tokenizer)}")
        print(f"  - requires_safety_checker: {type(requires_safety_checker)}")
        print(f"[DEBUG StreamBatchPipeline.__init__] Calling super().__init__...")
        # --- DEBUG PRINTS END ---

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        # Initialize custom attributes with default values or placeholders
        # These will be properly set after loading in the main script
        self.denoising_steps_num = None # Placeholder
        self.frame_bff_size = None # Placeholder
        self.use_denoising_batch = None # Placeholder
        self.latent_buffer_initialized = False
        self.batch_size = None # Placeholder
        self.trt_unet_batch_size = None # Placeholder
        self.inference_time_ema = 0
        self.latent_height = None # Placeholder
        self.latent_width = None # Placeholder
        # NOTE: The logic depending on use_denoising_batch and frame_buffer_size
        # for setting batch_size and trt_unet_batch_size needs to be moved
        # to where these attributes are set AFTER loading in pipeline_dream_pcm_sd15.py

    def set_stream_batch_options(self, num_inference_steps: int, frame_buffer_size: int, use_denoising_batch: bool):
        """Helper method to set custom options after initialization."""
        print(f"[DEBUG StreamBatchPipeline.set_stream_batch_options] Setting options...")
        self.denoising_steps_num = num_inference_steps
        self.frame_bff_size = frame_buffer_size
        self.use_denoising_batch = use_denoising_batch
        self.latent_buffer_initialized = False # Reset initialization status

        # Configure batch sizes for StreamBatch
        if self.use_denoising_batch:
            self.batch_size = self.denoising_steps_num * self.frame_bff_size
            self.trt_unet_batch_size = self.batch_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = self.frame_bff_size
        print(f"[DEBUG StreamBatchPipeline.set_stream_batch_options] Options set: batch_size={self.batch_size}")

    def prepare(
        self,
        prompt,
        negative_prompt="",
        guidance_scale=2.5,
        generator=None,
        device=None,
        eta=0.0, # Added eta parameter for DDIM variance calculation
    ):
        """Initialize the StreamBatch pipeline with precalculated parameters for DDIM."""
        if device is None:
            device = self.device
        dtype = self.unet.dtype # Use consistent dtype

        # Ensure scheduler tensors are on the correct device
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device)

        # Create buffers for intermediate latent states if needed
        if self.denoising_steps_num > 1:
            if not hasattr(self, 'latent_height') or not hasattr(self, 'latent_width'):
                 raise ValueError("Latent dimensions (height/width) must be set before calling prepare.")

            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4, # Latent channels
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=dtype, # Use consistent dtype
                device=device,
            )
        else:
            self.x_t_latent_buffer = None

        # Store guidance scale and eta
        self.guidance_scale = guidance_scale
        self.eta = eta # Store eta

        # Encode prompt and prepare for conditioning
        prompt_embeds_uc, prompt_embeds_c = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=self.frame_bff_size, # Generate embeds per frame
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
        )
        self.negative_prompt_embeds = prompt_embeds_uc # Store uncond embeds separately
        self.prompt_embeds = prompt_embeds_c        # Store cond embeds separately

        # Setup the timesteps based on self.denoising_steps_num and scheduler config
        self.scheduler.set_timesteps(self.denoising_steps_num, device=device)
        self.inference_timesteps = self.scheduler.timesteps # Already on device by set_timesteps
        if len(self.inference_timesteps) != self.denoising_steps_num:
             raise ValueError(f"Scheduler generated {len(self.inference_timesteps)} timesteps, but expected {self.denoising_steps_num}.")

        # --- Pre-calculate DDIM scheduler parameters based on actual inference_timesteps ---
        sub_timesteps_tensor_base = self.inference_timesteps.clone() # Already on device

        prev_timesteps_list = []
        step_ratio = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        for i, t in enumerate(self.inference_timesteps): # t is on device
            prev_t_calculated = t - step_ratio
            # Ensure the constant tensor is also on the device
            prev_t = max(prev_t_calculated, torch.tensor(-1, device=device, dtype=torch.long))
            prev_timesteps_list.append(prev_t)

        prev_timesteps_tensor_base = torch.stack(prev_timesteps_list).to(dtype=torch.long, device=device) # Explicitly on device

        # Indexing should work now as alphas_cumprod and indices are on the same device
        alphas_cumprod_t = self.scheduler.alphas_cumprod[sub_timesteps_tensor_base] # Removed .to(...)
        alpha_prod_t_sqrt_base = alphas_cumprod_t.sqrt()
        beta_prod_t_sqrt_base = (1 - alphas_cumprod_t).sqrt()

        alphas_cumprod_prev = torch.where(
            prev_timesteps_tensor_base >= 0,
            self.scheduler.alphas_cumprod[prev_timesteps_tensor_base], # Removed .to(...)
            # Ensure the fallback tensor is also on the correct device and dtype
            torch.tensor(self.scheduler.final_alpha_cumprod, device=device, dtype=dtype)
        )
        sqrt_alpha_prod_prev_base = alphas_cumprod_prev.sqrt()
        sqrt_one_minus_alpha_prod_prev_base = (1 - alphas_cumprod_prev).sqrt()

        # Replicate variance calculation from scheduler._get_variance element-wise
        beta_prod_t = 1 - alphas_cumprod_t
        beta_prod_t_prev = 1 - alphas_cumprod_prev
        variance = (beta_prod_t_prev / beta_prod_t.clamp(min=1e-6)) * (1 - alphas_cumprod_t / alphas_cumprod_prev.clamp(min=1e-6))
        variance_base = variance * (self.eta**2)

        reshape_dims = (self.denoising_steps_num, 1, 1, 1)
        # Ensure final pre-calculated tensors have the correct dtype
        alpha_prod_t_sqrt_base = alpha_prod_t_sqrt_base.view(reshape_dims).to(dtype=dtype)
        beta_prod_t_sqrt_base = beta_prod_t_sqrt_base.view(reshape_dims).to(dtype=dtype)
        sqrt_alpha_prod_prev_base = sqrt_alpha_prod_prev_base.view(reshape_dims).to(dtype=dtype)
        sqrt_one_minus_alpha_prod_prev_base = sqrt_one_minus_alpha_prod_prev_base.view(reshape_dims).to(dtype=dtype)
        variance_base = variance_base.view(reshape_dims).to(dtype=dtype)

        repeats = self.frame_bff_size if self.use_denoising_batch else 1

        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor_base, repeats=repeats, dim=0
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt_base, repeats=repeats, dim=0
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt_base, repeats=repeats, dim=0
        )
        self.sqrt_alpha_prod_prev = torch.repeat_interleave(
            sqrt_alpha_prod_prev_base, repeats=repeats, dim=0
        )
        self.sqrt_one_minus_alpha_prod_prev = torch.repeat_interleave(
            sqrt_one_minus_alpha_prod_prev_base, repeats=repeats, dim=0
        )
        self.variance = torch.repeat_interleave(
            variance_base, repeats=repeats, dim=0
        )

        if self.denoising_steps_num > 1:
            self.alpha_prod_t_sqrt_buffer_target = self.alpha_prod_t_sqrt[repeats:]
            self.beta_prod_t_sqrt_buffer_target = self.beta_prod_t_sqrt[repeats:]
        else:
            self.alpha_prod_t_sqrt_buffer_target = None
            self.beta_prod_t_sqrt_buffer_target = None

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            device=device, dtype=dtype,
            generator=generator
        )

        # stock_noise is potentially unused if not doing RCFG-like techniques
        # self.stock_noise = torch.zeros_like(self.init_noise)

        self.latent_buffer_initialized = True

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, value):
        self._guidance_scale = value
        
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
             timestep = torch.tensor(timestep, device=original_samples.device, dtype=torch.long)
        elif timestep.numel() != 1:
             raise ValueError("timestep must be a single value for add_noise")

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)

        if timestep < 0 or timestep >= len(alphas_cumprod):
             raise ValueError(f"Invalid timestep {timestep} provided to add_noise.")

        alpha_cumprod_t = alphas_cumprod[timestep]
        sqrt_alpha_prod = alpha_cumprod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod_t) ** 0.5

        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, *([1] * (original_samples.ndim - 1)))
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, *([1] * (original_samples.ndim - 1)))

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


    def scheduler_step_batch(self, model_output: torch.Tensor, x_t_latent_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the DDIM scheduler step logic to a batch of latents and noise predictions,
        using pre-calculated values.

        Args:
            model_output: The CFG-guided noise prediction batch (epsilon_theta).
                          Shape: (batch_size, channels, height, width)
            x_t_latent_batch: The input noisy latent batch at timestep t.
                             Shape: (batch_size, channels, height, width)

        Returns:
            A tuple containing:
            - prev_sample (torch.Tensor): The denoised latent batch at timestep t-1 (x_{t-1}).
            - pred_original_sample (torch.Tensor): The predicted original sample batch (x_0).
        """
        # NOTE: Removed check for expected_batch_size as model_output and x_t_latent_batch
        # should consistently have self.batch_size here.
        # expected_batch_size = self.batch_size * (2 if self.guidance_scale > 1.0 else 1)
        # if model_output.shape[0] != expected_batch_size or x_t_latent_batch.shape[0] != expected_batch_size:
        #      pass # Consider raising an error if shapes mismatch expectations

        # Use the pre-calculated tensors directly (they have shape [self.batch_size, 1, 1, 1])
        sqrt_alpha_prod_t = self.alpha_prod_t_sqrt
        sqrt_one_minus_alpha_prod_t = self.beta_prod_t_sqrt
        sqrt_alpha_prod_prev = self.sqrt_alpha_prod_prev
        sqrt_one_minus_alpha_prod_prev = self.sqrt_one_minus_alpha_prod_prev
        variance = self.variance

        # ---- REMOVE THIS BLOCK ----
        # if self.guidance_scale > 1.0 and sqrt_alpha_prod_t.shape[0] == self.batch_size:
        #      sqrt_alpha_prod_t = torch.cat([sqrt_alpha_prod_t] * 2)
        #      sqrt_one_minus_alpha_prod_t = torch.cat([sqrt_one_minus_alpha_prod_t] * 2)
        #      sqrt_alpha_prod_prev = torch.cat([sqrt_alpha_prod_prev] * 2)
        #      sqrt_one_minus_alpha_prod_prev = torch.cat([sqrt_one_minus_alpha_prod_prev] * 2)
        #      variance = torch.cat([variance] * 2)
        # ---- END REMOVAL ----

        # Shapes should now match:
        # x_t_latent_batch: [B, C, H, W]
        # model_output: [B, C, H, W]
        # sqrt_one_minus_alpha_prod_t: [B, 1, 1, 1]
        # sqrt_alpha_prod_t: [B, 1, 1, 1]
        pred_original_sample = (x_t_latent_batch - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        pred_original_sample_coeff = sqrt_alpha_prod_prev # Shape [B, 1, 1, 1]

        variance_clamped = torch.clamp(variance, min=0.0) # Shape [B, 1, 1, 1]
        # sqrt_alpha_prod_prev shape: [B, 1, 1, 1]
        pred_epsilon_coeff = torch.sqrt(1.0 - sqrt_alpha_prod_prev**2 - variance_clamped) # Shape [B, 1, 1, 1]

        prev_sample_mean = pred_original_sample_coeff * pred_original_sample # Shape [B, C, H, W]
        prev_sample_dir = pred_epsilon_coeff * model_output # Shape [B, C, H, W]

        # variance check needs adjusted logic if variance had shape [2*B, ...] before removal
        # Assuming variance now has shape [B, 1, 1, 1]
        if variance.sum() > 1e-6: # Check if any variance exists across the batch
             variance_noise = torch.randn_like(model_output, device=model_output.device, dtype=model_output.dtype) # Shape [B, C, H, W]
             prev_sample_noise = torch.sqrt(variance_clamped) * variance_noise # Shape [B, C, H, W]
        else:
             prev_sample_noise = torch.zeros_like(model_output) # Shape [B, C, H, W]

        prev_sample = prev_sample_mean + prev_sample_dir + prev_sample_noise # Shape [B, C, H, W]

        # pred_original_sample also has shape [B, C, H, W]
        return prev_sample, pred_original_sample

    def predict_x0_batch(self, x_t_latent, cond_embeddings=None, controlnet_cond=None, controlnet_conditioning_scale=1.0):
        """Core function for Stream Batch processing - processes all timesteps in parallel"""
        do_classifier_free_guidance = self.guidance_scale > 1.0

        if self.denoising_steps_num == 1:
            x_t_latent_batch = x_t_latent
        else:
            if self.frame_bff_size == 1:
                x_t_latent_batch = torch.cat([x_t_latent, self.x_t_latent_buffer], dim=0)
            else:
                buffer_reshaped = self.x_t_latent_buffer.reshape(
                    self.denoising_steps_num - 1,
                    self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ).transpose(0, 1).reshape(
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                )
                x_t_latent_batch = torch.cat([x_t_latent, buffer_reshaped], dim=0)
        
        timesteps_batch = self.sub_timesteps_tensor

        latent_model_input = self.scheduler.scale_model_input(x_t_latent_batch, timesteps_batch)

        if do_classifier_free_guidance:
            latent_model_input_cfg = torch.cat([latent_model_input] * 2)
            timesteps_batch_cfg = torch.cat([timesteps_batch] * 2)
            if self.negative_prompt_embeds.shape[0] != self.frame_bff_size or self.prompt_embeds.shape[0] != self.frame_bff_size:
                 neg_embeds_repeated = self.negative_prompt_embeds.repeat_interleave(self.denoising_steps_num, dim=0)
                 pos_embeds_repeated = self.prompt_embeds.repeat_interleave(self.denoising_steps_num, dim=0)
            else:
                 neg_embeds_repeated = self.negative_prompt_embeds.repeat_interleave(self.denoising_steps_num, dim=0)
                 pos_embeds_repeated = self.prompt_embeds.repeat_interleave(self.denoising_steps_num, dim=0)

            prompt_embeds_batch_cfg = torch.cat([neg_embeds_repeated, pos_embeds_repeated])

            if controlnet_cond is not None:
                 controlnet_cond_repeated = controlnet_cond.repeat_interleave(self.denoising_steps_num, dim=0)
                 controlnet_cond_batch_cfg = torch.cat([controlnet_cond_repeated] * 2)
            else:
                 controlnet_cond_batch_cfg = None
        else:
            latent_model_input_cfg = latent_model_input
            timesteps_batch_cfg = timesteps_batch
            prompt_embeds_batch_cfg = self.prompt_embeds.repeat_interleave(self.denoising_steps_num, dim=0)
            if controlnet_cond is not None:
                controlnet_cond_batch_cfg = controlnet_cond.repeat_interleave(self.denoising_steps_num, dim=0)
            else:
                controlnet_cond_batch_cfg = None

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input_cfg,
            timesteps_batch_cfg,
            encoder_hidden_states=prompt_embeds_batch_cfg,
            controlnet_cond=controlnet_cond_batch_cfg,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )

        model_pred_raw = self.unet(
            latent_model_input_cfg,
            timesteps_batch_cfg,
            encoder_hidden_states=prompt_embeds_batch_cfg,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_pred_raw.chunk(2)
            model_output_guided = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_output_guided = model_pred_raw

        prev_samples_batch, pred_original_samples_batch = self.scheduler_step_batch(
            model_output_guided,
            x_t_latent_batch
        )

        if self.denoising_steps_num > 1:
            if do_classifier_free_guidance:
                 if pred_original_samples_batch.shape[0] == self.batch_size * 2:
                      _, x0_guided = pred_original_samples_batch.chunk(2)
                 elif pred_original_samples_batch.shape[0] == self.batch_size:
                      x0_guided = pred_original_samples_batch
                 else:
                      raise ValueError("Unexpected shape for pred_original_samples_batch in buffer update.")
            else:
                 x0_guided = pred_original_samples_batch

            num_elements_for_buffer = (self.denoising_steps_num - 1) * self.frame_bff_size
            x0_for_buffer = x0_guided[:num_elements_for_buffer]

            noise_for_buffer = self.init_noise[self.frame_bff_size:]

            alpha_sqrt_next = self.alpha_prod_t_sqrt_buffer_target
            beta_sqrt_next = self.beta_prod_t_sqrt_buffer_target

            if x0_for_buffer.shape != noise_for_buffer.shape or \
               x0_for_buffer.shape[0] != alpha_sqrt_next.shape[0] * self.frame_bff_size or \
               x0_for_buffer.shape[0] != beta_sqrt_next.shape[0] * self.frame_bff_size:
                 repeats = self.frame_bff_size if self.use_denoising_batch else 1
                 alpha_sqrt_next_r = alpha_sqrt_next
                 beta_sqrt_next_r = beta_sqrt_next

                 alpha_sqrt_next_r = self.alpha_prod_t_sqrt_buffer_target
                 beta_sqrt_next_r = self.beta_prod_t_sqrt_buffer_target

            else:
                 alpha_sqrt_next_r = alpha_sqrt_next
                 beta_sqrt_next_r = beta_sqrt_next

            self.x_t_latent_buffer = (alpha_sqrt_next_r * x0_for_buffer + beta_sqrt_next_r * noise_for_buffer).detach()

        if do_classifier_free_guidance:
            if prev_samples_batch.shape[0] == self.batch_size * 2:
                 _, final_output_latents_guided = prev_samples_batch.chunk(2)
            elif prev_samples_batch.shape[0] == self.batch_size:
                 final_output_latents_guided = prev_samples_batch
            else:
                 raise ValueError("Unexpected shape for prev_samples_batch in final output.")
        else:
            final_output_latents_guided = prev_samples_batch

        final_output_latents = final_output_latents_guided[-self.frame_bff_size:].detach()
        return final_output_latents
    
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 2.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        controlnet_conditioning_scale: float = 1.0,
        **kwargs,
    ):
        device = self.device
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.latent_height = height // self.vae_scale_factor
        self.latent_width = width // self.vae_scale_factor

        if num_images_per_prompt != self.frame_bff_size:
             print(f"Warning: num_images_per_prompt ({num_images_per_prompt}) differs from frame_buffer_size ({self.frame_bff_size}). Using frame_buffer_size.")

        if not self.latent_buffer_initialized or kwargs.get("force_prepare", False):
            if prompt is None and prompt_embeds is None:
                 raise ValueError("Must provide `prompt` or `prompt_embeds`")
            if prompt is not None and not isinstance(prompt, (str, list)):
                 raise TypeError(f"`prompt` must be a string or list of strings, but got {type(prompt)}")
            if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
                 raise TypeError(f"`negative_prompt` must be a string or list of strings, but got {type(negative_prompt)}")

            self.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                device=device,
                eta=eta
            )

        if image is None:
            raise ValueError("ControlNet condition `image` cannot be None for MyControlNetPipeline")

        controlnet_image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=self.frame_bff_size,
            num_images_per_prompt=1,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=guidance_scale > 1.0,
        )

        if latents is None:
            shape = (
                self.frame_bff_size,
                self.unet.config.in_channels,
                self.latent_height,
                self.latent_width
            )
            latents = torch.randn(
                shape, 
                generator=generator, 
                device=device, 
                dtype=self.unet.dtype
            )
            latents = latents * self.scheduler.init_noise_sigma
        elif latents.shape[0] != self.frame_bff_size:
             raise ValueError(f"Provided latents batch size ({latents.shape[0]}) must match frame_buffer_size ({self.frame_bff_size})")

        start_time = time.time()
        x_0_pred_latents = self.predict_x0_batch(
            x_t_latent=latents,
            controlnet_cond=controlnet_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        inference_time = time.time() - start_time

        alpha = 0.8
        self.inference_time_ema = alpha * self.inference_time_ema + (1 - alpha) * inference_time

        if output_type == "latent":
            image = x_0_pred_latents
            has_nsfw_concept = None
        else:
            x_0_pred_latents = x_0_pred_latents / self.vae.config.scaling_factor
            image = self.vae.decode(x_0_pred_latents, return_dict=False)[0]

            image, has_nsfw_concept = self.run_safety_checker(image, device, self.unet.dtype)

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)