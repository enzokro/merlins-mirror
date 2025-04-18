import time
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

class MyControlNetPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
        t_index_list=[0],  # Timestep indices for StreamBatch
        use_denoising_batch=True,  # Enable StreamBatch processing
        frame_buffer_size=1,  # Number of frames to process simultaneously
    ):
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

        
        # StreamBatch-specific parameters
        self.t_list = t_index_list
        self.denoising_steps_num = len(self.t_list)
        self.frame_bff_size = frame_buffer_size
        self.use_denoising_batch = use_denoising_batch
        self.latent_buffer_initialized = False
        
        # Configure batch sizes for StreamBatch
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size
            
        # Track inference time with EMA
        self.inference_time_ema = 0
    
    def prepare(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=2.5,
        generator=None,
        device=None,
        eta=0.0, # Added eta parameter for DDIM variance calculation
    ):
        """Initialize the StreamBatch pipeline with precalculated parameters for DDIM."""
        if device is None:
            device = self.device
        dtype = self.unet.dtype # Use consistent dtype

        # Create buffers for intermediate latent states if needed
        if self.denoising_steps_num > 1:
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

        # Store guidance scale
        self.guidance_scale = guidance_scale
        self.eta = eta # Store eta

        # Encode prompt and prepare for conditioning
        # NOTE: Ensure encode_prompt handles batching correctly for frame_bff_size
        prompt_embeds_uc, prompt_embeds_c = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=self.frame_bff_size, # Generate embeds per frame
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
        )
        self.negative_prompt_embeds = prompt_embeds_uc # Store uncond embeds separately
        self.prompt_embeds = prompt_embeds_c        # Store cond embeds separately

        # Setup the timesteps based on num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps.to(device)

        # Extract the specific timesteps we'll use based on t_list indices
        self.sub_timesteps = []
        for t_index in self.t_list:
            # Ensure t_index is within the bounds of the calculated timesteps
            clamped_index = min(t_index, len(self.timesteps) - 1)
            self.sub_timesteps.append(self.timesteps[clamped_index])

        # --- Pre-calculate DDIM scheduler parameters ---
        sub_timesteps_tensor_base = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=device
        )

        # Get corresponding previous timesteps
        prev_timesteps_list = []
        for t in self.sub_timesteps:
            # Need tensor input for previous_timestep
            prev_t = self.scheduler.previous_timestep(t)
            prev_timesteps_list.append(prev_t)
        prev_timesteps_tensor_base = torch.tensor(
            prev_timesteps_list, dtype=torch.long, device=device
        )

        # Calculate alphas and betas for current sub_timesteps
        alphas_cumprod_t = self.scheduler.alphas_cumprod[sub_timesteps_tensor_base]
        alpha_prod_t_sqrt_base = alphas_cumprod_t.sqrt()
        beta_prod_t_sqrt_base = (1 - alphas_cumprod_t).sqrt()

        # Calculate alphas for previous timesteps
        # Handle potential edge case where prev_timestep might be -1 or out of bounds if t=0
        alphas_cumprod_prev = torch.where(
            prev_timesteps_tensor_base >= 0,
            self.scheduler.alphas_cumprod[prev_timesteps_tensor_base],
            self.scheduler.final_alpha_cumprod # Use final alpha if prev_t < 0
        )
        sqrt_alpha_prod_prev_base = alphas_cumprod_prev.sqrt()
        sqrt_one_minus_alpha_prod_prev_base = (1 - alphas_cumprod_prev).sqrt()

        # Calculate variance (sigma_t^2) using scheduler's variance calculation
        variance_base = self.scheduler._get_variance(
            sub_timesteps_tensor_base, prev_timesteps_tensor_base
        ) * (self.eta**2) # Scale variance by eta^2

        # Reshape base tensors for broadcasting
        reshape_dims = (len(self.t_list), 1, 1, 1)
        alpha_prod_t_sqrt_base = alpha_prod_t_sqrt_base.view(reshape_dims).to(dtype=dtype)
        beta_prod_t_sqrt_base = beta_prod_t_sqrt_base.view(reshape_dims).to(dtype=dtype)
        sqrt_alpha_prod_prev_base = sqrt_alpha_prod_prev_base.view(reshape_dims).to(dtype=dtype)
        sqrt_one_minus_alpha_prod_prev_base = sqrt_one_minus_alpha_prod_prev_base.view(reshape_dims).to(dtype=dtype)
        variance_base = variance_base.view(reshape_dims).to(dtype=dtype)

        # Repeat/Interleave tensors for the full batch based on frame_bff_size and use_denoising_batch
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

        # Pre-calculate values needed specifically for the buffer update step (timesteps t_list[1:])
        if self.denoising_steps_num > 1:
            self.alpha_prod_t_sqrt_buffer_target = self.alpha_prod_t_sqrt[repeats:]
            self.beta_prod_t_sqrt_buffer_target = self.beta_prod_t_sqrt[repeats:]
        else:
            self.alpha_prod_t_sqrt_buffer_target = None
            self.beta_prod_t_sqrt_buffer_target = None

        # Initialize random noise using torch.randn
        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            device=device, dtype=dtype, # Use consistent dtype
            generator=generator # Use the provided generator for reproducibility
        )

        # stock_noise is potentially unused if not doing RCFG-like techniques
        # self.stock_noise = torch.zeros_like(self.init_noise)

        # Mark as initialized
        self.latent_buffer_initialized = True

    ## NOTE: old for LCM and turbo
    # def prepare(
    #     self,
    #     prompt,
    #     negative_prompt="",
    #     num_inference_steps=4,
    #     guidance_scale=2.5,
    #     generator=None,
    #     device=None
    # ):
    #     """Initialize the StreamBatch pipeline with precalculated parameters."""
    #     if device is None:
    #         device = self.device
            
    #     # Create buffers for intermediate latent states if needed
    #     if self.denoising_steps_num > 1:
    #         self.x_t_latent_buffer = torch.zeros(
    #             (
    #                 (self.denoising_steps_num - 1) * self.frame_bff_size,
    #                 4,
    #                 self.latent_height,
    #                 self.latent_width,
    #             ),
    #             dtype=self.dtype,
    #             device=device,
    #         )
    #     else:
    #         self.x_t_latent_buffer = None
            
    #     # Store guidance scale
    #     self.guidance_scale = guidance_scale
            
    #     # Encode prompt and prepare for conditioning
    #     self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
    #         prompt=prompt,
    #         device=device,
    #         num_images_per_prompt=self.frame_bff_size,
    #         do_classifier_free_guidance=guidance_scale > 1.0,
    #         negative_prompt=negative_prompt,
    #     )
        
    #     # Create the correct prompt embeddings shape for batched processing
    #     if guidance_scale > 1.0:
    #         self.prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)
            
    #     # Setup the timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     self.timesteps = self.scheduler.timesteps.to(device)
        
    #     # Extract the specific timesteps we'll use for StreamBatch
    #     self.sub_timesteps = []
    #     for t in self.t_list:
    #         self.sub_timesteps.append(self.timesteps[t])
            
    #     # Create tensors of timesteps for batched processing
    #     sub_timesteps_tensor = torch.tensor(
    #         self.sub_timesteps, dtype=torch.long, device=device
    #     )
    #     self.sub_timesteps_tensor = torch.repeat_interleave(
    #         sub_timesteps_tensor,
    #         repeats=self.frame_bff_size if self.use_denoising_batch else 1,
    #         dim=0,
    #     )
        
    #     # Pre-compute scheduler parameters for efficiency
    #     # Pre-calculate c_skip and c_out for scheduler
    #     c_skip_list = []
    #     c_out_list = []
    #     for timestep in self.sub_timesteps:
    #         c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
    #             timestep
    #         )
    #         c_skip_list.append(c_skip)
    #         c_out_list.append(c_out)
            
    #     self.c_skip = (
    #         torch.stack(c_skip_list)
    #         .view(len(self.t_list), 1, 1, 1)
    #         .to(dtype=self.unet.dtype, device=device)
    #     )
    #     self.c_out = (
    #         torch.stack(c_out_list)
    #         .view(len(self.t_list), 1, 1, 1)
    #         .to(dtype=self.unet.dtype, device=device)
    #     )
        
    #     # Pre-calculate alpha and beta for the scheduler
    #     alpha_prod_t_sqrt_list = []
    #     beta_prod_t_sqrt_list = []
    #     for timestep in self.sub_timesteps:
    #         alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
    #         beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
    #         alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
    #         beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
            
    #     alpha_prod_t_sqrt = (
    #         torch.stack(alpha_prod_t_sqrt_list)
    #         .view(len(self.t_list), 1, 1, 1)
    #         .to(dtype=self.unet.dtype, device=device)
    #     )
    #     beta_prod_t_sqrt = (
    #         torch.stack(beta_prod_t_sqrt_list)
    #         .view(len(self.t_list), 1, 1, 1)
    #         .to(dtype=self.unet.dtype, device=device)
    #     )
        
    #     self.alpha_prod_t_sqrt = torch.repeat_interleave(
    #         alpha_prod_t_sqrt,
    #         repeats=self.frame_bff_size if self.use_denoising_batch else 1,
    #         dim=0,
    #     )
    #     self.beta_prod_t_sqrt = torch.repeat_interleave(
    #         beta_prod_t_sqrt,
    #         repeats=self.frame_bff_size if self.use_denoising_batch else 1,
    #         dim=0,
    #     )
        
    #     # Initialize random noise
    #     self.init_noise = torch.zeros(
    #         (self.batch_size, 4, self.latent_height, self.latent_width),
    #         device=device, dtype=self.unet.dtype
    #     )
        
    #     # Used for noise residuals in RCFG (not fully utilized in this version)
    #     self.stock_noise = torch.zeros_like(self.init_noise)
        
    #     # Mark as initialized
    #     self.latent_buffer_initialized = True


    @property
    def latent_height(self):
        return self.unet.config.sample_size // self.vae_scale_factor
        
    @property
    def latent_width(self):
        return self.unet.config.sample_size // self.vae_scale_factor
        
    def add_noise(self, original_samples, noise, t_index):
        """Add noise to samples at specified timestep index"""
        timestep = self.sub_timesteps[t_index]
        alpha_cumprod = self.scheduler.alphas_cumprod[timestep]
        sqrt_alpha_prod = alpha_cumprod ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
    ## NOTE: For LCM
    # def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch):
    #     """Apply scheduler step to a batch of predictions efficiently"""
    #     # Extract noise prediction 
    #     noise_pred = model_pred_batch
        
    #     # If using CFG, separate and combine unconditional and conditional
    #     if self.guidance_scale > 1.0:
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
    #     # Apply the c_skip and c_out precalculated factors
    #     # This is a batched version of the scheduler's step function
    #     if self.use_denoising_batch:
    #         c_skip = torch.repeat_interleave(self.c_skip, repeats=self.frame_bff_size, dim=0)
    #         c_out = torch.repeat_interleave(self.c_out, repeats=self.frame_bff_size, dim=0)
    #     else:
    #         # Handle single step case
    #         c_skip = self.c_skip
    #         c_out = self.c_out
            
    #     # Calculate the predicted original sample x_0
    #     pred_original_sample = c_skip * x_t_latent_batch + c_out * noise_pred
        
    #     # Get previous sample x_t-1
    #     variance = 0  # simplified for this implementation
    #     x_prev_batch = pred_original_sample
        
    #     return x_prev_batch

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
        # Ensure pre-calculated tensors match the batch dimension of inputs
        # These tensors should already be shaped correctly in 'prepare' based on use_denoising_batch etc.
        sqrt_alpha_prod_t = self.alpha_prod_t_sqrt
        sqrt_one_minus_alpha_prod_t = self.beta_prod_t_sqrt
        sqrt_alpha_prod_prev = self.sqrt_alpha_prod_prev # Pre-calculated in prepare
        sqrt_one_minus_alpha_prod_prev = self.sqrt_one_minus_alpha_prod_prev # Pre-calculated in prepare
        variance = self.variance # Pre-calculated in prepare (will be 0 for eta=0)

        # 1. Calculate predicted original sample (x_0)
        # Formula: (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
        pred_original_sample = (x_t_latent_batch - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        # 2. Calculate coefficients for Equation (12) of DDIM paper
        # mean component: sqrt(alpha_{t-1}) * x0
        pred_original_sample_coeff = sqrt_alpha_prod_prev

        # direction pointing to x_t component: sqrt(1 - alpha_{t-1} - variance) * epsilon
        pred_epsilon_coeff = torch.sqrt(1.0 - sqrt_alpha_prod_prev**2 - variance) # Re-calculate based on pre-calculated parts

        # 3. Calculate x_{t-1} sample
        prev_sample_mean = pred_original_sample_coeff * pred_original_sample
        prev_sample_dir = pred_epsilon_coeff * model_output

        # 4. Add variance (noise) - only if variance > 0 (i.e., eta > 0)
        if variance.sum() > 0: # Check if any variance is non-zero
             # Need noise. Reusing init_noise rolled/shifted might be complex.
             # Generating fresh noise is simpler but less deterministic if generator not managed carefully.
             # Let's use fresh noise for now. Ensure generator state is handled externally if needed.
             variance_noise = torch.randn_like(model_output, device=model_output.device, dtype=model_output.dtype)
             prev_sample_noise = torch.sqrt(variance) * variance_noise
        else:
             prev_sample_noise = torch.zeros_like(model_output) # No noise for eta=0

        prev_sample = prev_sample_mean + prev_sample_dir + prev_sample_noise

        return prev_sample, pred_original_sample

    def predict_x0_batch(self, x_t_latent, cond_embeddings=None, controlnet_cond=None, controlnet_conditioning_scale=1.0):
        """Core function for Stream Batch processing - processes all timesteps in parallel"""
        # Create batch from current latent and buffer
        if self.denoising_steps_num == 1:
            # Single-step case (like SD-Turbo)
            x_t_latent_batch = x_t_latent
        else:
            # Multi-step case with buffer
            if self.frame_bff_size == 1:
                # Single frame case
                x_t_latent_batch = torch.cat([x_t_latent, self.x_t_latent_buffer], dim=0)
            else:
                # Multiple frame case (reshape for proper batching)
                x_t_latent_batch = torch.cat(
                    [
                        x_t_latent,
                        self.x_t_latent_buffer.reshape(
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
                        ),
                    ],
                    dim=0,
                )
        
        # Process through UNet with ControlNet support
        do_classifier_free_guidance = self.guidance_scale > 1.0
        
        # Prepare inputs for the ControlNet forward pass
        timesteps = self.sub_timesteps_tensor
        
        # Scale the inputs according to the scheduler
        latent_model_input = x_t_latent_batch
        
        # --- Start: Apply CFG ---
        do_classifier_free_guidance = self.guidance_scale > 1.0
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([x_t_latent_batch] * 2)
            timesteps_batch = torch.cat([self.sub_timesteps_tensor] * 2)
            prompt_embeds_batch = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])
            # Handle controlnet_cond doubling if needed
            if controlnet_cond is not None:
                 controlnet_cond_batch = torch.cat([controlnet_cond] * 2)
            else:
                 controlnet_cond_batch = None
        else:
            latent_model_input = x_t_latent_batch
            timesteps_batch = self.sub_timesteps_tensor
            prompt_embeds_batch = self.prompt_embeds # Use prompt_embeds prepared in prepare()
            controlnet_cond_batch = controlnet_cond
        # --- End: Apply CFG ---

        # --- ControlNet and UNet Calls ---
        # Pass the doubled inputs if CFG is used
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            timesteps_batch,
            encoder_hidden_states=prompt_embeds_batch,
            controlnet_cond=controlnet_cond_batch, # Use doubled cond
            conditioning_scale=controlnet_conditioning_scale, # This might need doubling logic too? Check base pipeline
            return_dict=False,
        )

        model_pred_raw = self.unet(
            latent_model_input,
            timesteps_batch,
            encoder_hidden_states=prompt_embeds_batch,
            down_block_additional_residuals=down_block_res_samples, # Check if residuals need doubling based on CFG
            mid_block_additional_residual=mid_block_res_sample,   # Check if residuals need doubling based on CFG
            return_dict=False,
        )[0]

        # Perform guidance *after* UNet call
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_pred_raw.chunk(2)
            model_output_guided = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_output_guided = model_pred_raw
        # --- End ControlNet and UNet Calls ---


        # Call the revised DDIM scheduler step batch
        # It receives the already guided noise prediction
        prev_samples_batch, pred_original_samples_batch = self.scheduler_step_batch(
            model_output_guided,
            x_t_latent_batch # Input noisy latents for this batch
        )

        # --- Buffer Update Logic ---
        if self.denoising_steps_num > 1:
            # We need the predicted x0 corresponding to the steps that will *feed* the buffer
            # These are the first (denoising_steps_num - 1) steps of the current batch.
            # If CFG, model_output_guided has double batch size. pred_original_samples_batch
            # should also have double batch size. Need to extract the guided part.
            if do_classifier_free_guidance:
                 # Assuming the guided part is the second half after chunking in UNet output
                 _, x0_for_buffer_calc = pred_original_samples_batch.chunk(2)
            else:
                 x0_for_buffer_calc = pred_original_samples_batch

            # Select the x0 predictions needed to generate the buffer content
            # These correspond to the inputs t_k to t_{N-1}
            x0_for_buffer = x0_for_buffer_calc[:-self.frame_bff_size]

            # Get the noise to add. Use the self.init_noise buffer, selecting parts
            # corresponding to the *target* timesteps for the buffer (t_{k+1} to t_N).
            # This requires careful indexing based on how init_noise is structured/rolled.
            # A simple approach matching StreamDiffusion's apparent logic:
            noise_for_buffer = self.init_noise[self.frame_bff_size:] # Noise for steps t_list[1:]

            # Get the alpha_prod_t_sqrt and beta_prod_t_sqrt for the *target* timesteps (t_{k+1} to t_N).
            # These should correspond to the indices [1:] of the original t_list.
            # Ensure these are pre-calculated and stored with correct batch repeats.
            # Example: self.alpha_prod_t_sqrt_buffer_target = precalculated sqrt(alpha) for steps t_list[1:] repeated N times
            alpha_sqrt_next = self.alpha_prod_t_sqrt_buffer_target # Need to pre-calculate this
            beta_sqrt_next = self.beta_prod_t_sqrt_buffer_target   # Need to pre-calculate this

            # Add noise using the forward process formula: x_t = sqrt(alpha_t)*x0 + sqrt(1-alpha_t)*noise
            self.x_t_latent_buffer = (alpha_sqrt_next * x0_for_buffer + beta_sqrt_next * noise_for_buffer).detach()

            # Optional: Roll the init_noise buffer if using a persistent noise pattern
            # self.init_noise = torch.roll(self.init_noise, shifts=(-self.frame_bff_size), dims=0)

        # --- End Buffer Update ---

        # Return the final denoised latent (x_{t-1}) for the *last* step(s) of the current frame
        # If CFG, prev_samples_batch might be doubled. Extract guided part.
        if do_classifier_free_guidance:
            _, final_output_latents_guided = prev_samples_batch.chunk(2)
        else:
            final_output_latents_guided = prev_samples_batch

        # Select the samples corresponding to the last step(s)
        final_output_latents = final_output_latents_guided[-self.frame_bff_size:].detach()
        return final_output_latents
    
        ### NOTE: below for turbo and LCM
        # # Handle classifier-free guidance by duplicating for uncond/cond
        # if do_classifier_free_guidance:
        #     latent_model_input = torch.cat([latent_model_input] * 2)
        #     timesteps = torch.cat([timesteps] * 2)
            
        # # Process through ControlNet
        # down_block_res_samples, mid_block_res_sample = self.controlnet(
        #     latent_model_input,
        #     timesteps,
        #     encoder_hidden_states=self.prompt_embeds,
        #     controlnet_cond=controlnet_cond,
        #     conditioning_scale=controlnet_conditioning_scale,
        #     return_dict=False,
        # )
        
        # # Process through UNet with ControlNet residuals
        # model_pred_batch = self.unet(
        #     latent_model_input,
        #     timesteps,
        #     encoder_hidden_states=self.prompt_embeds,
        #     down_block_additional_residuals=down_block_res_samples,
        #     mid_block_additional_residual=mid_block_res_sample,
        #     return_dict=False,
        # )[0]
        
        # # Apply scheduler step to entire batch
        # x_0_pred_batch = self.scheduler_step_batch(model_pred_batch, x_t_latent_batch)
        
        # # Update internal buffer for subsequent calls
        # if self.denoising_steps_num > 1:
        #     if self.frame_bff_size == 1:
        #         # For single frame buffer
        #         self.x_t_latent_buffer = x_0_pred_batch[:-self.frame_bff_size].detach()
        #     else:
        #         # For multi-frame buffer (reshape as needed)
        #         self.x_t_latent_buffer = x_0_pred_batch[:-self.frame_bff_size].reshape(
        #             self.frame_bff_size,
        #             self.denoising_steps_num - 1,
        #             4,
        #             self.latent_height,
        #             self.latent_width,
        #         ).transpose(0, 1).reshape(
        #             (self.denoising_steps_num - 1) * self.frame_bff_size,
        #             4,
        #             self.latent_height,
        #             self.latent_width,
        #         ).detach()
                
        # # Return the final output (last element of the batch)
        # return x_0_pred_batch[-self.frame_bff_size:].detach()
        
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        image=None,
        height=None,
        width=None,
        num_inference_steps=4,
        guidance_scale=2.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        controlnet_conditioning_scale=1.0,
        **kwargs,
    ):
        # Initialize if needed
        device = self.device
        
        # 1. Check if we need to initialize the pipeline or if it's already ready
        if not self.latent_buffer_initialized:
            # Use the height/width from inputs or defaults
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            
            # Calculate latent dimensions
            self.latent_height = height // self.vae_scale_factor
            self.latent_width = width // self.vae_scale_factor
            
            # Initialize stream batch pipeline
            self.prepare(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                device=device
            )
        
        # 2. Preprocess input image for conditioning
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
        
        # 3. Create initial latents if not provided
        if latents is None:
            shape = (
                self.frame_bff_size,
                4,
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
        
        # 4. Run StreamBatch prediction to get denoised latents
        start_time = time.time()
        x_0_pred = self.predict_x0_batch(
            latents,
            controlnet_cond=controlnet_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        inference_time = time.time() - start_time
        
        # Update the EMA time tracker
        alpha = 0.8
        self.inference_time_ema = alpha * self.inference_time_ema + (1 - alpha) * inference_time
        
        # 5. Post-process to get final image
        if output_type != "latent":
            # Decode the image
            x_0_pred = x_0_pred / self.vae.config.scaling_factor
            image = self.vae.decode(x_0_pred, return_dict=False)[0]
            
            # Convert to proper output format
            image, has_nsfw_concept = self.run_safety_checker(image, device, self.unet.dtype)
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = x_0_pred
            has_nsfw_concept = None
        
        # 6. Return results
        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)