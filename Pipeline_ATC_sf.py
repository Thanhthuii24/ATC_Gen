import torch
import inspect
from torch import nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Callable, Dict, List, Optional, Union


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion1DPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: nn.Module,
        scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor = None,
        signal_length: int = 512,
        num_inference_steps: int = 50,
        generator=None,
        return_dict: bool = True,
        output_type: str = "np",
        timesteps: List[int] = None,
        sigmas: List[float] = None,
    ):
        device = self._execution_device
        batch_size = latents.shape[0] if latents is not None else 1

        latent_channels = (
            self.unet.config.get("in_channels", 1)
            if isinstance(self.unet.config, dict)
            else getattr(self.unet.config, "in_channels", 1)
        )

        if latents is None:
            latents = randn_tensor(
                (batch_size, latent_channels, signal_length),
                generator=generator,
                device=device,
            )
        else:
            latents = latents.to(device)

        latents *= self.scheduler.init_noise_sigma

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
                model_input = self.scheduler.scale_model_input(latents, t)
                noise_pred = self.unet(model_input, t_batch)[0]
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "np":
            signal = latents.squeeze(1).detach().cpu().numpy()
        else:
            signal = latents

        if return_dict:
            return {"signals": signal}
        return signal
