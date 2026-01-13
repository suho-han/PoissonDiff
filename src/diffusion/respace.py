import numpy as np
import torch as th

from src.diffusion.binomial_diffusion import BinomialDiffusion
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.diffusion.poisson_diffusion import PoissonDiffusion
from src.diffusion.prior_binomial_diffusion import PriorBinomialDiffusion
from src.diffusion.prior_poisson_diffusion import PriorPoissonDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddimuni"):
            desired_count = int(section_counts[len("ddimuni"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
        if section_counts.startswith("ddimqua"):
            desired_count = int(section_counts[len("ddimqua"):])
            seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
            return set([int(s) for s in list(seq)])
        raise ValueError(
            f"cannot create exactly {num_timesteps} steps with an integer stride"
        )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class _SpacedDiffusionMixin:
    """Shared logic for spaced diffusion variants."""

    def __init__(self, use_timesteps, **kwargs):
        if "betas" not in kwargs:
            raise ValueError("Spaced diffusion requires beta schedules.")

        betas = np.array(kwargs["betas"], dtype=np.float64)
        if betas.ndim != 1:
            raise ValueError("Betas must be a 1-D sequence.")

        self.use_timesteps = set(use_timesteps)
        if not self.use_timesteps:
            raise ValueError("use_timesteps cannot be empty.")
        self.timestep_map = []
        self.original_num_steps = len(betas)

        last_alpha_cumprod = 1.0
        new_betas = []
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        spaced_kwargs = dict(kwargs)
        spaced_kwargs["betas"] = np.array(new_betas, dtype=np.float64)
        super().__init__(**spaced_kwargs)

    def p_mean(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class BinomialSpacedDiffusion(_SpacedDiffusionMixin, BinomialDiffusion):
    """Spaced diffusion for binomial-style processes."""


class PriorBinomialSpacedDiffusion(_SpacedDiffusionMixin, PriorBinomialDiffusion):
    """Spaced diffusion for binomial-style processes."""


class GaussianSpacedDiffusion(_SpacedDiffusionMixin, GaussianDiffusion):
    """Spaced diffusion for gaussian processes."""


class PoissonSpacedDiffusion(_SpacedDiffusionMixin, PoissonDiffusion):
    """Spaced diffusion for poisson-style processes."""


class PriorPoissonSpacedDiffusion(_SpacedDiffusionMixin, PriorPoissonDiffusion):
    """Spaced diffusion for poisson-style processes."""


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
