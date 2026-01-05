import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
Xingchao Liu, Chengyue Gong, Qiang Liu
https://arxiv.org/abs/2209.03003
'''


class PoissonFlow(nn.Module):
    def __init__(self, num_timesteps=100, t_min=0.0, t_max=1.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.t_min = t_min
        self.t_max = t_max

    def get_time_embedding(self, t_batch, x_shape):
        return t_batch.view(-1, *([1] * (len(x_shape) - 1)))

    def training_losses(self, model, x_1, t, model_kwargs=None):
        """
        Rectified Flow Loss: Eq(2) in Paper
        L = || v_theta(X_t, t) - (X_1 - X_0) ||^2
        """
        # 1. Sample x_0 and t
        x_0 = model_kwargs["prior"]
        img = model_kwargs["img"]
        b, c, h, w = x_1.shape
        # Sample x_0 from a Poisson distribution (lambda=1.0 as default)
        x_0 = torch.poisson(x_0)
        t_expand = self.get_time_embedding(t, x_0.shape)

        # 2. Generate x_t and calculate target vector
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        target = x_1 - x_0

        # 3. Model prediction and loss calculation
        # Try to call with prior if model requires it
        pred = model(x_t, t, img=img)
        loss = F.mse_loss(pred, target, reduction='none')

        samples = {}
        samples["f(x)"] = model_kwargs["prior"][0]
        samples["y_t"] = x_0[0]
        samples["y_start"] = x_1[0]
        samples["model_output"] = pred[0]
        samples["image"] = model_kwargs["img"][0]
        return {"loss": loss.flatten(1).mean(1)}, samples

    @torch.no_grad()
    def sample(self, model, shape, model_kwargs, *args, **kwargs):
        """
        Euler Method ODE Solver
        dX_t = v(X_t, t) dt
        """
        x_0 = model_kwargs["prior"]

        # 1. Set number of steps and dt, initialize x
        steps = self.num_timesteps
        dt = (self.t_max - self.t_min) / steps
        x = x_0

        intermediates = [x.clone()]
        # 2. For each step, calculate time t and v
        for i in range(steps):
            t_value = self.t_min + i * dt
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
            # Try to call with prior if model requires it
            try:
                v = model(x, t, x_0)
            except TypeError:
                v = model(x, t)

            # 3. Update x using Euler's method
            x = x + v * dt
            intermediates.append(x.clone())
        return x, intermediates
