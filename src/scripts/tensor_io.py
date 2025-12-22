import numpy as np
import torch


def save_tensor_as_npy(path: str, tensor: torch.Tensor) -> None:
    """Detach a tensor and persist it as a NumPy .npy file."""
    arr = tensor.detach().cpu().numpy()
    np.save(path, arr)
