from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch

Array: TypeAlias = NDArray[np.float64]  # type alias for numpy array of float64
Tensor: TypeAlias = torch.Tensor  # type alias for torch tensor
ArrayOrTensor: TypeAlias = Array | Tensor  # type alias for numpy array or torch tensor

def numpy_to_torch(x: Array) -> Tensor:
    return torch.from_numpy(x.astype(np.float32))       
    pass


def torch_to_numpy(x: Tensor) -> Array:
    return x.detach().cpu().numpy().astype(np.float64)
    pass


def to_numpy(x: ArrayOrTensor) -> Array:
    if isinstance(x, np.ndarray):
        return x
        pass
    elif isinstance(x, torch.Tensor):
        return torch_to_numpy(x=x)
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
        pass
    pass


def to_torch(x: ArrayOrTensor) -> Tensor:
    if isinstance(x, np.ndarray):
        return numpy_to_torch(x=x)
        pass
    elif isinstance(x, torch.Tensor):
        return x
        pass
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
        pass
    pass


def concat_array(*arrays: ArrayOrTensor) -> ArrayOrTensor:
    if not arrays:
        raise ValueError("No arrays provided for concatenation")
        pass
    if all(isinstance(arr, np.ndarray) for arr in arrays):
        return np.concatenate(arrays, axis=-1)
        pass
    elif all(isinstance(arr, torch.Tensor) for arr in arrays):
        return torch.cat(tensors=[to_torch(x=arr) for arr in arrays], dim=-1)
        pass
    else:
        raise TypeError("All inputs must be of the same type (either all numpy arrays or all torch tensors)")
        pass
    pass