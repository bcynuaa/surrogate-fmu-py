from __future__ import annotations

from typing import List, Dict, Self, Type

import numpy as np

from ..contracts import ScalerLike
from ..arrays import Array, ArrayOrTensor, to_numpy, to_torch, concat_array


class Scaler(ScalerLike):
    
    def __init__(self) -> None:
        pass

    def feed(self, x: Array) -> None: ...

    def finalize(self) -> Self: ...

    def init(self, shift: ArrayOrTensor, scale: ArrayOrTensor) -> Self:
        if shift.shape[0] != scale.shape[0]:
            raise ValueError(f"{self.__class__.__name__}: shift and scale must have the same dimension, got {shift.shape[0]} and {scale.shape[0]}")
            pass
        self.shift_: ArrayOrTensor = shift
        self.scale_: ArrayOrTensor = scale
        return self
        pass

    def shift(self) -> ArrayOrTensor:
        return self.shift_
        pass

    def scale(self) -> ArrayOrTensor:
        return self.scale_
        pass

    def dim(self) -> int:
        return self.shift_.shape[0]
        pass

    def encode(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return (x - self.shift()) / self.scale()
        pass

    def decode(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return x * self.scale() + self.shift()
        pass

    def to_numpy(self) -> Self:
        self.shift_ = to_numpy(x=self.shift_)
        self.scale_ = to_numpy(x=self.scale_)
        return self
        pass

    def to_torch(self) -> Self:
        self.shift_ = to_torch(x=self.shift_)
        self.scale_ = to_torch(x=self.scale_)
        return self
        pass

    pass


class NoneScaler(Scaler):

    def __init__(self) -> None:
        super().__init__()
        self.dim_: int = 0
        pass

    def feed(self, x: Array) -> None:
        if self.dim_ == 0:
            self.dim_ = x.shape[-1]
            pass
        if self.dim_ != 0 and self.dim_ != x.shape[-1]:
            raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.dim_}, got {x.shape[-1]}")
            pass
        pass

    def finalize(self) -> Self:
        self.shift_: ArrayOrTensor = np.zeros(self.dim_, dtype=np.float64)
        self.scale_: ArrayOrTensor = np.ones(self.dim_, dtype=np.float64)
        return self
        pass

    pass


class MinMaxScaler(Scaler):

    def __init__(self) -> None:
        super().__init__()
        self.min_: Array | None = None
        self.max_: Array | None = None
        pass

    def feed(self, x: Array) -> None:
        if self.min_ is None or self.max_ is None:
            self.min_ = np.min(x, axis=0)
            self.max_ = np.max(x, axis=0)
            pass
        else:
            if x.shape[1] != self.min_.shape[0]:
                raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.min_.shape[0]}, got {x.shape[1]}")
                pass
            if x.shape[1] != self.max_.shape[0]:
                raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.max_.shape[0]}, got {x.shape[1]}")
                pass
            self.min_ = np.minimum(self.min_, np.min(x, axis=0))
            self.max_ = np.maximum(self.max_, np.max(x, axis=0))
            pass
        pass

    def finalize(self) -> Self:
        self.shift_: ArrayOrTensor = self.min_ if self.min_ is not None else np.zeros(0, dtype=np.float64)
        self.scale_: ArrayOrTensor = self.max_ - self.min_ if self.min_ is not None and self.max_ is not None else np.ones(0, dtype=np.float64)
        self.scale_[self.scale_ <= 0.0] = 1.0  # avoid division by zero
        return self
        pass

    pass


class GlobalMinMaxScaler(Scaler):

    def __init__(self) -> None:
        super().__init__()
        self.global_min_: float | None = None
        self.global_max_: float | None = None
        self.dim_: int | None = None
        pass

    def feed(self, x: Array) -> None:
        if self.dim_ is None:
            self.dim_ = x.shape[1]
            pass
        elif self.dim_ != x.shape[1]:
            raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.dim_}, got {x.shape[1]}")
            pass
        current_min: float = np.min(x)
        current_max: float = np.max(x)
        if self.global_min_ is None or self.global_max_ is None:
            self.global_min_ = current_min
            self.global_max_ = current_max
            pass
        else:
            self.global_min_ = min(self.global_min_, current_min)
            self.global_max_ = max(self.global_max_, current_max)
            pass
        pass

    def finalize(self) -> Self:
        if self.global_min_ is None or self.global_max_ is None:
            raise ValueError(f"{self.__class__.__name__}: no data fed to scaler")
            pass
        self.shift_: ArrayOrTensor = np.zeros(self.dim_, dtype=np.float64) if self.dim_ is not None else np.zeros(0, dtype=np.float64)
        self.scale_: ArrayOrTensor = np.full(self.dim_, self.global_max_ - self.global_min_, dtype=np.float64) if self.dim_ is not None else np.ones(0, dtype=np.float64)
        self.scale_[self.scale_ <= 0.0] = 1.0  # avoid division by zero
        return self
        pass

    pass


class NormalScaler(Scaler):

    def __init__(self) -> None:
        super().__init__()
        self.mean_: Array | None = None
        self.m2_: Array | None = None
        self.count_: int = 0
        pass

    def feed(self, x: Array) -> None:
        # use Welford's algorithm for numerically stable mean and variance calculation
        if self.mean_ is None or self.m2_ is None:
            self.mean_ = np.mean(x, axis=0)
            self.m2_ = np.var(x, axis=0) * x.shape[0]
            self.count_ = x.shape[0]
            pass
        else:
            if x.shape[1] != self.mean_.shape[0]:
                raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.mean_.shape[0]}, got {x.shape[1]}")
                pass
            if x.shape[1] != self.m2_.shape[0]:
                raise ValueError(f"{self.__class__.__name__}: input dimension mismatch, expected {self.m2_.shape[0]}, got {x.shape[1]}")
                pass
            new_count: int = self.count_ + x.shape[0]
            delta: Array = np.mean(x, axis=0) - self.mean_
            self.mean_ += delta * (x.shape[0] / new_count)
            self.m2_ += np.var(x, axis=0) * x.shape[0] + delta**2 * (self.count_ * x.shape[0] / new_count)
            self.count_ = new_count
            pass
        pass

    def finalize(self) -> Self:
        if self.mean_ is None or self.m2_ is None:
            raise ValueError(f"{self.__class__.__name__}: no data fed to scaler")
            pass
        self.shift_: ArrayOrTensor = self.mean_.copy() if self.mean_ is not None else np.zeros(0, dtype=np.float64)
        self.scale_: ArrayOrTensor = np.sqrt(self.m2_ / self.count_) if self.m2_ is not None and self.count_ > 0 else np.ones(0, dtype=np.float64)
        self.scale_[self.scale_ <= 0.0] = 1.0  # avoid division by zero
        return self
        pass

    pass


def concat_scalers(*scalers: Scaler) -> Scaler:
    if not scalers:
        raise ValueError("No scalers provided for concatenation")
        pass
    shift: ArrayOrTensor = concat_array(*[scaler.shift() for scaler in scalers])
    scale: ArrayOrTensor = concat_array(*[scaler.scale() for scaler in scalers])
    new_scaler: Scaler = Scaler()
    new_scaler.init(shift=shift, scale=scale)
    return new_scaler
    pass


SCALERS_DICT: Dict[str, Type[Scaler]] = {
    "none": NoneScaler,
    "minmax": MinMaxScaler,
    "global_minmax": GlobalMinMaxScaler,
    "normal": NormalScaler,
}


def get_available_scalers() -> List[str]:
    return list[str](SCALERS_DICT.keys())
    pass


def get_scaler(scaler_mode: str) -> Scaler:
    if scaler_mode not in SCALERS_DICT:
        raise ValueError(f"Scaler '{scaler_mode}' is not available. Available scalers: {get_available_scalers()}")
        pass
    return SCALERS_DICT[scaler_mode]()
    pass