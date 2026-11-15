from __future__ import annotations

from typing import Callable

import numpy as np

from ..contracts import DataBlockLike
from ..arrays import Array


class DataBlock(DataBlockLike):

    def __init__(self, x: Array, y: Array | None = None, t: Array | None = None) -> None:
        self.x_: Array = x
        self.y_: Array | None = y
        self.t_: Array | None = t
        self.__post_init__()
        pass

    def __post_init__(self) -> None:
        assert isinstance(self.x_, np.ndarray)
        assert self.x_.ndim == 2
        if self.y_ is not None:
            assert isinstance(self.y_, np.ndarray)
            assert self.y_.ndim == 2
            assert self.x_.shape[0] == self.y_.shape[0]
            pass
        if self.t_ is not None:
            assert isinstance(self.t_, np.ndarray)
            assert self.t_.ndim == 1
            assert self.x_.shape[0] == self.t_.shape[0]
            pass
        pass

    def __len__(self) -> int:
        return self.x_.shape[0]
        pass

    def x(self) -> Array:
        return self.x_
        pass

    def y(self) -> Array:
        if self.y_ is None:
            raise ValueError(f"{self.__class__.__name__}: y is None")
            pass
        return self.y_
        pass

    def t(self) -> Array:
        if self.t_ is None:
            raise ValueError(f"{self.__class__.__name__}: t is None")
            pass
        return self.t_
        pass

    pass