from __future__ import annotations

from typing import Self

from ..contracts import CoderLike, ScalerLike
from ..arrays import ArrayOrTensor
from .scaler import Scaler


class Coder(CoderLike):

    def __init__(self) -> None:
        pass

    def init(self, input_scaler: ScalerLike, output_scaler: ScalerLike) -> Self:
        self.input_scaler_: ScalerLike = input_scaler.to_numpy()
        self.output_scaler_: ScalerLike = output_scaler.to_numpy()
        return self
        pass

    def encode(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return self.input_scaler_.encode(x=x)
        pass

    def decode(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return self.output_scaler_.decode(x=x)
        pass

    def encode_input(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return self.encode(x=x)
        pass

    def encode_output(self, x: ArrayOrTensor) -> ArrayOrTensor:
        return self.output_scaler_.encode(x=x)
        pass

    def to_numpy(self) -> Self:
        self.input_scaler_ = self.input_scaler_.to_numpy()
        self.output_scaler_ = self.output_scaler_.to_numpy()
        return self
        pass

    def to_torch(self) -> Self:
        self.input_scaler_ = self.input_scaler_.to_torch()
        self.output_scaler_ = self.output_scaler_.to_torch()
        return self
        pass

    def input_dim(self) -> int:
        return self.input_scaler_.dim()
        pass

    def output_dim(self) -> int:
        return self.output_scaler_.dim()
        pass

    def save(self, path: str) -> Self:
        # TODO
        return self
        pass

    def load(self, path: str) -> Self:
        # TODO
        input_scaler: ScalerLike = Scaler()
        output_scaler: ScalerLike = Scaler()
        return self.init(input_scaler=input_scaler, output_scaler=output_scaler)
        pass

    pass