from __future__ import annotations

from typing import Self
from pathlib import Path
import struct

import numpy as np

from ..contracts import CoderLike, ScalerLike
from ..arrays import ArrayOrTensor, to_numpy
from .scaler import Scaler


def byte_array(x: ArrayOrTensor) -> bytes:
    return np.ascontiguousarray(to_numpy(x), dtype=np.float64).tobytes()
    pass


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

    def input_scaler(self) -> ScalerLike:
        return self.input_scaler_
        pass

    def output_scaler(self) -> ScalerLike:
        return self.output_scaler_
        pass

    def save(self, path: str | Path) -> Self:
        binary_path: Path = Path(path)
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        nx: int = self.input_dim()
        ny: int = self.output_dim()
        with open(binary_path, "wb") as file:
            file.write(struct.pack("i", nx))
            file.write(struct.pack("i", ny))
            file.write(byte_array(self.input_scaler_.shift()))
            file.write(byte_array(self.input_scaler_.scale()))
            file.write(byte_array(self.output_scaler_.shift()))
            file.write(byte_array(self.output_scaler_.scale()))
            pass
        return self
        pass

    def load(self, path: str | Path) -> Self:
        binary_path: Path = Path(path)
        with open(binary_path, "rb") as file:
            nx: int = struct.unpack("i", file.read(4))[0]
            ny: int = struct.unpack("i", file.read(4))[0]
            input_shift: ArrayOrTensor = np.frombuffer(file.read(nx * 8), dtype=np.float64)
            input_scale: ArrayOrTensor = np.frombuffer(file.read(nx * 8), dtype=np.float64)
            output_shift: ArrayOrTensor = np.frombuffer(file.read(ny * 8), dtype=np.float64)
            output_scale: ArrayOrTensor = np.frombuffer(file.read(ny * 8), dtype=np.float64)
            input_scaler: ScalerLike = Scaler().init(shift=input_shift, scale=input_scale)
            output_scaler: ScalerLike = Scaler().init(shift=output_shift, scale=output_scale)
            pass
        return self.init(input_scaler=input_scaler, output_scaler=output_scaler)
        pass

    pass