from __future__ import annotations

from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict

import torch

from ....contracts import DatasetLike, DataBlockLike
from ....arrays import Array, Tensor, to_torch, concat_array
from ....implements import DataBlock


@dataclass(frozen=True)
class SequenceWindowDatasetImplConfig:
    window: int = 16
    stride: int = 8
    @classmethod
    def from_dict(cls, config: Dict[str, int]) -> SequenceWindowDatasetImplConfig:
        window: int = config.get("window", 16)
        stride: int = config.get("stride", 8)
        if window <= 1 or not isinstance(window, int):
            raise ValueError(f"Window must be an integer greater than 1, got {window}")
            pass
        if stride <= 0 or not isinstance(stride, int):
            raise ValueError(f"Stride must be a positive integer, got {stride}")
            pass
        return SequenceWindowDatasetImplConfig(
            window=window,
            stride=stride,
        )
        pass

    def to_dict(self) -> Dict:
        return asdict(self)
        pass
    pass


class SequenceWindowDatasetImpl(torch.utils.data.Dataset, DatasetLike):
    """
    * input: [
    *     x[t-w+1, X], y[t-w, Y],
    *     x[t-w+2, X], y[t-w+1, Y],
    *     x[t-w+3, X], y[t-w+2, Y],
    *     ...
    *     x[t-2, X], y[t-3, Y],
    *     x[t-1, X], y[t-2, Y],
    *     x[t, X], y[t-1, Y],
    * ] 
    * output: y[t, Y]

    ! thus, a sample will contain **w+1** rows
    """
    def __init__(self) -> None:
        self.init(params=SequenceWindowDatasetImplConfig().to_dict())
        pass

    def init(self, params: Dict[str, int]) -> SequenceWindowDatasetImpl:
        self.config_ = SequenceWindowDatasetImplConfig.from_dict(params)
        self.input_list_: List[Tensor] = []
        self.output_list_: List[Tensor] = []
        self.raw_xs_: List[Array] = []
        self.raw_ys_: List[Array] = []
        return self
        pass

    def feed(self, datablock: DataBlockLike) -> None:
        xs: Array = datablock.x()
        ys: Array = datablock.y()
        self.raw_xs_.append(xs)
        self.raw_ys_.append(ys)
        assert xs.shape[0] >= self.config_.window, f"Datablock x has {xs.shape[0]} rows, but window size is {self.config_.window}"
        current_start_index: int = 0
        while current_start_index + self.config_.window + 1 <= xs.shape[0]:
            x_input: Array = xs[current_start_index+1:current_start_index+self.config_.window+1, :]
            y_input: Array = ys[current_start_index:current_start_index+self.config_.window, :]
            y_output: Array = ys[current_start_index+self.config_.window, :]
            input_tensor: Tensor = to_torch(x=concat_array(x_input, y_input))
            output_tensor: Tensor = to_torch(x=y_output)
            self.input_list_.append(input_tensor)
            self.output_list_.append(output_tensor)
            current_start_index += self.config_.stride
            pass
        pass

    def finalize(self) -> None:
        pass

    def block_len(self) -> int:
        return len(self.raw_xs_)
        pass

    def block(self, index: int) -> DataBlock:
        return DataBlock(x=self.raw_xs_[index], y=self.raw_ys_[index])
        pass

    def __len__(self) -> int:
        return len(self.input_list_)
        pass

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.input_list_[index], self.output_list_[index]
        pass

    pass