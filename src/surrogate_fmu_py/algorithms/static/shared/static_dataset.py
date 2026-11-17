from __future__ import annotations

from typing import List, Sequence, Self, Dict, Any

import numpy as np
import torch

from ....contracts import DataBlockLike, DatasetLike
from ....arrays import Array, Tensor, to_numpy, to_torch
from ....implements import DataBlock


class StaticDatasetTmpl(DatasetLike):
    def x(self) -> Array: raise NotImplementedError(f"x() is not implemented in {type(self)}")
    def y(self) -> Array: raise NotImplementedError(f"y() is not implemented in {type(self)}")
    pass


class StaticFullDatasetImpl(StaticDatasetTmpl):

    def __init__(self, params: Dict[str, Any]) -> None:
        self.x_list_: List[Array] = []
        self.y_list_: List[Array] = []
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        return self
        pass

    def feed(self, datablock: DataBlockLike) -> None:
        self.x_list_.append(datablock.x())
        self.y_list_.append(datablock.y())
        pass

    def finalize(self) -> None:
        pass

    def x(self) -> Array:
        return np.concatenate(self.x_list_, axis=0)
        pass

    def y(self) -> Array:
        return np.concatenate(self.y_list_, axis=0)
        pass

    def block_len(self) -> int:
        return len(self.x_list_)
        pass

    def block(self, index: int) -> DataBlockLike:
        return DataBlock(x=self.x_list_[index], y=self.y_list_[index])
        pass

    @classmethod
    def from_datablocks(cls, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> StaticFullDatasetImpl:
        dataset: StaticFullDatasetImpl = cls(params=params)
        for datablock in datablocks:
            dataset.feed(datablock=datablock)
            pass
        dataset.finalize()
        return dataset
        pass

    pass


class StaticTorchDatasetImpl(torch.utils.data.Dataset, StaticDatasetTmpl):

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.x_list_: List[Tensor] = []
        self.y_list_: List[Tensor] = []
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        return self
        pass

    def feed(self, datablock: DataBlockLike) -> None:
        self.x_list_.append(to_torch(x=datablock.x()))
        self.y_list_.append(to_torch(x=datablock.y()))
        pass

    def finalize(self) -> None:
        self.x_: Tensor = torch.cat(tensors=self.x_list_, dim=0)
        self.y_: Tensor = torch.cat(tensors=self.y_list_, dim=0)
        pass

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x_[index], self.y_[index]
        pass

    def __len__(self) -> int:
        return self.x_.shape[0]
        pass

    def x(self) -> Array:
        return to_numpy(x=self.x_)
        pass

    def y(self) -> Array:
        return to_numpy(x=self.y_)
        pass

    def block_len(self) -> int:
        return len(self.x_list_)
        pass

    def block(self, index: int) -> DataBlockLike:
        return DataBlock(x=to_numpy(x=self.x_list_[index]), y=to_numpy(x=self.y_list_[index]))
        pass

    @classmethod
    def from_datablocks(cls, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> StaticTorchDatasetImpl:
        dataset: StaticTorchDatasetImpl = cls(params=params)
        for datablock in datablocks:
            dataset.feed(datablock=datablock)
            pass
        dataset.finalize()
        return dataset
        pass

    pass