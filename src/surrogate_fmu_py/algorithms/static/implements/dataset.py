from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch

from ....contracts import DatasetLike, DataBlockLike
from ....arrays import Array, Tensor, to_numpy, to_torch


class StaticFullDataset(DatasetLike):

    def __init__(self, params: Dict[str, Any]) -> None:
        self.x_list_: List[Array] = []
        self.y_list_: List[Array] = []
        pass

    def feed(self, datablock: DataBlockLike) -> None:
        self.x_list_.append(datablock.x())
        self.y_list_.append(datablock.y())
        pass

    def finalize(self) -> None:
        self.x_: Array = np.concatenate(self.x_list_, axis=0)
        self.y_: Array = np.concatenate(self.y_list_, axis=0)
        del self.x_list_
        del self.y_list_
        pass

    def x(self) -> Array:
        return self.x_
        pass

    def y(self) -> Array:
        return self.y_
        pass

    pass


class StaticTorchDataset(torch.utils.data.Dataset, DatasetLike):

    def __init__(self, params: Dict[str, Any]) -> None:
        self.x_list_: List[Tensor] = []
        self.y_list_: List[Tensor] = []
        pass

    def feed(self, datablock: DataBlockLike) -> None:
        self.x_list_.append(to_torch(x=datablock.x()))
        self.y_list_.append(to_torch(x=datablock.y()))
        pass

    def finalize(self) -> None:
        self.x_: Tensor = torch.cat(tensors=self.x_list_, dim=0)
        self.y_: Tensor = torch.cat(tensors=self.y_list_, dim=0)
        del self.x_list_
        del self.y_list_
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

    pass

    