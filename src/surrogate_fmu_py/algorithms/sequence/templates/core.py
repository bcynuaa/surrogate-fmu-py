from __future__ import annotations

from typing import Dict, Any, Self
from pathlib import Path

import torch

from ....contracts import CoreLike
from ....arrays import Array, to_numpy, to_torch


class ConfigLike:
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> ConfigLike: ...
    def to_dict(self) -> Dict[str, Any]: ...
    pass


class SequenceCoreTemplate(torch.nn.Module, CoreLike):

    def alg_family(self) -> str:
        return "sequence"
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        self.config_: ConfigLike = ConfigLike.from_dict(config=params)
        return self
        pass

    def mode_train(self) -> Self:
        self.train()
        return self
        pass

    def mode_eval(self) -> Self:
        self.eval()
        return self
        pass

    def predict(self, x: Array) -> Array:
        x_tensor: torch.Tensor = to_torch(x=x)
        with torch.no_grad():
            y_tensor: torch.Tensor = self.forward(x_tensor)
            pass
        y: Array = to_numpy(x=y_tensor)
        return y
        pass
    
    def save(self, path: str | Path) -> Self:
        model_path: Path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict: Dict[str, Any] = {
            "config": self.config_.to_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(state_dict, model_path)
        return self
        pass

    def load(self, path: str | Path) -> Self:
        model_path: Path = Path(path)
        state_dict: Dict[str, Any] = torch.load(model_path, map_location="cpu")
        config_dict: Dict[str, Any] = state_dict.get("config", {})
        self.init(params=config_dict)
        self.load_state_dict(state_dict.get("state_dict", {}))
        return self
        pass

    pass