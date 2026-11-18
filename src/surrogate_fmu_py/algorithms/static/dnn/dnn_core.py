from __future__ import annotations

from typing import List, Dict, Any, Self
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from ....arrays import Array, Tensor, to_numpy, to_torch
from ....utils import dense_layer
from ..shared import StaticCoreTmpl


@dataclass(frozen=True)
class DNNCoreConfig:
    input_dim: int = 1
    output_dim: int = 1
    hidden_layers: List[int] = field(default_factory=lambda: [8, 16, 8])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu", "relu"])
    use_batchnorm: bool = False
    dropout: float = 0.0
    seed: int = 42

    @classmethod
    def from_dict(cls, config: Dict[str, Any]={}) -> DNNCoreConfig:
        return cls(
            input_dim=config.get("input_dim", 1),
            output_dim=config.get("output_dim", 1),
            hidden_layers=config.get("hidden_layers", [8, 16, 8]),
            activations=config.get("activations", ["relu", "relu", "relu"]),
            use_batchnorm=config.get("use_batchnorm", False),
            dropout=config.get("dropout", 0.0),
            seed=config.get("seed", 42),
        )
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        pass

    pass


class DNNCore(torch.nn.Module, StaticCoreTmpl):

    def alg_name(self) -> str:
        return "dnn"
        pass

    def __init__(self) -> None:
        super().__init__()
        self.config_: DNNCoreConfig = DNNCoreConfig()
        pass

    def _set_config(self, config: DNNCoreConfig) -> None:
        self.config_ = config
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        self._set_config(config=DNNCoreConfig.from_dict(config=params))
        torch.manual_seed(self.config_.seed)
        layer_list: List[torch.nn.Module] = []
        input_dim: int = self.config_.input_dim
        for i, hidden_dim in enumerate(self.config_.hidden_layers):
            activation: str = self.config_.activations[i]
            layer_list.append(
                dense_layer(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    use_batchnorm=self.config_.use_batchnorm,
                    dropout=self.config_.dropout,
                )
            )
            input_dim = hidden_dim
            pass
        layer_list.append(
            dense_layer(
                input_dim=input_dim,
                output_dim=self.config_.output_dim,
                activation="linear",
                use_batchnorm=False,
                dropout=0.0,
            )
        )
        self.layers_ = torch.nn.Sequential(*layer_list)
        return self
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.layers_(x)
        pass

    def mode_train(self) -> Self:
        self.train()
        return self
        pass

    def mode_eval(self) -> Self:
        self.to(device="cpu")
        self.eval()
        return self
        pass

    def fmu_export_core_artifacts(self, fmu_path: str | Path) -> None:
        self.mode_eval()
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        input_dim: int = self.config_.input_dim
        dummy_input: Tensor = torch.zeros((1, input_dim), dtype=torch.float32)
        torch.onnx.export(
            model=self,
            args=(dummy_input,),
            f=fmu_resources_dir / "model.onnx",
            external_data=False,
            input_names=["x"],
            output_names=["y"],
            dynamic_shapes={
                "x": {0: "batch_size"},
            },
            verbose=False,
        )
        pass

    def predict(self, x: Array) -> Array:
        # default assume self.mode_eval() has been called
        x_tensor: Tensor = to_torch(x)
        with torch.no_grad():
            y_tensor: Tensor = self.forward(x_tensor)
            pass
        y: Array = to_numpy(y_tensor)
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
        # load to cpu
        model_path: Path = Path(path)
        state_dict: Dict[str, Any] = torch.load(model_path, map_location="cpu")
        config_dict: Dict[str, Any] = state_dict.get("config", {})
        self.init(params=config_dict)
        self.load_state_dict(state_dict.get("state_dict", {}))
        return self
        pass

    pass