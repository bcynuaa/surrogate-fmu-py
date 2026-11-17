from __future__ import annotations

from typing import List, Dict, Any, Self
from dataclasses import dataclass, asdict, field

import torch

from ....arrays import Tensor
from ....utils import dense_layer
from ..shared import SequenceCoreTmpl


@dataclass(frozen=True)
class NARXCoreConfig:
    window: int = 16
    input_dim: int = 1
    output_dim: int = 1
    wx_dim: int = 16
    wy_dim: int = 8
    hidden_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu", "relu"])
    use_batchnorm: bool = False
    dropout: float = 0.0
    delta_limit: float = 0.3
    seed: int = 42

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> NARXCoreConfig:
        return cls(
            window=config.get("window", 16),
            input_dim=config.get("input_dim", 1),
            output_dim=config.get("output_dim", 1),
            wx_dim=config.get("wx_dim", 16),
            wy_dim=config.get("wy_dim", 8),
            hidden_layers=config.get("hidden_layers", [32, 16, 8]),
            activations=config.get("activations", ["relu", "relu", "relu"]),
            use_batchnorm=config.get("use_batchnorm", False),
            dropout=config.get("dropout", 0.0),
            delta_limit=config.get("delta_limit", 0.3),
            seed=config.get("seed", 42)
        )
        pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        pass

    pass


class NARXCore(SequenceCoreTmpl):

    def alg_name(self) -> str:
        return "narx"
        pass

    def __init__(self) -> None:
        super().__init__()
        self.config_: NARXCoreConfig = NARXCoreConfig()
        pass

    def _set_config(self, config: NARXCoreConfig) -> None:
        self.config_ = config
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        self._set_config(config=NARXCoreConfig.from_dict(config=params))
        torch.manual_seed(self.config_.seed)
        layer_list: List[torch.nn.Module] = []
        input_dim: int = self.config_.input_dim * self.config_.wx_dim + self.config_.output_dim * self.config_.wy_dim
        for i, hidden_dim in enumerate(self.config_.hidden_layers):
            activation: str = self.config_.activations[i]
            layer_list.append(
                dense_layer(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    use_batchnorm=self.config_.use_batchnorm,
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
            )
        )
        self.model_: torch.nn.Sequential = torch.nn.Sequential(*layer_list)
        return self
        pass

    def forward(self, u: Tensor) -> Tensor:
        # u: (B, W, X+Y)
        y0: Tensor = u[:, -1, self.config_.input_dim:]  # (B, Y)
        x: Tensor = u[:, -self.config_.wx_dim:, :self.config_.input_dim]  # (B, WX, X)
        y: Tensor = u[:, -self.config_.wy_dim:, self.config_.input_dim:]  # (B, WY, Y)
        x_flat: Tensor = x.reshape(x.shape[0], -1)  # (B, WX*X)
        y_flat: Tensor = y.reshape(y.shape[0], -1)  # (B, WY*Y)
        xy_flat: Tensor = torch.cat([x_flat, y_flat], dim=1)
        dy_pred: Tensor = self.model_(xy_flat) # (B, Y)
        dy_pred = torch.tanh(dy_pred) * self.config_.delta_limit
        return dy_pred + y0
        pass

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config_.to_dict()
        pass

    pass