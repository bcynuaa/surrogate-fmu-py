from __future__ import annotations

from typing import Dict, Any, Self
from dataclasses import dataclass, asdict

import torch

from ....arrays import Tensor
from ..shared import SequenceCoreTmpl


@dataclass(frozen=True)
class RNNCoreConfig:
    window: int = 16
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 8
    num_layers: int = 1
    activation: str = "tanh"
    dropout: float = 0.0
    delta_limit: float = 0.3
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> RNNCoreConfig:
        return cls(
            window=config.get("window", 16),
            input_dim=config.get("input_dim", 1),
            output_dim=config.get("output_dim", 1),
            hidden_dim=config.get("hidden_dim", 8),
            num_layers=config.get("num_layers", 1),
            activation=config.get("activation", "tanh"),
            dropout=config.get("dropout", 0.0),
            delta_limit=config.get("delta_limit", 0.3),
            seed=config.get("seed", 42)
        )
        pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        pass

    pass


class RNNCore(SequenceCoreTmpl):

    def alg_name(self) -> str:
        return "rnn"
        pass

    def __init__(self) -> None:
        super().__init__()
        self.config_: RNNCoreConfig = RNNCoreConfig()
        pass

    def _set_config(self, config: RNNCoreConfig) -> None:
        self.config_ = config
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        self._set_config(config=RNNCoreConfig.from_dict(config=params))
        # * here, x: input_dim, y: output_dim
        # * for rnn, the model input should be [x, y], thus the input size is input_dim + output_dim
        torch.manual_seed(self.config_.seed)
        self.rnn_: torch.nn.RNN = torch.nn.RNN(
            input_size=self.config_.input_dim + self.config_.output_dim,
            hidden_size=self.config_.hidden_dim,
            num_layers=self.config_.num_layers,
            nonlinearity=self.config_.activation,
            dropout=self.config_.dropout if self.config_.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_: torch.nn.Linear = torch.nn.Linear(
            in_features=self.config_.hidden_dim,
            out_features=self.config_.output_dim
        )
        return self
        pass

    def forward(self, u: Tensor) -> Tensor:
        rnn_out, _ = self.rnn_(u)
        last_output: Tensor = rnn_out[:, -1, :]
        dy_pred: Tensor = self.fc_(last_output)
        dy_pred = torch.tanh(dy_pred) * self.config_.delta_limit
        return dy_pred + u[:, -1, self.config_.input_dim:]
        pass

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config_.to_dict()
        pass

    pass