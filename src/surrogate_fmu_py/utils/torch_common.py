from __future__ import annotations

from collections.abc import Iterable
from typing import List, Dict, Callable, Type

import torch


class ResLinear(torch.nn.Module):

    def __init__(self, dim: int) ->None:
        super().__init__()
        self.linear_: torch.nn.Linear = torch.nn.Linear(dim, dim)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear_(x)
        pass

    pass


def linear_layer(input_dim: int, output_dim: int) -> torch.nn.Module:
    if input_dim != output_dim:
        return torch.nn.Linear(input_dim, output_dim)
        pass
    else:
        return ResLinear(input_dim)
        pass
    pass


def get_activation(activation_str: str) -> torch.nn.Module:
    ACTIVATION_MAP: Dict[str, type[torch.nn.Module]] = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        "softplus": torch.nn.Softplus,
        "softsign": torch.nn.Softsign,
        "identity": torch.nn.Identity,
    }
    activation: str = activation_str.lower()
    activation = "identity" if activation not in ACTIVATION_MAP else activation
    return ACTIVATION_MAP[activation]()
    pass


def get_loss_function(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    LOSS_FN_MAP: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "mse": torch.nn.functional.mse_loss,
        "mae": torch.nn.functional.l1_loss,
        "smoothl1": torch.nn.functional.smooth_l1_loss,
        "huber": torch.nn.functional.huber_loss,
    }
    loss_type_str: str = loss_type.lower()
    loss_type_str = "mse" if loss_type_str not in LOSS_FN_MAP else loss_type_str
    return LOSS_FN_MAP[loss_type_str]
    pass


def get_optimizer_class(optimizer_name: str) -> Type[torch.optim.Optimizer]:
    OPTIMIZER_MAP: Dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
    }
    optimizer: str = optimizer_name.lower()
    optimizer = "adam" if optimizer not in OPTIMIZER_MAP else optimizer
    return OPTIMIZER_MAP[optimizer]
    pass


def create_optimizer(
    optimizer_name: str,
    params: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "rprop":
        if weight_decay != 0.0:
            raise ValueError("Rprop does not support weight_decay")
        return torch.optim.Rprop(params, lr=learning_rate)
    if name == "lbfgs":
        raise ValueError("LBFGS requires a closure and is not supported by DNNTrainer")

    factories: dict[str, Callable[[Iterable[torch.nn.Parameter], float, float], torch.optim.Optimizer]] = {
        "adam": lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
        "sgd": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd),
        "adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
        "adagrad": lambda p, lr, wd: torch.optim.Adagrad(p, lr=lr, weight_decay=wd),
        "adadelta": lambda p, lr, wd: torch.optim.Adadelta(p, lr=lr, weight_decay=wd),
        "rmsprop": lambda p, lr, wd: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
        "asgd": lambda p, lr, wd: torch.optim.ASGD(p, lr=lr, weight_decay=wd),
    }
    return factories.get(name, factories["adam"])(params, learning_rate, weight_decay)


def dense_layer(
    input_dim: int,
    output_dim: int,
    activation: str = "relu",
    use_batchnorm: bool = False,
    dropout: float = 0.0,
) -> torch.nn.Sequential:
    layer_list: List[torch.nn.Module] = []
    layer_list.append(linear_layer(input_dim=input_dim, output_dim=output_dim))
    if use_batchnorm:
        layer_list.append(torch.nn.BatchNorm1d(num_features=output_dim))
        pass
    layer_list.append(get_activation(activation_str=activation))
    if dropout > 0.0:
        layer_list.append(torch.nn.Dropout(p=dropout))
        pass
    return torch.nn.Sequential(*layer_list)
    pass
