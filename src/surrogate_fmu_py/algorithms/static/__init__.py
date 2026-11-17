from __future__ import annotations

from typing import Dict, Type

from ...contracts import AlgorithmLike
from .dnn import DNNAlgorithm
from .gp import GPAlgorithm


STATIC_ALGORITHM_DICT: Dict[str, Type[AlgorithmLike]] = {}


def register_algorithm(name: str, algorithm_type: Type[AlgorithmLike]) -> None:
    if name in STATIC_ALGORITHM_DICT:
        raise ValueError(f"Static algorithm '{name}' is already registered")
        pass
    STATIC_ALGORITHM_DICT[name] = algorithm_type
    pass


def get_algorithms() -> Dict[str, Type[AlgorithmLike]]:
    return STATIC_ALGORITHM_DICT.copy()
    pass


register_algorithm("dnn", DNNAlgorithm)
register_algorithm("gp", GPAlgorithm)
