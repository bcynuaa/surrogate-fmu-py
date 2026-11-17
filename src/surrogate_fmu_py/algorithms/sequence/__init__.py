from __future__ import annotations

from typing import Dict, Type

from ...contracts import AlgorithmLike

from .rnn import RNNAlgorithm
from .lstm import LSTMAlgorithm
from .narx import NARXAlgorithm


SEQUENCE_ALGORITHM_DICT: Dict[str, Type[AlgorithmLike]] = {}


def register_algorithm(name: str, algorithm_type: Type[AlgorithmLike]) -> None:
    if name in SEQUENCE_ALGORITHM_DICT:
        raise ValueError(f"Sequence algorithm '{name}' is already registered")
        pass
    SEQUENCE_ALGORITHM_DICT[name] = algorithm_type
    pass


def get_algorithms() -> Dict[str, Type[AlgorithmLike]]:
    return SEQUENCE_ALGORITHM_DICT.copy()
    pass


register_algorithm("rnn", RNNAlgorithm)
register_algorithm("lstm", LSTMAlgorithm)
register_algorithm("narx", NARXAlgorithm)