from __future__ import annotations

from typing import List, Dict, Type

from .contracts import AlgorithmLike


from .algorithms.static.dnn.algorithm import DNNAlgorithm
ALGORITHM_DICT: Dict[str, Type[AlgorithmLike]] = {
    "dnn": DNNAlgorithm,
}


def get_algorithm(alg_name: str) -> AlgorithmLike:
    if alg_name not in get_available_algorithms():
        raise ValueError(f"Algorithm '{alg_name}' is not available. Available algorithms: {get_available_algorithms()}")
        pass
    return ALGORITHM_DICT[alg_name]()
    pass


def get_available_algorithms() -> List[str]:
    return list[str](ALGORITHM_DICT.keys())
    pass