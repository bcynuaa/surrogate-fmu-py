from __future__ import annotations

from typing import List, Dict, Type

from ..contracts import AlgorithmLike
from . import static
from . import sequence


def _all_algorithms() -> Dict[str, Type[AlgorithmLike]]:
    algorithms: Dict[str, Type[AlgorithmLike]] = {}
    owners: Dict[str, str] = {}
    for family_name, family_algorithms in (
        ("static", static.get_algorithms()),
        ("sequence", sequence.get_algorithms()),
    ):
        for name, algorithm_type in family_algorithms.items():
            if name in algorithms:
                raise ValueError(
                    f"Algorithm '{name}' is registered in both '{owners[name]}' and '{family_name}'"
                )
            algorithms[name] = algorithm_type
            owners[name] = family_name
            pass
    return algorithms
    pass


def get_algorithm(alg_name: str) -> AlgorithmLike:
    algorithms = _all_algorithms()
    if alg_name not in algorithms:
        raise ValueError(f"Algorithm '{alg_name}' is not available. Available algorithms: {list(algorithms)}")
        pass
    return algorithms[alg_name]()
    pass


def get_available_algorithms() -> List[str]:
    return list(_all_algorithms())
    pass
