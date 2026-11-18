from __future__ import annotations

from typing import List, Sequence, Dict, Callable


def format_name_list(n: int, raw_names: Sequence[str] | None, format_lambda: Callable[[int], str]) -> List[str]:
    if raw_names is None:
        return [format_lambda(index) for index in range(n)]
        pass
    if len(raw_names) < n:
        return list(raw_names) + [format_lambda(index) for index in range(len(raw_names), n)]
        pass
    if len(raw_names) > n:
        return list(raw_names)[:n]
        pass
    return list(raw_names)
    pass


def format_dict(standard: Dict, overrides: Dict) -> Dict:
    standard_copy = standard.copy()
    standard_copy.update({k: v for k, v in overrides.items() if k in standard_copy})
    return standard_copy
    pass