from __future__ import annotations

import math

def format_number(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"number must be finite, got {value!r}")
        pass
    return repr(value)
    pass


def format_start(value: float | int | bool | str, variable_type: str) -> str:
    if variable_type in ("Float64", "Float32"):
        if isinstance(value, (bool, str)):
            raise ValueError(f"{variable_type} start must be a finite number")
            pass
        return format_number(value)
        pass
    if variable_type in ("Int32", "Int64"):
        bits = 32 if variable_type == "Int32" else 64
        if type(value) is not int or not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
            raise ValueError(f"{variable_type} start must be an in-range integer")
            pass
        return str(value)
        pass
    if variable_type == "Boolean":
        if type(value) is not bool:
            raise ValueError("Boolean start must be a bool")
            pass
        return "true" if value else "false"
        pass
    if variable_type == "String":
        if not isinstance(value, str):
            raise ValueError("String start must be a string")
            pass
        return value
        pass
    raise ValueError(f"unsupported variable type {variable_type!r}")
    pass
