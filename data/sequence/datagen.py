from __future__ import annotations

from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as scp

CURRENT_DIR: Path = Path(__file__).resolve().parent

def build_ode_func(param_func: Callable, ode_func_with_params: Callable) -> Callable:
    def ode_func(y: np.ndarray, t: float) -> np.ndarray:
        params: np.ndarray = param_func(t)
        dydt: np.ndarray = ode_func_with_params(y, params, t)
        return dydt
        pass
    return ode_func
    pass

def ode_func_template(y: np.ndarray, params: np.ndarray, t: float) -> np.ndarray:
    p1, p2, p3, p4, p5, p6 = params
    return np.array([
        -(0 * y[0] + p2 * y[1] + p3 * y[2] + p4) * 1,
        (p1 * y[0] + 0 * y[1] + p3 * y[2] + p5) * 2,
        -(p1 * y[0] + p2 * y[1] + 0 * y[2] + p6) * 3,
    ])
    pass

def build_param_func(i: int=0) -> Callable:
    def param_func(t: float) -> np.ndarray:
        t = t - i * 0.1
        p1: float = np.exp(-t / 1) * np.sin(np.pi * t / 1)
        p2: float = np.exp(-t / 2) * np.cos(np.pi * t / 2)
        p3: float = np.exp(-t / 3) * np.sin(np.pi * t / 3)
        p4: float = np.exp(-t / 1) * np.cos(np.pi * t / 3)
        p5: float = np.exp(-t / 2) * np.sin(np.pi * t / 1)
        p6: float = np.exp(-t / 3) * np.cos(np.pi * t / 2)
        return np.array([p1, p2, p3, p4, p5, p6])
        pass
    return param_func
    pass

def build_init_state(i: int=0) -> np.ndarray:
    return np.array([1.0, 0.0, 0.0]) + i * 0.1
    pass

def build_time_points() -> np.ndarray:
    return np.linspace(0, 10, 501)
    pass

def build_trajectory(i: int=0) -> pd.DataFrame:
    param_func: Callable = build_param_func(i)
    init_state: np.ndarray = build_init_state(i)
    ts: np.ndarray = build_time_points()
    ode_func: Callable = build_ode_func(param_func, ode_func_template)
    ys: np.ndarray = scp.integrate.odeint(ode_func, init_state, ts)
    params: np.ndarray = np.array([param_func(t) for t in ts])
    return pd.DataFrame({
        'time': ts,
        'y1': ys[:, 0],
        'y2': ys[:, 1],
        'y3': ys[:, 2],
        'p1': params[:, 0],
        'p2': params[:, 1],
        'p3': params[:, 2],
        'p4': params[:, 3],
        'p5': params[:, 4],
        'p6': params[:, 5]
    })
    pass

def main() -> None:
    for i in range(10):
        df: pd.DataFrame = build_trajectory(i)
        df.to_csv(CURRENT_DIR / f"sequence_{i:03d}.csv", index=False)
        pass
    pass

if __name__ == "__main__":
    main()
    pass