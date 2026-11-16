from __future__ import annotations

import os
from pathlib import Path

CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# x0, x1, x2, x3
# x0*x1, x0*x2, x3+x0/2*(x1**2 + x2**2)

def func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    y: NDArray[np.float64] = np.zeros((x.shape[0], 3), dtype=np.float64)
    y[:, 0] = x[:, 0] * x[:, 1]
    y[:, 1] = x[:, 0] * x[:, 2]
    y[:, 2] = x[:, 3] + x[:, 0] / 2 * (x[:, 1] ** 2 + x[:, 2] ** 2)
    return y
    pass

n_samples_per_df: int = 100
n_dfs: int = 10
x0: NDArray[np.float64] = np.random.uniform(800, 1200, size=(n_samples_per_df * n_dfs, 1))
x1: NDArray[np.float64] = np.random.uniform(0, 10, size=(n_samples_per_df * n_dfs, 1))
x2: NDArray[np.float64] = np.random.uniform(-3, 3, size=(n_samples_per_df * n_dfs, 1))
x3: NDArray[np.float64] = np.random.uniform(0.8e5, 1.2e5, size=(n_samples_per_df * n_dfs, 1))
x: NDArray[np.float64] = np.hstack((x0, x1, x2, x3))
y: NDArray[np.float64] = func(x)

for i in range(n_dfs):
    df: pd.DataFrame = pd.DataFrame(
        data=np.hstack((x[i * n_samples_per_df:(i + 1) * n_samples_per_df, :],
                        y[i * n_samples_per_df:(i + 1) * n_samples_per_df, :])),
        columns=['rho', 'u', 'v', 'p', 'px', 'py', 'e']
    )
    df.to_csv(Path(CURRENT_DIR) / f"static_{i:03d}.csv", index=False)
    pass