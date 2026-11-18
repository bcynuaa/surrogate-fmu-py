from __future__ import annotations

import warnings
from typing import Dict, Any, Sequence
from pathlib import Path

import numpy as np
import torch

from ....contracts import DataBlockLike
from ....arrays import Tensor
from ....templates import PipelineTmpl
from ....implements import DataBlock
from ....arrays import Array, concat_array, to_numpy
from ....utils import AuxiliaryIntegratedCoderTorchModule, format_name_list
from .sequence_core import SequenceCoreTmpl


class SequenceWindowPipelineImpl(PipelineTmpl):

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        super().fmu_export_artifacts(fmu_path=fmu_path)
        if not isinstance(self.core_, SequenceCoreTmpl):
            raise ValueError(f"core_ must be an instance of SequenceCoreTmpl, but got {type(self.core_)}")
            pass
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        torch_model: AuxiliaryIntegratedCoderTorchModule = AuxiliaryIntegratedCoderTorchModule(coder=self.coder_, core=self.core_)
        torch_model.eval()
        dummy_input: Tensor = torch.zeros((1, self.window_dim(), self.coder_.input_dim()), dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            torch.onnx.export(
                torch_model,
                (dummy_input,),
                fmu_resources_dir / "model.onnx",
                export_params=True,
                external_data=False,
                input_names=["u"],
                output_names=["y"],
                dynamic_axes={
                    "u": {0: "batch_size"},
                },
                verbose=False,
                dynamo=False,
            )
            pass
        pass

    def rollout(self, datablock: DataBlockLike, y0: Array, *args, **kwargs) -> DataBlock:
        x: Array = datablock.x()
        window_dim: int = self.window_dim() # W
        x_dim: int = self.coder_.input_dim() - self.coder_.output_dim() # X
        y_dim: int = self.coder_.output_dim() # Y
        if x.shape[0] < window_dim:
            raise ValueError(f"Input sequence length {x.shape[0]} is shorter than the required window dimension {window_dim}.")
            pass
        if x.shape[1] != x_dim:
            raise ValueError(f"Input sequence dimension {x.shape[1]} does not match the expected input dimension {x_dim}.")
            pass
        # y0 dim can be (K, output_dim)
        if y0.ndim == 1:
            y0 = y0.reshape(1, -1)
            pass
        if y0.shape[1] != y_dim:
            raise ValueError(f"Initial output sequence dimension {y0.shape[1]} does not match the expected output dimension {y_dim}.")
            pass
        if y0.shape[0] < window_dim:
            # K < window_dim, complete the initial sequence with the last K values of x
            y_initial: Array = np.zeros((window_dim, y_dim), dtype=x.dtype)
            y_initial[:y0.shape[0], :] = y0
            for i in range(y0.shape[0], window_dim):
                y_initial[i, :] = y0[-1, :] # take the last y_dim values from y0
                pass
            pass
        else:
            # K >= window_dim, take the last window_dim values of y0
            y_initial = y0[-window_dim:, :]
            pass
        y_traj: Array = y_initial.copy()
        for t in range(window_dim, x.shape[0]):
            x_buffer: Array = x[t-window_dim+1:t+1, :]
            y_buffer: Array = y_traj[t-window_dim:t, :]
            input_buffer: Array = to_numpy(concat_array(to_numpy(x_buffer), to_numpy(y_buffer)))
            input_buffer = input_buffer.reshape(1, window_dim, x_dim + y_dim)
            y_next: Array = self.predict(x=input_buffer)
            y_traj = np.concatenate([y_traj, y_next], axis=0)
            pass
        return DataBlock(x, y_traj)
        pass

    def window_dim(self) -> int:
        if not isinstance(self.core_, SequenceCoreTmpl):
            raise ValueError(f"core_ must be an instance of SequenceCoreTmpl, but got {type(self.core_)}")
            pass
        return self.core_.window_dim()
        pass

    def fmu_export_spec(self, input_names: Sequence[str] | None=None, output_names: Sequence[str] | None=None) -> Dict[str, Any]:
        parent_spec: Dict[str, Any] = super().fmu_export_spec(input_names=input_names, output_names=output_names)
        input_names = format_name_list(self.input_dim() - self.output_dim(), input_names, lambda index: f"x_{index+1}")
        output_names = format_name_list(self.output_dim(), output_names, lambda index: f"y_{index+1}")
        updated_spec: Dict[str, Any] = {
            "input_names": input_names,
            "output_names": output_names,
            "require_initial_values": True,
            "initial_names": format_name_list(self.output_dim(), None, lambda index: f"{output_names[index]}_initial"),
            "initial_values": [0.0] * self.output_dim(),
        }
        parent_spec.update(updated_spec)
        return parent_spec
        pass

    pass