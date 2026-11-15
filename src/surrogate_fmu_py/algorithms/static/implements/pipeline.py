from __future__ import annotations

from typing import Dict, Any, Self

from ....contracts import CoderLike, CoreLike, PipelineLike
from ....arrays import Array, to_numpy


class StaticPipeline(PipelineLike):

    def __init__(self) -> None:
        pass

    def init(self, coder: CoderLike, core: CoreLike, params: Dict[str, Any]) -> Self:
        self.coder_: CoderLike = coder.to_numpy()
        self.core_: CoreLike = core.eval_mode()
        self.params_: Dict[str, Any] = params.copy()
        return self
        pass

    def predict(self, x: Array) -> Array:
        xb: Array = to_numpy(x=self.coder_.encode(x=x))
        yb: Array = self.core_.predict(x=xb)
        y: Array = to_numpy(x=self.coder_.decode(x=yb))
        return y

    pass