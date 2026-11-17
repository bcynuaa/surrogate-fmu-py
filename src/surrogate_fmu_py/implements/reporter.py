from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from ..contracts import ReporterLike


class Reporter(ReporterLike):

    def __init__(self) -> None:
        self.pg_start_: float = 0.0
        self.pg_end_: float = 100.0
        pass

    def set_progress_range(self, start_point: float, end_point: float) -> None:
        self.pg_start_ = start_point
        self.pg_end_ = end_point
        pass

    def format_progress(self, progress: float | None) -> str:
        if progress is None:
            return "N/A"
            pass
        else:
            pg: float = self.pg_start_ + (self.pg_end_ - self.pg_start_) * progress / 100
            return f"{pg:06.2f}"
            pass
        pass

    def format_message(self, message: str | None) -> str:
        if message is None:
            return ""
            pass
        else:
            return message
            pass
        pass

    def message(self, message: str, **payloads: Any) -> None: ...
    
    def progress(self, progress: float, *, message: str | None = None, **payloads: Any) -> None: ...

    pass


class NoneReporter(Reporter):

    def message(self, message: str, **payloads: Any) -> None:
        pass
    
    def progress(self, progress: float, *, message: str | None = None, **payloads: Any) -> None:
        pass

    pass


class ConsoleReporter(Reporter):

    def message(self, message: str, **payloads: Any) -> None:
        msg_str: str = self.format_message(message=message)
        print(f"[MESSAGE] {msg_str}")
        pass
    
    def progress(self, progress: float, *, message: str | None = None, **payloads: Any) -> None:
        pg_str: str = self.format_progress(progress=progress)
        msg_str: str = self.format_message(message=message)
        print(f"[PROGRESS={pg_str}%] {msg_str}")
        pass

    pass


class LossRecordReporter(Reporter):

    def __init__(self) -> None:
        super().__init__()
        self.loss_records_: List[Dict[str, Any]] = []
        pass

    def parse_payloads(self, payloads: Dict[str, Any]) -> Dict[str, Any]:
        loss_record: Dict[str, Any] = {}
        loss_record["epoch"] = payloads.get("epoch", None)
        loss_record["train_loss"] = payloads.get("train_loss", None)
        loss_record["validation_loss"] = payloads.get("validation_loss", None)
        loss_record["epoch_loss"] = payloads.get("epoch_loss", None)
        return loss_record
        pass

    def message(self, message: str, **payloads: Any) -> None:
        loss_record: Dict[str, Any] = self.parse_payloads(payloads=payloads)
        self.loss_records_.append(loss_record)
        pass

    def progress(self, progress: float, *, message: str | None = None, **payloads: Any) -> None:
        loss_record: Dict[str, Any] = self.parse_payloads(payloads=payloads)
        self.loss_records_.append(loss_record)
        pass

    def loss_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_records_)
        pass

    pass