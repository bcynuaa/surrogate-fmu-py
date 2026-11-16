from __future__ import annotations

import datetime


def time_stamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string in the format YYYYMMDD_HHMMSS."""
    return datetime.datetime.now().strftime(format)
    pass