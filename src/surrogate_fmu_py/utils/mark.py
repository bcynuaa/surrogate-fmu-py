from __future__ import annotations

import datetime
import uuid

def time_stamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string in the format YYYYMMDD_HHMMSS."""
    return datetime.datetime.now().strftime(format)
    pass

def get_token(text: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_DNS, text).hex
    pass