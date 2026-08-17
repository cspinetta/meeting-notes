"""Shared presentation helpers."""

from __future__ import annotations

import math


def format_timestamp(seconds: float) -> str:
    """Format nonnegative seconds as an hours:minutes:seconds timestamp."""

    total = max(0, math.floor(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
