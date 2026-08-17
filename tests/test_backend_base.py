from __future__ import annotations

import math

import pytest

from meeting_notes.backends.base import normalize_segments
from meeting_notes.errors import TranscriptionError


def test_normalize_segments_rejects_missing_segment_list() -> None:
    with pytest.raises(TranscriptionError, match="segment list"):
        normalize_segments(None)


@pytest.mark.parametrize(
    "segments",
    [
        ["not an object"],
        [{"start": "bad", "end": 1.0, "text": "speech"}],
        [{"start": 0.0, "end": 1.0, "text": None}],
        [{"start": True, "end": 1.0, "text": "speech"}],
        [{"start": math.nan, "end": 1.0, "text": "speech"}],
        [{"start": 0.0, "end": math.inf, "text": "speech"}],
    ],
)
def test_normalize_segments_rejects_malformed_entries(segments: object) -> None:
    with pytest.raises(TranscriptionError):
        normalize_segments(segments)


def test_normalize_segments_clamps_negative_times_sequentially() -> None:
    segments = normalize_segments([{"start": -5.0, "end": -3.0, "text": " speech "}])

    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 0.0
    assert segments[0].text == "speech"
