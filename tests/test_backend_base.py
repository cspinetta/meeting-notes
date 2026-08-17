from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path

import pytest

from meeting_notes.backends.base import (
    filter_near_silent_segments,
    language_probe,
    normalize_segments,
    select_language_probe_start,
)
from meeting_notes.errors import TranscriptionError
from meeting_notes.transcript import Segment


def write_pcm_wav(path: Path, sections: list[tuple[int, int]]) -> None:
    samples = array("h")
    for amplitude, seconds in sections:
        samples.extend([amplitude] * (16_000 * seconds))
    if sys.byteorder == "big":
        samples.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


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


def test_language_probe_skips_leading_silence(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.wav"
    write_pcm_wav(audio_path, [(0, 6), (10_000, 5), (0, 30)])

    assert select_language_probe_start(audio_path) == 6.0
    with language_probe(audio_path) as probe_path:
        assert probe_path.is_file()
        assert probe_path != audio_path
        with wave.open(str(probe_path), "rb") as probe:
            assert probe.getnframes() == 30 * 16_000
            assert probe.readframes(1) != b"\0\0"

    assert not probe_path.exists()


def test_near_silent_segments_are_removed(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.wav"
    write_pcm_wav(audio_path, [(1, 1), (10_000, 1)])
    silent = Segment(0.0, 1.0, "Thank you.")
    speech = Segment(1.0, 2.0, "Actual speech.")

    assert filter_near_silent_segments(audio_path, (silent, speech)) == (speech,)
