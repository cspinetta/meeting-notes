"""Abstract interface shared by local Whisper implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from meeting_notes.errors import TranscriptionError
from meeting_notes.platform import BackendSpec
from meeting_notes.transcript import Segment


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized transcription output from a backend."""

    segments: tuple[Segment, ...]
    detected_language: str | None = None


class TranscriptionBackend(ABC):
    """A stateful, reusable local speech-to-text backend."""

    def __init__(self, spec: BackendSpec, *, verbose: bool = False) -> None:
        self.spec = spec
        self.verbose = verbose

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
        """Transcribe one local WAV file without translating it."""


def normalize_segments(raw_segments: object) -> tuple[Segment, ...]:
    """Normalize the segment dictionaries returned by either Whisper package."""

    if not isinstance(raw_segments, list):
        raise TranscriptionError("Whisper returned a missing or malformed segment list.")

    normalized: list[Segment] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise TranscriptionError(f"Whisper segment {index} is not an object.")
        if isinstance(item.get("start"), bool) or isinstance(item.get("end"), bool):
            raise TranscriptionError(f"Whisper segment {index} has invalid timestamps.")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, OverflowError, TypeError, ValueError):
            raise TranscriptionError(f"Whisper segment {index} has invalid timestamps.") from None
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            raise TranscriptionError(f"Whisper segment {index} has invalid text.")
        if not math.isfinite(start) or not math.isfinite(end):
            raise TranscriptionError(f"Whisper segment {index} has non-finite timestamps.")
        start = max(0.0, start)
        end = max(start, end)
        text = " ".join(raw_text.split())
        if not text:
            continue
        normalized.append(Segment(start=start, end=end, text=text))
    return tuple(normalized)
