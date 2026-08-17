"""Official OpenAI Whisper adapter for Linux local inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from meeting_notes.backends.base import (
    TranscriptionBackend,
    TranscriptionResult,
    normalize_segments,
)
from meeting_notes.errors import DependencyError, TranscriptionError


class OpenAIWhisperBackend(TranscriptionBackend):
    """Run the official openai-whisper implementation entirely locally."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            whisper: Any = importlib.import_module("whisper")
        except ImportError as exc:
            raise DependencyError(
                "openai-whisper is not installed. From the project root, run `uv sync`."
            ) from exc
        try:
            self._model: Any = whisper.load_model(self.spec.model, device=self.spec.device)
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load OpenAI Whisper model {self.spec.model!r}: {exc}"
            ) from exc

    def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
        kwargs: dict[str, object] = {
            "task": "transcribe",
            "verbose": None if not self.verbose else True,
            "fp16": self.spec.device == "cuda",
        }
        if language is not None:
            kwargs["language"] = language
        try:
            result: object = self._model.transcribe(str(audio_path), **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"OpenAI Whisper failed for {audio_path.name}: {exc}") from exc

        if not isinstance(result, dict):
            raise TranscriptionError("OpenAI Whisper returned an unexpected result format.")
        detected = result.get("language")
        return TranscriptionResult(
            segments=normalize_segments(result.get("segments")),
            detected_language=str(detected) if detected else None,
        )
