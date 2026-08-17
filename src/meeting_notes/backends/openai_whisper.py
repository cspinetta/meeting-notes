"""Official OpenAI Whisper adapter for Linux local inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from meeting_notes.backends.base import (
    TranscriptionBackend,
    TranscriptionResult,
    filter_near_silent_segments,
    language_probe,
    normalize_segments,
    whisper_decode_options,
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
        try:
            detected_language = language
            if detected_language is None:
                with language_probe(audio_path) as probe_path:
                    probe_options = whisper_decode_options(language=None, verbose=False)
                    probe_options["fp16"] = self.spec.device == "cuda"
                    probe_result: object = self._model.transcribe(str(probe_path), **probe_options)
                if not isinstance(probe_result, dict) or not probe_result.get("language"):
                    raise TranscriptionError(
                        "OpenAI Whisper could not detect a language from the voiced audio probe."
                    )
                detected_language = str(probe_result["language"])

            kwargs = whisper_decode_options(language=detected_language, verbose=self.verbose)
            kwargs["fp16"] = self.spec.device == "cuda"
            result: object = self._model.transcribe(str(audio_path), **kwargs)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"OpenAI Whisper failed for {audio_path.name}: {exc}") from exc

        if not isinstance(result, dict):
            raise TranscriptionError("OpenAI Whisper returned an unexpected result format.")
        normalized = normalize_segments(result.get("segments"))
        return TranscriptionResult(
            segments=filter_near_silent_segments(audio_path, normalized),
            detected_language=detected_language,
        )
