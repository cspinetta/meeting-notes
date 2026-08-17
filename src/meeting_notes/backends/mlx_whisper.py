"""MLX Whisper adapter for Apple Silicon macOS."""

from __future__ import annotations

import importlib
import os
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


class MLXWhisperBackend(TranscriptionBackend):
    """Run mlx-whisper locally through Apple's MLX framework."""

    def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
        # Model download is allowed, but usage telemetry is not part of this tool's privacy model.
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            mlx_whisper: Any = importlib.import_module("mlx_whisper")
        except ImportError as exc:
            raise DependencyError(
                "mlx-whisper is not installed. From the project root, run `uv sync`."
            ) from exc

        try:
            detected_language = language
            if detected_language is None:
                with language_probe(audio_path) as probe_path:
                    probe_options = whisper_decode_options(language=None, verbose=False)
                    probe_options["path_or_hf_repo"] = self.spec.model
                    probe_result: object = mlx_whisper.transcribe(str(probe_path), **probe_options)
                if not isinstance(probe_result, dict) or not probe_result.get("language"):
                    raise TranscriptionError(
                        "MLX Whisper could not detect a language from the voiced audio probe."
                    )
                detected_language = str(probe_result["language"])

            kwargs = whisper_decode_options(language=detected_language, verbose=self.verbose)
            kwargs["path_or_hf_repo"] = self.spec.model
            result: object = mlx_whisper.transcribe(str(audio_path), **kwargs)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"MLX Whisper failed for {audio_path.name}: {exc}") from exc

        if not isinstance(result, dict):
            raise TranscriptionError("MLX Whisper returned an unexpected result format.")
        normalized = normalize_segments(result.get("segments"))
        return TranscriptionResult(
            segments=filter_near_silent_segments(audio_path, normalized),
            detected_language=detected_language,
        )
