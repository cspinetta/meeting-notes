"""MLX Whisper adapter for Apple Silicon macOS."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from meeting_notes.backends.base import (
    TranscriptionBackend,
    TranscriptionResult,
    normalize_segments,
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

        kwargs: dict[str, object] = {
            "path_or_hf_repo": self.spec.model,
            "task": "transcribe",
            "verbose": self.verbose,
        }
        if language is not None:
            kwargs["language"] = language

        try:
            result: object = mlx_whisper.transcribe(str(audio_path), **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"MLX Whisper failed for {audio_path.name}: {exc}") from exc

        if not isinstance(result, dict):
            raise TranscriptionError("MLX Whisper returned an unexpected result format.")
        detected = result.get("language")
        return TranscriptionResult(
            segments=normalize_segments(result.get("segments")),
            detected_language=str(detected) if detected else None,
        )
