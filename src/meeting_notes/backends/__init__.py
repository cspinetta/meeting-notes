"""Factory for local transcription backends."""

from meeting_notes.backends.base import TranscriptionBackend
from meeting_notes.backends.mlx_whisper import MLXWhisperBackend
from meeting_notes.backends.openai_whisper import OpenAIWhisperBackend
from meeting_notes.errors import PlatformNotSupportedError
from meeting_notes.platform import BackendSpec


def create_backend(spec: BackendSpec, *, verbose: bool = False) -> TranscriptionBackend:
    """Instantiate the backend selected for this platform."""

    if spec.key == "mlx-whisper":
        return MLXWhisperBackend(spec, verbose=verbose)
    if spec.key == "openai-whisper":
        return OpenAIWhisperBackend(spec, verbose=verbose)
    raise PlatformNotSupportedError(f"Unknown transcription backend: {spec.key}")


__all__ = ["TranscriptionBackend", "create_backend"]
