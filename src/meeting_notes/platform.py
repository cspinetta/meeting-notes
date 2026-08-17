"""Platform detection and local Whisper backend selection."""

from __future__ import annotations

import importlib
import platform as stdlib_platform
import sys
from dataclasses import dataclass
from typing import Any

from meeting_notes.errors import PlatformNotSupportedError

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_OPENAI_MODEL = "turbo"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """The selected local transcription implementation and runtime device."""

    key: str
    name: str
    model: str
    device: str
    device_label: str
    os_name: str
    architecture: str


def _cuda_available() -> bool:
    """Ask PyTorch about CUDA without importing it on non-Linux platforms."""

    try:
        torch: Any = importlib.import_module("torch")
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def select_backend(
    model: str | None = None,
    *,
    sys_platform: str | None = None,
    machine: str | None = None,
    cuda_available: bool | None = None,
) -> BackendSpec:
    """Select the required local backend for the current operating system."""

    current_platform = sys_platform if sys_platform is not None else sys.platform
    current_machine = machine if machine is not None else stdlib_platform.machine()

    if current_platform == "darwin":
        if current_machine != "arm64":
            raise PlatformNotSupportedError(
                "Intel macOS is not supported. meeting-notes currently uses MLX Whisper "
                "and therefore requires Apple Silicon (arm64) on macOS."
            )
        return BackendSpec(
            key="mlx-whisper",
            name="MLX Whisper",
            model=model or DEFAULT_MLX_MODEL,
            device="metal",
            device_label="Apple Silicon / Metal",
            os_name="macOS",
            architecture=current_machine,
        )

    if current_platform.startswith("linux"):
        has_cuda = _cuda_available() if cuda_available is None else cuda_available
        return BackendSpec(
            key="openai-whisper",
            name="OpenAI Whisper",
            model=model or DEFAULT_OPENAI_MODEL,
            device="cuda" if has_cuda else "cpu",
            device_label="CUDA" if has_cuda else "CPU",
            os_name="Linux",
            architecture=current_machine,
        )

    raise PlatformNotSupportedError(
        f"Unsupported operating system {current_platform!r} ({current_machine}). "
        "Supported platforms are Apple Silicon macOS and Linux."
    )
