from __future__ import annotations

import pytest

from meeting_notes.errors import PlatformNotSupportedError
from meeting_notes.platform import (
    DEFAULT_MLX_MODEL,
    DEFAULT_OPENAI_MODEL,
    select_backend,
)


def test_selects_mlx_on_apple_silicon() -> None:
    spec = select_backend(sys_platform="darwin", machine="arm64")

    assert spec.key == "mlx-whisper"
    assert spec.model == DEFAULT_MLX_MODEL
    assert spec.device_label == "Apple Silicon / Metal"


@pytest.mark.parametrize(("cuda", "expected"), [(True, "CUDA"), (False, "CPU")])
def test_selects_official_whisper_device_on_linux(cuda: bool, expected: str) -> None:
    spec = select_backend(sys_platform="linux", machine="x86_64", cuda_available=cuda)

    assert spec.key == "openai-whisper"
    assert spec.model == DEFAULT_OPENAI_MODEL
    assert spec.device_label == expected


def test_model_override_is_preserved() -> None:
    spec = select_backend("small", sys_platform="linux", machine="aarch64", cuda_available=False)

    assert spec.model == "small"


def test_intel_mac_fails_clearly() -> None:
    with pytest.raises(PlatformNotSupportedError, match="Intel macOS"):
        select_backend(sys_platform="darwin", machine="x86_64")


def test_unsupported_platform_fails_clearly() -> None:
    with pytest.raises(PlatformNotSupportedError, match="Supported platforms"):
        select_backend(sys_platform="win32", machine="AMD64")
