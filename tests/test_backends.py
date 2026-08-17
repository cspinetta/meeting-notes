from __future__ import annotations

import sys
import wave
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from meeting_notes.backends.mlx_whisper import MLXWhisperBackend
from meeting_notes.backends.openai_whisper import OpenAIWhisperBackend
from meeting_notes.platform import BackendSpec


def write_speech_wav(path: Path) -> None:
    samples = array("h", [10_000] * 16_000)
    if sys.byteorder == "big":
        samples.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


@contextmanager
def same_file_probe(path: Path) -> Iterator[Path]:
    yield path


def mlx_spec() -> BackendSpec:
    return BackendSpec(
        key="mlx-whisper",
        name="MLX Whisper",
        model="test-model",
        device="metal",
        device_label="Apple Silicon / Metal",
        os_name="macOS",
        architecture="arm64",
    )


def openai_spec() -> BackendSpec:
    return BackendSpec(
        key="openai-whisper",
        name="OpenAI Whisper",
        model="tiny",
        device="cpu",
        device_label="CPU",
        os_name="Linux",
        architecture="x86_64",
    )


def fake_result(*, include_segments: bool) -> dict[str, object]:
    segments: list[dict[str, object]] = []
    if include_segments:
        segments.append({"start": 0.0, "end": 1.0, "text": " hola "})
    return {"language": "es", "segments": segments}


def assert_calls_use_safe_auto_language_options(calls: list[dict[str, object]]) -> None:
    assert len(calls) == 2
    assert "language" not in calls[0]
    assert calls[1]["language"] == "es"
    for kwargs in calls:
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["word_timestamps"] is True
        assert kwargs["hallucination_silence_threshold"] == 2.0


def test_mlx_backend_detects_language_from_probe_and_reuses_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    audio_path = tmp_path / "remote.wav"
    write_speech_wav(audio_path)
    calls: list[dict[str, object]] = []

    def transcribe(_path: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return fake_result(include_segments=len(calls) == 2)

    monkeypatch.setattr("meeting_notes.backends.mlx_whisper.language_probe", same_file_probe)
    monkeypatch.setattr(
        "meeting_notes.backends.mlx_whisper.importlib.import_module",
        lambda _name: SimpleNamespace(transcribe=transcribe),
    )

    result = MLXWhisperBackend(mlx_spec()).transcribe(audio_path, None)

    assert result.detected_language == "es"
    assert [segment.text for segment in result.segments] == ["hola"]
    assert_calls_use_safe_auto_language_options(calls)
    assert all(call["path_or_hf_repo"] == "test-model" for call in calls)


def test_openai_backend_detects_language_from_probe_and_reuses_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    audio_path = tmp_path / "remote.wav"
    write_speech_wav(audio_path)
    calls: list[dict[str, object]] = []

    class FakeModel:
        def transcribe(self, _path: str, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return fake_result(include_segments=len(calls) == 2)

    monkeypatch.setattr("meeting_notes.backends.openai_whisper.language_probe", same_file_probe)
    monkeypatch.setattr(
        "meeting_notes.backends.openai_whisper.importlib.import_module",
        lambda _name: SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeModel()),
    )

    result = OpenAIWhisperBackend(openai_spec()).transcribe(audio_path, None)

    assert result.detected_language == "es"
    assert [segment.text for segment in result.segments] == ["hola"]
    assert_calls_use_safe_auto_language_options(calls)
    assert all(call["fp16"] is False for call in calls)
