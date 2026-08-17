from __future__ import annotations

import subprocess
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from meeting_notes.errors import MediaError
from meeting_notes.media import (
    extract_audio,
    inferred_role,
    parse_probe_data,
    render_inspection,
    resolve_speaker_mode,
    valid_extracted_wav,
)


def probe_fixture(audio_count: int = 3) -> dict[str, object]:
    streams: list[dict[str, object]] = [{"index": 0, "codec_type": "video", "codec_name": "h264"}]
    for number in range(audio_count):
        stream: dict[str, object] = {
            "index": number + 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 1 if number == 2 else 2,
            "channel_layout": "mono" if number == 2 else "stereo",
            "sample_rate": "48000",
        }
        if number == 2:
            stream["tags"] = {"title": "My Microphone"}
        streams.append(stream)
    return {
        "format": {
            "format_name": "matroska,webm",
            "format_long_name": "Matroska / WebM",
            "duration": "4054.25",
        },
        "streams": streams,
    }


def test_parses_ffprobe_audio_streams_and_metadata() -> None:
    info = parse_probe_data(Path("meeting.mkv"), probe_fixture())

    assert info.format_name == "matroska,webm"
    assert info.duration == 4054.25
    assert [stream.selector for stream in info.audio_streams] == ["0:a:0", "0:a:1", "0:a:2"]
    assert info.audio_streams[2].metadata == {"title": "My Microphone"}
    assert inferred_role(info.audio_streams[2], 3) == "microphone"


def test_obs_three_track_detection_does_not_require_metadata() -> None:
    info = parse_probe_data(Path("meeting.mkv"), probe_fixture())

    assert resolve_speaker_mode("auto", len(info.audio_streams)) == "obs-3track"
    assert inferred_role(info.audio_streams[0], 3) == "mixed meeting"
    assert inferred_role(info.audio_streams[1], 3) == "remote participants"


def test_auto_falls_back_to_mixed() -> None:
    assert resolve_speaker_mode("auto", 1) == "mixed"


def test_explicit_three_track_rejects_short_recording() -> None:
    with pytest.raises(MediaError, match="requires at least three"):
        resolve_speaker_mode("obs-3track", 2)


def test_inspection_renders_required_fields() -> None:
    info = parse_probe_data(Path("meeting.mkv"), probe_fixture())
    rendered = render_inspection(info)

    assert "Duration: 01:07:34" in rendered
    assert "0:a:1" in rendered
    assert "stream 2" in rendered
    assert "48 kHz" in rendered
    assert "remote participants" in rendered
    assert "metadata title: My Microphone" in rendered


def test_mixed_meeting_metadata_is_not_mistaken_for_google_meet() -> None:
    info = parse_probe_data(Path("meeting.mkv"), probe_fixture())
    mixed = replace(info.audio_streams[0], metadata={"title": "Mixed meeting"})

    assert inferred_role(mixed, 3) == "mixed meeting"


def write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)


def test_extract_audio_atomically_promotes_valid_partial_wav(
    tmp_path: Path, monkeypatch: Any
) -> None:
    output_path = tmp_path / "mixed.wav"
    partial_paths: list[Path] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        partial_path = Path(command[-1])
        partial_paths.append(partial_path)
        assert partial_path != output_path
        assert partial_path.name.endswith(".partial.wav")
        write_test_wav(partial_path)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("meeting_notes.media.subprocess.run", fake_run)

    extract_audio(Path("meeting.mkv"), audio_ordinal=0, output_path=output_path)

    assert valid_extracted_wav(output_path)
    assert len(partial_paths) == 1
    assert not partial_paths[0].exists()


def test_extract_audio_removes_partial_wav_after_interruption(
    tmp_path: Path, monkeypatch: Any
) -> None:
    output_path = tmp_path / "mixed.wav"
    partial_paths: list[Path] = []

    def interrupted_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        partial_path = Path(command[-1])
        partial_paths.append(partial_path)
        write_test_wav(partial_path)
        raise subprocess.CalledProcessError(255, command, stderr="received signal 2")

    monkeypatch.setattr("meeting_notes.media.subprocess.run", interrupted_run)

    with pytest.raises(MediaError, match="received signal 2"):
        extract_audio(Path("meeting.mkv"), audio_ordinal=0, output_path=output_path)

    assert not output_path.exists()
    assert len(partial_paths) == 1
    assert not partial_paths[0].exists()
