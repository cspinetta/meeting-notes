"""Media inspection and lossless-to-source audio extraction helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from meeting_notes.errors import DependencyError, MediaError
from meeting_notes.formatting import format_timestamp

SpeakerMode = Literal["auto", "mixed", "obs-3track"]
ResolvedSpeakerMode = Literal["mixed", "obs-3track"]


@dataclass(frozen=True, slots=True)
class AudioStream:
    """One audio stream reported by ffprobe."""

    ordinal: int
    index: int
    codec: str
    channels: int | None
    channel_layout: str | None
    sample_rate: int | None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def selector(self) -> str:
        return f"0:a:{self.ordinal}"


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """Relevant container and audio information about a recording."""

    path: Path
    format_name: str
    format_long_name: str | None
    duration: float
    audio_streams: tuple[AudioStream, ...]


def dependency_install_instructions() -> str:
    """Return platform-appropriate, non-automatic ffmpeg installation guidance."""

    if sys.platform == "darwin":
        return "Install it with Homebrew: `brew install ffmpeg`."
    if sys.platform.startswith("linux"):
        return (
            "Install it with your system package manager, for example "
            "`sudo apt install ffmpeg`, `sudo dnf install ffmpeg`, or "
            "`sudo pacman -S ffmpeg`."
        )
    return "Install ffmpeg (which includes ffprobe) using your operating system package manager."


def ensure_system_dependencies() -> None:
    """Require both ffmpeg executables without attempting to install them."""

    missing = [command for command in ("ffmpeg", "ffprobe") if shutil.which(command) is None]
    if missing:
        names = ", ".join(missing)
        raise DependencyError(
            f"Missing required system command(s): {names}. " + dependency_install_instructions()
        )


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_probe_data(path: Path, data: dict[str, Any]) -> RecordingInfo:
    """Parse ffprobe JSON into typed recording metadata."""

    format_data = data.get("format")
    if not isinstance(format_data, dict):
        raise MediaError("ffprobe did not return container format information.")

    duration = _optional_float(format_data.get("duration"))
    if duration is None:
        duration = 0.0

    raw_streams = data.get("streams")
    if not isinstance(raw_streams, list):
        raw_streams = []

    audio_streams: list[AudioStream] = []
    for stream in raw_streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags")
        metadata = (
            {str(key): str(value) for key, value in tags.items()} if isinstance(tags, dict) else {}
        )
        audio_streams.append(
            AudioStream(
                ordinal=len(audio_streams),
                index=_optional_int(stream.get("index")) or 0,
                codec=str(stream.get("codec_name") or "unknown"),
                channels=_optional_int(stream.get("channels")),
                channel_layout=(
                    str(stream["channel_layout"]) if stream.get("channel_layout") else None
                ),
                sample_rate=_optional_int(stream.get("sample_rate")),
                metadata=metadata,
            )
        )

    if not audio_streams:
        raise MediaError("The recording contains no audio streams.")

    return RecordingInfo(
        path=path,
        format_name=str(format_data.get("format_name") or "unknown"),
        format_long_name=(
            str(format_data["format_long_name"]) if format_data.get("format_long_name") else None
        ),
        duration=max(0.0, duration),
        audio_streams=tuple(audio_streams),
    )


def probe_recording(path: Path, *, verbose: bool = False) -> RecordingInfo:
    """Inspect a local recording with ffprobe."""

    recording = path.expanduser()
    if not recording.is_file():
        raise MediaError(f"Recording does not exist or is not a file: {recording}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(recording),
    ]
    if verbose:
        print("Running ffprobe...", file=sys.stderr)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise DependencyError(
            "ffprobe was not found. " + dependency_install_instructions()
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise MediaError(f"Could not inspect {recording.name}: {detail}") from exc

    try:
        parsed: object = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise MediaError("ffprobe returned an unexpected JSON value.")
    return parse_probe_data(recording.resolve(), parsed)


def resolve_speaker_mode(requested: SpeakerMode, audio_stream_count: int) -> ResolvedSpeakerMode:
    """Resolve automatic OBS layout selection, validating explicit requests."""

    if requested == "auto":
        return "obs-3track" if audio_stream_count >= 3 else "mixed"
    if requested == "obs-3track" and audio_stream_count < 3:
        raise MediaError(
            "--speaker-mode obs-3track requires at least three audio streams: "
            "track 1 mixed, track 2 remote participants, and track 3 microphone."
        )
    return requested


def inferred_role(stream: AudioStream, stream_count: int) -> str:
    """Infer the conventional OBS role, using metadata only as a helpful hint."""

    metadata_text = " ".join(stream.metadata.values()).casefold()
    metadata_words = set(re.findall(r"[a-z]+", metadata_text))
    if {"microphone", "mic"} & metadata_words or "my voice" in metadata_text:
        return "microphone"
    if {"mixed", "mix", "everyone", "master"} & metadata_words:
        return "mixed meeting"
    if {
        "remote",
        "desktop",
        "participant",
        "participants",
    } & metadata_words or "google meet" in metadata_text:
        return "remote participants"

    if stream_count >= 3:
        conventional = {0: "mixed meeting", 1: "remote participants", 2: "microphone"}
        return conventional.get(stream.ordinal, "additional audio track")
    return "mixed meeting" if stream.ordinal == 0 else "additional audio track"


def render_inspection(info: RecordingInfo) -> str:
    """Render concise, human-readable ffprobe information."""

    lines = [
        f"Recording: {info.path}",
        f"Container: {info.format_long_name or info.format_name} ({info.format_name})",
        f"Duration: {format_timestamp(info.duration)}",
        "",
        "Audio streams:",
    ]
    count = len(info.audio_streams)
    for stream in info.audio_streams:
        sample_rate = f"{stream.sample_rate / 1000:g} kHz" if stream.sample_rate else "unknown rate"
        channels = stream.channel_layout or (
            f"{stream.channels} channel(s)" if stream.channels is not None else "unknown channels"
        )
        lines.append(
            f"  {stream.selector:<7} stream {stream.index:<3} {stream.codec.upper():<8} "
            f"{sample_rate:<10} "
            f"{channels:<12} inferred: {inferred_role(stream, count)}"
        )
        for key, value in sorted(stream.metadata.items()):
            lines.append(f"           metadata {key}: {value}")

    inferred = "obs-3track" if count >= 3 else "mixed"
    lines.extend(("", f"Inferred OBS layout: {inferred}"))
    if inferred == "obs-3track":
        lines.append("  Track 1: mixed; Track 2: remote participant(s); Track 3: microphone")
    else:
        lines.append("  The first audio stream will be transcribed as a mixed recording.")
    return "\n".join(lines) + "\n"


def valid_extracted_wav(path: Path) -> bool:
    """Check whether an existing extraction is usable for resumable processing."""

    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getnchannels() == 1
                and wav.getframerate() == 16_000
                and wav.getsampwidth() == 2
                and wav.getnframes() > 0
            )
    except (EOFError, wave.Error, OSError):
        return False


def extract_audio(
    recording: Path,
    *,
    audio_ordinal: int,
    output_path: Path,
    verbose: bool = False,
) -> None:
    """Atomically extract one stream as mono 16 kHz signed 16-bit PCM."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}-",
        suffix=".partial.wav",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        partial_path = Path(temporary_file.name)

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning" if verbose else "error",
        "-y",
        "-i",
        str(recording),
        "-map",
        f"0:a:{audio_ordinal}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(partial_path),
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise DependencyError(
                "ffmpeg was not found. " + dependency_install_instructions()
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown ffmpeg error"
            raise MediaError(
                f"Could not extract audio stream 0:a:{audio_ordinal}: {detail}"
            ) from exc

        if not valid_extracted_wav(partial_path):
            raise MediaError(f"ffmpeg produced an invalid WAV file: {output_path}")
        try:
            partial_path.replace(output_path)
        except OSError as exc:
            raise MediaError(f"Could not finalize extracted audio: {output_path}") from exc
    finally:
        with suppress(OSError):
            partial_path.unlink(missing_ok=True)

    if verbose and completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)
