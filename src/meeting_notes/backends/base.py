"""Abstract interface shared by local Whisper implementations."""

from __future__ import annotations

import math
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from array import array
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from meeting_notes.errors import TranscriptionError
from meeting_notes.platform import BackendSpec
from meeting_notes.transcript import Segment

LANGUAGE_PROBE_SECONDS = 30
LANGUAGE_PROBE_BLOCK_SECONDS = 1
LANGUAGE_PROBE_ACTIVE_RUN_SECONDS = 3
LANGUAGE_PROBE_ACTIVE_RMS_DBFS = -45.0
SILENT_SEGMENT_RMS_DBFS = -55.0
_PCM16_FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized transcription output from a backend."""

    segments: tuple[Segment, ...]
    detected_language: str | None = None


class TranscriptionBackend(ABC):
    """A stateful, reusable local speech-to-text backend."""

    def __init__(self, spec: BackendSpec, *, verbose: bool = False) -> None:
        self.spec = spec
        self.verbose = verbose

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
        """Transcribe one local WAV file without translating it."""


def whisper_decode_options(*, language: str | None, verbose: bool) -> dict[str, object]:
    """Return shared decoding defaults that resist silence/repetition hallucinations."""

    options: dict[str, object] = {
        "task": "transcribe",
        "verbose": True if verbose else None,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
    }
    if language is not None:
        options["language"] = language
    return options


def _pcm16_mean_square(data: bytes) -> float:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return sum(float(sample) * sample for sample in samples) / len(samples)


def _validate_probe_wav(reader: wave.Wave_read, audio_path: Path) -> int:
    if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
        raise TranscriptionError(
            f"Expected mono 16-bit PCM for transcription diagnostics: {audio_path.name}"
        )
    sample_rate = reader.getframerate()
    if sample_rate <= 0:
        raise TranscriptionError(f"Invalid sample rate in extracted audio: {audio_path.name}")
    return sample_rate


def select_language_probe_start(audio_path: Path) -> float:
    """Find an early sustained voiced region for Whisper language detection."""

    active_threshold = (_PCM16_FULL_SCALE * 10 ** (LANGUAGE_PROBE_ACTIVE_RMS_DBFS / 20.0)) ** 2
    try:
        with wave.open(str(audio_path), "rb") as reader:
            sample_rate = _validate_probe_wav(reader, audio_path)
            block_frames = sample_rate * LANGUAGE_PROBE_BLOCK_SECONDS
            window_blocks = LANGUAGE_PROBE_SECONDS // LANGUAGE_PROBE_BLOCK_SECONDS
            energies: deque[float] = deque()
            rolling_energy = 0.0
            best_energy = -1.0
            best_start_block = 0
            active_run = 0
            active_run_start = 0
            block_index = 0

            while data := reader.readframes(block_frames):
                mean_square = _pcm16_mean_square(data)
                energies.append(mean_square)
                rolling_energy += mean_square
                if len(energies) > window_blocks:
                    rolling_energy -= energies.popleft()
                if len(energies) == window_blocks and rolling_energy > best_energy:
                    best_energy = rolling_energy
                    best_start_block = block_index - window_blocks + 1

                if mean_square >= active_threshold:
                    if active_run == 0:
                        active_run_start = block_index
                    active_run += 1
                    if active_run >= LANGUAGE_PROBE_ACTIVE_RUN_SECONDS:
                        return float(active_run_start * LANGUAGE_PROBE_BLOCK_SECONDS)
                else:
                    active_run = 0
                block_index += 1
    except (EOFError, OSError, wave.Error) as exc:
        raise TranscriptionError(
            f"Could not analyze extracted audio {audio_path.name}: {exc}"
        ) from exc

    return float(best_start_block * LANGUAGE_PROBE_BLOCK_SECONDS)


@contextmanager
def language_probe(audio_path: Path) -> Iterator[Path]:
    """Yield a temporary voiced excerpt used only for automatic language detection."""

    start_seconds = select_language_probe_start(audio_path)
    temporary_path: Path | None = None
    try:
        with wave.open(str(audio_path), "rb") as reader:
            sample_rate = _validate_probe_wav(reader, audio_path)
            start_frame = min(
                round(start_seconds * sample_rate),
                max(0, reader.getnframes() - 1),
            )
            reader.setpos(start_frame)
            probe_frames = reader.readframes(sample_rate * LANGUAGE_PROBE_SECONDS)

        with tempfile.NamedTemporaryFile(
            prefix=f".{audio_path.stem}-language-",
            suffix=".wav",
            dir=audio_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with wave.open(str(temporary_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(probe_frames)
        yield temporary_path
    except (EOFError, OSError, wave.Error) as exc:
        raise TranscriptionError(
            f"Could not prepare a language probe for {audio_path.name}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def filter_near_silent_segments(
    audio_path: Path, segments: tuple[Segment, ...]
) -> tuple[Segment, ...]:
    """Remove Whisper text aligned to effectively silent PCM without touching real speech."""

    silence_threshold = (_PCM16_FULL_SCALE * 10 ** (SILENT_SEGMENT_RMS_DBFS / 20.0)) ** 2
    retained: list[Segment] = []
    try:
        with wave.open(str(audio_path), "rb") as reader:
            sample_rate = _validate_probe_wav(reader, audio_path)
            total_frames = reader.getnframes()
            for segment in segments:
                start_frame = min(total_frames, max(0, round(segment.start * sample_rate)))
                end_frame = min(
                    total_frames, max(start_frame + 1, round(segment.end * sample_rate))
                )
                if start_frame >= total_frames:
                    continue
                reader.setpos(start_frame)
                data = reader.readframes(end_frame - start_frame)
                if _pcm16_mean_square(data) >= silence_threshold:
                    retained.append(segment)
    except (EOFError, OSError, wave.Error) as exc:
        raise TranscriptionError(f"Could not validate speech in {audio_path.name}: {exc}") from exc
    return tuple(retained)


def normalize_segments(raw_segments: object) -> tuple[Segment, ...]:
    """Normalize the segment dictionaries returned by either Whisper package."""

    if not isinstance(raw_segments, list):
        raise TranscriptionError("Whisper returned a missing or malformed segment list.")

    normalized: list[Segment] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise TranscriptionError(f"Whisper segment {index} is not an object.")
        if isinstance(item.get("start"), bool) or isinstance(item.get("end"), bool):
            raise TranscriptionError(f"Whisper segment {index} has invalid timestamps.")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, OverflowError, TypeError, ValueError):
            raise TranscriptionError(f"Whisper segment {index} has invalid timestamps.") from None
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            raise TranscriptionError(f"Whisper segment {index} has invalid text.")
        if not math.isfinite(start) or not math.isfinite(end):
            raise TranscriptionError(f"Whisper segment {index} has non-finite timestamps.")
        start = max(0.0, start)
        end = max(start, end)
        text = " ".join(raw_text.split())
        if not text:
            continue
        normalized.append(Segment(start=start, end=end, text=text))
    return tuple(normalized)
