"""Transcript segment models, chronological merging, and rendering."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from meeting_notes.formatting import format_timestamp


@dataclass(frozen=True, slots=True)
class Segment:
    """One timestamped utterance, optionally attributed to a track-level speaker."""

    start: float
    end: float
    text: str
    speaker: str | None = None


def with_speaker(segments: Iterable[Segment], speaker: str | None) -> tuple[Segment, ...]:
    return tuple(replace(segment, speaker=speaker) for segment in segments)


def merge_segments(*tracks: Sequence[Segment]) -> tuple[Segment, ...]:
    """Merge tracks chronologically while retaining simultaneous/overlapping speech."""

    indexed = [
        (segment.start, track_index, segment_index, segment)
        for track_index, track in enumerate(tracks)
        for segment_index, segment in enumerate(track)
    ]
    indexed.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(item[3] for item in indexed)


def render_markdown(
    segments: Sequence[Segment],
    *,
    recording_name: str,
    duration: float,
    backend_name: str,
    device: str,
    model: str,
    language: str | None,
    speaker_mode: str,
) -> str:
    """Render the canonical human-readable meeting transcript."""

    language_label = language or "Automatically detected"
    lines = [
        "# Meeting Transcript",
        "",
        f"- Recording: `{recording_name}`",
        f"- Duration: {format_timestamp(duration)}",
        f"- Backend: {backend_name}",
        f"- Device: {device}",
        f"- Model: `{model}`",
        f"- Language: {language_label}",
        f"- Speaker mode: {speaker_mode}",
        "",
        "## Transcript",
        "",
    ]
    if not segments:
        lines.append("_No speech segments were detected._")
    else:
        for segment in segments:
            prefix = f"[{format_timestamp(segment.start)}]"
            if segment.speaker:
                lines.append(f"{prefix} {segment.speaker}: {segment.text}")
            else:
                lines.append(f"{prefix} {segment.text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def transcript_document(
    segments: Sequence[Segment],
    *,
    recording_name: str,
    language: str | None,
    speaker_mode: str,
) -> dict[str, Any]:
    """Build the stable machine-readable transcript document."""

    return {
        "schema_version": 1,
        "recording": recording_name,
        "language": language,
        "speaker_mode": speaker_mode,
        "segments": [asdict(segment) for segment in segments],
    }


def render_json(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def is_valid_transcript_document(value: object) -> bool:
    """Validate enough structure to prevent accidental reuse of partial output."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return False
    segments = value.get("segments")
    if not isinstance(segments, list):
        return False
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        if not isinstance(segment.get("text"), str):
            return False
        if isinstance(segment.get("start"), bool) or not isinstance(
            segment.get("start"), (int, float)
        ):
            return False
        if isinstance(segment.get("end"), bool) or not isinstance(segment.get("end"), (int, float)):
            return False
        start = float(segment["start"])
        end = float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            return False
        if segment.get("speaker") is not None and not isinstance(segment.get("speaker"), str):
            return False
    return True
