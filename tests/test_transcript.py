from __future__ import annotations

import json

from meeting_notes.transcript import (
    Segment,
    merge_segments,
    render_json,
    render_markdown,
    transcript_document,
)


def test_merge_sorts_chronologically_and_preserves_overlap() -> None:
    remote = (
        Segment(10.0, 15.0, "Remote starts.", "Other participant(s)"),
        Segment(20.0, 21.0, "Remote ends.", "Other participant(s)"),
    )
    me = (Segment(12.0, 16.0, "I overlap.", "Me"),)

    merged = merge_segments(remote, me)

    assert [segment.text for segment in merged] == [
        "Remote starts.",
        "I overlap.",
        "Remote ends.",
    ]
    assert len(merged) == 3


def test_markdown_renders_timestamps_and_speakers() -> None:
    markdown = render_markdown(
        [Segment(74.2, 80.8, "Deployment is the issue.", "Cris")],
        recording_name="meeting.mkv",
        duration=4054.0,
        backend_name="MLX Whisper",
        device="Apple Silicon / Metal",
        model="large",
        language="en",
        speaker_mode="obs-3track",
    )

    assert markdown.startswith("# Meeting Transcript\n")
    assert "- Duration: 01:07:34" in markdown
    assert "[00:01:14] Cris: Deployment is the issue." in markdown


def test_json_rendering_contains_structured_segments() -> None:
    document = transcript_document(
        [Segment(74.2, 80.8, "Deployment is the issue.", "Me")],
        recording_name="meeting.mkv",
        language="en",
        speaker_mode="obs-3track",
    )
    parsed = json.loads(render_json(document))

    assert parsed["segments"] == [
        {
            "speaker": "Me",
            "start": 74.2,
            "end": 80.8,
            "text": "Deployment is the issue.",
        }
    ]
