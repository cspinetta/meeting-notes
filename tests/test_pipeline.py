from __future__ import annotations

import json
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from meeting_notes.backends.base import TranscriptionResult
from meeting_notes.errors import ArtifactError, MediaError
from meeting_notes.media import AudioStream, RecordingInfo
from meeting_notes.pipeline import ProcessOptions, choose_run_directory, run_process
from meeting_notes.platform import BackendSpec
from meeting_notes.transcript import Segment


def backend_spec() -> BackendSpec:
    return BackendSpec(
        key="fake",
        name="Fake Whisper",
        model="fake-model",
        device="cpu",
        device_label="CPU",
        os_name="Linux",
        architecture="x86_64",
    )


def write_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)


def test_output_path_collision_and_matching_resume(tmp_path: Path) -> None:
    root = tmp_path / "output"
    base = root / "meeting"
    base.mkdir(parents=True)
    (base / "metadata.json").write_text(json.dumps({"resume_key": {"id": "one"}}), encoding="utf-8")

    matching, resumes = choose_run_directory(root, "meeting", {"id": "one"}, force=False)
    collision, collision_resumes = choose_run_directory(root, "meeting", {"id": "two"}, force=False)
    forced, forced_resumes = choose_run_directory(root, "meeting", {"id": "one"}, force=True)

    assert (matching, resumes) == (base, True)
    assert (collision, collision_resumes) == (root / "meeting-2", False)
    assert (forced, forced_resumes) == (root / "meeting-3", False)
    assert collision.is_dir()
    assert forced.is_dir()


def test_concurrent_run_directory_reservations_are_unique(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    barrier = Barrier(2)

    def reserve() -> Path:
        barrier.wait()
        return choose_run_directory(root, "meeting", {"id": "same"}, force=True)[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = {future.result() for future in (executor.submit(reserve), executor.submit(reserve))}

    assert paths == {root / "meeting", root / "meeting-2"}
    assert all(path.is_dir() for path in paths)


def test_completed_transcript_is_resumed_without_whisper(tmp_path: Path, monkeypatch: Any) -> None:
    recording = tmp_path / "meeting.mkv"
    recording.write_bytes(b"local fake recording")
    info = RecordingInfo(
        path=recording,
        format_name="matroska",
        format_long_name="Matroska",
        duration=90.0,
        audio_streams=(AudioStream(0, 1, "aac", 2, "stereo", 48_000),),
    )
    calls = {"transcribe": 0, "extract": 0}

    class FakeBackend:
        def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
            assert audio_path.is_file()
            calls["transcribe"] += 1
            return TranscriptionResult((Segment(1.5, 3.0, "A local test segment."),), "en")

    def fake_extract(
        recording_path: Path,
        *,
        audio_ordinal: int,
        output_path: Path,
        verbose: bool,
    ) -> None:
        del recording_path, audio_ordinal, verbose
        calls["extract"] += 1
        write_test_wav(output_path)

    monkeypatch.setattr("meeting_notes.pipeline.ensure_system_dependencies", lambda: None)
    monkeypatch.setattr("meeting_notes.pipeline.probe_recording", lambda path, verbose=False: info)
    monkeypatch.setattr("meeting_notes.pipeline.select_backend", lambda model=None: backend_spec())
    monkeypatch.setattr(
        "meeting_notes.pipeline.create_backend", lambda spec, verbose=False: FakeBackend()
    )
    monkeypatch.setattr("meeting_notes.pipeline.extract_audio", fake_extract)

    options = ProcessOptions(recording=recording, output_root=tmp_path / "output")
    first = run_process(options)
    assert not first.resumed
    assert calls == {"transcribe": 1, "extract": 1}
    assert not (first.run_directory / "mixed.wav").exists()
    assert (first.run_directory / "summary-prompt.md").is_file()

    def fail_if_backend_created(spec: BackendSpec, *, verbose: bool = False) -> FakeBackend:
        del spec, verbose
        raise AssertionError("Whisper reran")

    monkeypatch.setattr("meeting_notes.pipeline.create_backend", fail_if_backend_created)
    messages: list[str] = []
    second = run_process(replace(options, keep_audio=True), progress=messages.append)

    assert second.resumed
    assert second.run_directory == first.run_directory
    assert calls == {"transcribe": 1, "extract": 1}
    assert any("--force" in message for message in messages)
    metadata = json.loads((second.run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["input_filename"] == "meeting.mkv"
    assert "input_path" not in metadata
    assert set(metadata["transcript_artifacts"]) == {"transcript.json", "transcript.md"}

    first.transcript_path.write_text("# Meeting Transcript\n\nTampered.\n", encoding="utf-8")
    monkeypatch.setattr(
        "meeting_notes.pipeline.create_backend", lambda spec, verbose=False: FakeBackend()
    )
    third = run_process(options)

    assert not third.resumed
    assert third.run_directory != first.run_directory
    assert calls == {"transcribe": 2, "extract": 2}

    monkeypatch.setattr("meeting_notes.pipeline.create_backend", fail_if_backend_created)
    fourth = run_process(options)

    assert fourth.resumed
    assert fourth.run_directory == third.run_directory
    assert calls == {"transcribe": 2, "extract": 2}


def test_recording_change_during_extraction_fails_before_whisper(
    tmp_path: Path, monkeypatch: Any
) -> None:
    recording = tmp_path / "active.mkv"
    recording.write_bytes(b"still recording")
    info = RecordingInfo(
        path=recording,
        format_name="matroska",
        format_long_name="Matroska",
        duration=90.0,
        audio_streams=(AudioStream(0, 1, "aac", 2, "stereo", 48_000),),
    )

    def changing_extract(
        recording_path: Path,
        *,
        audio_ordinal: int,
        output_path: Path,
        verbose: bool,
    ) -> None:
        del audio_ordinal, verbose
        write_test_wav(output_path)
        with recording_path.open("ab") as handle:
            handle.write(b"more data")

    monkeypatch.setattr("meeting_notes.pipeline.ensure_system_dependencies", lambda: None)
    monkeypatch.setattr("meeting_notes.pipeline.probe_recording", lambda path, verbose=False: info)
    monkeypatch.setattr("meeting_notes.pipeline.select_backend", lambda model=None: backend_spec())
    monkeypatch.setattr("meeting_notes.pipeline.extract_audio", changing_extract)
    monkeypatch.setattr(
        "meeting_notes.pipeline.create_backend",
        lambda spec, verbose=False: (_ for _ in ()).throw(AssertionError("Whisper started")),
    )

    with pytest.raises(MediaError, match="changed during processing"):
        run_process(ProcessOptions(recording=recording, output_root=tmp_path / "output"))

    metadata = json.loads(
        (tmp_path / "output" / "active" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"


def test_keyboard_interrupt_marks_run_interrupted(tmp_path: Path, monkeypatch: Any) -> None:
    recording = tmp_path / "meeting.mkv"
    recording.write_bytes(b"local fake recording")
    info = RecordingInfo(
        path=recording,
        format_name="matroska",
        format_long_name="Matroska",
        duration=90.0,
        audio_streams=(AudioStream(0, 1, "aac", 2, "stereo", 48_000),),
    )

    class InterruptedBackend:
        def transcribe(self, audio_path: Path, language: str | None) -> TranscriptionResult:
            del audio_path, language
            raise KeyboardInterrupt

    def fake_extract(
        recording_path: Path,
        *,
        audio_ordinal: int,
        output_path: Path,
        verbose: bool,
    ) -> None:
        del recording_path, audio_ordinal, verbose
        write_test_wav(output_path)

    monkeypatch.setattr("meeting_notes.pipeline.ensure_system_dependencies", lambda: None)
    monkeypatch.setattr("meeting_notes.pipeline.probe_recording", lambda path, verbose=False: info)
    monkeypatch.setattr("meeting_notes.pipeline.select_backend", lambda model=None: backend_spec())
    monkeypatch.setattr(
        "meeting_notes.pipeline.create_backend", lambda spec, verbose=False: InterruptedBackend()
    )
    monkeypatch.setattr("meeting_notes.pipeline.extract_audio", fake_extract)

    with pytest.raises(KeyboardInterrupt):
        run_process(ProcessOptions(recording=recording, output_root=tmp_path / "output"))

    metadata = json.loads(
        (tmp_path / "output" / "meeting" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "interrupted"


def test_output_root_file_reports_artifact_error(tmp_path: Path, monkeypatch: Any) -> None:
    recording = tmp_path / "meeting.mkv"
    recording.write_bytes(b"local fake recording")
    output_root = tmp_path / "not-a-directory"
    output_root.write_text("file", encoding="utf-8")
    info = RecordingInfo(
        path=recording,
        format_name="matroska",
        format_long_name="Matroska",
        duration=90.0,
        audio_streams=(AudioStream(0, 1, "aac", 2, "stereo", 48_000),),
    )

    monkeypatch.setattr("meeting_notes.pipeline.ensure_system_dependencies", lambda: None)
    monkeypatch.setattr("meeting_notes.pipeline.probe_recording", lambda path, verbose=False: info)
    monkeypatch.setattr("meeting_notes.pipeline.select_backend", lambda model=None: backend_spec())

    with pytest.raises(ArtifactError, match="Could not prepare output directory"):
        run_process(ProcessOptions(recording=recording, output_root=output_root))
