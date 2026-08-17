from __future__ import annotations

import argparse
from typing import Any

from meeting_notes.cli import main


def test_cli_turns_filesystem_error_into_friendly_failure(monkeypatch: Any, capsys: Any) -> None:
    def fail(_args: argparse.Namespace) -> int:
        raise OSError("disk unavailable")

    monkeypatch.setattr("meeting_notes.cli._run_process", fail)

    assert main(["process", "meeting.mkv"]) == 2
    assert "Error: local filesystem operation failed: disk unavailable" in capsys.readouterr().err


def test_cli_returns_130_for_keyboard_interrupt(monkeypatch: Any, capsys: Any) -> None:
    def interrupt(_args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("meeting_notes.cli._run_inspect", interrupt)

    assert main(["inspect", "meeting.mkv"]) == 130
    assert "Interrupted." in capsys.readouterr().err
