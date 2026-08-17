# Project guidance

`meeting-notes` is a Python 3.11+ local CLI that inspects OBS recordings with ffprobe, extracts
16 kHz mono PCM with ffmpeg, and transcribes with MLX Whisper on Apple Silicon macOS or official
OpenAI Whisper on Linux. `pipeline.py` orchestrates resumability; `media.py`, `platform.py`, the
backend adapters, and `transcript.py` keep inspection, selection, inference, and rendering separate.

Privacy is an invariant: raw recordings and extracted audio must remain local and must never be
sent to an AI provider, transcription API, telemetry service, or model context. Only generated text
transcripts may be supplied to the current Codex or Claude provider for agent-authored summaries.

Use `uv sync`, `uv run pytest`, `uv run ruff check .`, and `uv run mypy src`. Do not require model
downloads in unit tests; mock transcription backends. Keep `skills/process-meeting/SKILL.md`
portable Agent Skills Markdown. `.agents/skills/process-meeting` and
`.claude/skills/process-meeting` must remain relative symlinks to that one canonical skill.

