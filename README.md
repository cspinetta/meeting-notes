# meeting-notes

`meeting-notes` turns a local OBS `.mkv` recording into a timestamped transcript and a structured
meeting summary. Media inspection, audio extraction, and speech-to-text all run locally. A single
portable Agent Skill lets either Codex or Claude Code read the generated text transcript and write
the summary.

## Privacy model

```text
Audio transcription: local only.

AI summary:
When invoked through Codex or Claude Code, the generated text transcript is
processed by the currently selected AI provider to produce the summary.
The raw recording is never supplied to the AI agent.
```

The deterministic CLI never calls OpenAI, Anthropic, or any other LLM/transcription API. It has no
telemetry, analytics, upload, server, or cloud-storage integration. It invokes local `ffprobe` and
`ffmpeg`, then either local MLX Whisper or the official local OpenAI Whisper Python implementation.
Model weights may be downloaded from their normal model repositories on first use.

The shared skill enforces a narrower boundary: the agent runs local commands against media paths
but reads only `transcript.md` into model context. Thus agent-generated summaries expose transcript
text to the currently selected Codex or Claude provider, but never raw audio or video.

## Architecture

```text
OBS recording
  -> ffprobe inspection and OBS layout inference
  -> ffmpeg mono 16 kHz PCM extraction
  -> local platform Whisper backend
  -> timestamped segment normalization and chronological merge
  -> metadata.json + transcript.json + transcript.md + summary-prompt.md
  -> current Codex/Claude agent reads transcript.md -> summary.md
```

`media.py` owns ffprobe/ffmpeg interaction, `platform.py` selects a backend, `backends/` adapts the
two Whisper packages, `transcript.py` merges/renders segments, and `pipeline.py` owns resumability
and artifact creation. Subprocesses use argument lists and never `shell=True`.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on `PATH`
- Apple Silicon macOS, or Linux

System packages are never installed automatically. On macOS:

```bash
brew install ffmpeg
```

On Debian/Ubuntu Linux:

```bash
sudo apt update
sudo apt install ffmpeg
```

Other common Linux choices include `sudo dnf install ffmpeg` and `sudo pacman -S ffmpeg`.

## Install with uv

From the project root:

```bash
uv sync
```

Platform environment markers install only the relevant speech-to-text implementation:

- Apple Silicon macOS: `mlx-whisper`, default model
`mlx-community/whisper-large-v3-turbo`, using Metal.
- Linux: official `openai-whisper`, default model `turbo`; CUDA is selected when PyTorch reports it
available, otherwise CPU is used.

Intel macOS exits with a clear error because this project currently supports only MLX Whisper on
Apple Silicon macOS. Other operating systems are unsupported. The CLI prints its chosen backend,
device, and model before transcription.

## Inspect an OBS recording

```bash
uv run meeting-notes inspect "/Users/me/Movies/meeting.mkv"
```

Inspection reports the container, duration, every audio stream's ffmpeg selector and stream index,
codec, channel layout/count, sample rate, available tags, and inferred OBS layout. Both `ffmpeg` and
`ffprobe` are checked at startup.

## Transcribe

```bash
uv run meeting-notes process \
  "/Users/me/Movies/meeting.mkv" \
  --speaker-mode auto \
  --me-name Cris
```

Language detection is automatic and Whisper transcribes rather than translates. Supply a hint only
when useful:

```bash
uv run meeting-notes process meeting.mkv --language es
uv run meeting-notes process meeting.mkv --model small
```

The model override is passed to the selected backend, so use a model identifier understood by MLX
Whisper on macOS or official Whisper on Linux.

To perform transcription without any agent summary step:

```bash
uv run meeting-notes process meeting.mkv --no-summary
```

This still writes `summary-prompt.md`, which can later be supplied alongside `transcript.md` to any
LLM. The CLI itself never generates `summary.md`, even when `--no-summary` is omitted; without that
flag it simply reminds you to invoke the Agent Skill.

Useful options:

```text
--output-dir DIR
--model MODEL
--language CODE
--speaker-mode {auto,mixed,obs-3track}
--me-name NAME
--keep-audio
--no-summary
--force
--verbose
```

Run `uv run meeting-notes process --help` for complete help.

## OBS multi-track behavior

The default `--speaker-mode auto` uses the conventional layout below when at least three audio
streams are present; otherwise it transcribes the first audio stream as mixed audio:

```text
Track 1 / 0:a:0 = everyone together
Track 2 / 0:a:1 = Google Meet / remote participant(s)
Track 3 / 0:a:2 = microphone
```

In `obs-3track` mode, tracks 2 and 3 are extracted independently, transcribed with timestamps, and
merged chronologically. Overlapping segments are both retained. Microphone speech is labelled `Me`
(or `--me-name Cris`), while the remote track is labelled `Other participant(s)`. The tool does not
claim to diarize individual people sharing the remote track. `mixed` mode uses only `0:a:0` and adds
no speaker label.

Extracted WAVs are mono, 16 kHz, signed 16-bit PCM inside the run directory. They are deleted only
after successful transcription unless `--keep-audio` is used. Failures retain extraction state for
resumption. Each WAV is first written to a unique hidden partial file and is atomically promoted to
its reusable filename only after ffmpeg exits successfully and validation passes. The original
recording is never modified or deleted.

## Outputs and resumability

The default output root is `./output`. For `meeting-2026-08-17.mkv`, a run resembles:

```text
output/meeting-2026-08-17/
├── metadata.json
├── transcript.json
├── transcript.md
├── summary-prompt.md
└── summary.md          # written only by the Agent Skill
```

Existing valid transcripts are reused when the input fingerprint, backend, device, model, language,
requested/resolved speaker mode, and microphone name match. The fingerprint combines file size,
nanosecond modification time, and SHA-256 over the first and last 1 MiB, avoiding a full reread of a
multi-gigabyte recording on every run. Different configurations receive numbered collision-safe
directories such as `meeting-2026-08-17-2`. `--force` deliberately runs Whisper again in a new
collision-safe directory, preserving earlier successful artifacts.

Use a different artifact root with:

```bash
uv run meeting-notes process meeting.mkv --output-dir /path/to/output
```

`metadata.json` records source filename (not its absolute path), sampled fingerprint, duration,
timestamp, OS/architecture, backend, model, device, language, speaker mode, audio stream metadata,
transcription duration, status, and tool version. It avoids hostname, username, and other unnecessary
machine identifiers.

## Shared Agent Skill

The one canonical implementation is `skills/process-meeting/SKILL.md`. Relative symlinks expose it
to both hosts without copying it:

```text
.agents/skills/process-meeting -> ../../skills/process-meeting
.claude/skills/process-meeting -> ../../skills/process-meeting
```

Invoke it naturally:

```text
# Codex
$process-meeting Process "/Users/me/Movies/meeting.mkv"

# Claude Code
/process-meeting "/Users/me/Movies/meeting.mkv"
```

The skill validates the path, runs `uv sync --frozen`, inspects the media, runs
`meeting-notes process --no-summary`, reads only the resulting text transcript, and creates a
grounded `summary.md`. If transcription fails, it stops without writing a misleading summary.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

Normal tests mock local Whisper and require no model download. No real recording is opened or
modified by the test suite.

## Current limitations

- `auto` intentionally assumes the documented OBS mapping whenever three or more audio streams
exist; use `mixed` or `obs-3track` explicitly for unusual layouts.
- Track-level labelling is not full speaker diarization. Multiple remote people remain
`Other participant(s)`.
- Audio duplicated across OBS tracks may produce overlapping duplicate utterances; preserving both
is safer than silently discarding potentially distinct speech.
- CPU transcription on Linux can be substantially slower than CUDA or Apple Silicon MLX.

