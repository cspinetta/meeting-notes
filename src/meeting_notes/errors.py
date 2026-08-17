"""User-facing error types for the meeting-notes pipeline."""


class MeetingNotesError(RuntimeError):
    """Base class for expected, user-actionable failures."""


class DependencyError(MeetingNotesError):
    """A required local dependency is unavailable."""


class MediaError(MeetingNotesError):
    """A recording cannot be inspected or decoded."""


class PlatformNotSupportedError(MeetingNotesError):
    """No local transcription backend is supported on this platform."""


class TranscriptionError(MeetingNotesError):
    """A local Whisper backend failed to transcribe audio."""


class ArtifactError(MeetingNotesError):
    """An output artifact or run directory cannot be used safely."""
