"""Local meeting recording transcription."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meeting-notes")
except PackageNotFoundError:  # Source tree imported without installation.
    __version__ = "0+unknown"
