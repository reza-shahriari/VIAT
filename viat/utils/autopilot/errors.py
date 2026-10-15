"""Shared exceptions for the Qt-free Autopilot pipeline."""


class Cancelled(Exception):
    """Raised when a worker or headless run is cancelled by the user."""
