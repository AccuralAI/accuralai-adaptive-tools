"""Shared telemetry system for V1 and V2."""

from .collector import SharedTelemetry
from .router import TelemetryRouter
from .storage import TelemetryStorage

# Alias for compatibility
TelemetryCollector = SharedTelemetry

__all__ = [
    "SharedTelemetry",
    "TelemetryCollector",
    "TelemetryRouter",
    "TelemetryStorage",
]
