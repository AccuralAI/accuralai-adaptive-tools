"""AccuralAI Adaptive Tools - Self-improving tool ecosystem.

V3 unified architecture combining:
- V1: Pattern-based tool generation
- V2: Learning-based workflow optimization
- V3: Coordinated compound gains
"""

__version__ = "0.1.0"

from .contracts import (
    CompoundGains,
    Plan,
    SafetyReport,
    TelemetryEvent,
    ToolMetrics,
    ToolProposal,
    V3Status,
)
from .registry import UnifiedRegistry
from .telemetry import SharedTelemetry, TelemetryRouter, TelemetryStorage

__all__ = [
    # Version
    "__version__",
    # Telemetry
    "SharedTelemetry",
    "TelemetryRouter",
    "TelemetryStorage",
    "TelemetryEvent",
    # Registry
    "UnifiedRegistry",
    # Models
    "ToolProposal",
    "Plan",
    "SafetyReport",
    "ToolMetrics",
    "CompoundGains",
    "V3Status",
]
