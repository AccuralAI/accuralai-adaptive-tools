"""Main pattern detector that coordinates all analysis types."""

from typing import Any, Dict, List

from ...contracts.models import TelemetryEvent
from ...contracts.protocols import PatternDetector
from .capability_detector import CapabilityDetector
from .failure_analyzer import FailureAnalyzer
from .sequence_analyzer import SequenceAnalyzer


class PatternDetectorImpl(PatternDetector):
    """Main pattern detector implementation coordinating all analyzers."""

    def __init__(
        self,
        sequence_analyzer: SequenceAnalyzer | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        capability_detector: CapabilityDetector | None = None,
    ):
        """Initialize pattern detector with analyzers.

        Args:
            sequence_analyzer: Analyzer for repeated sequences
            failure_analyzer: Analyzer for high failure rates
            capability_detector: Detector for missing capabilities
        """
        self.sequence_analyzer = sequence_analyzer or SequenceAnalyzer()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer()
        self.capability_detector = capability_detector or CapabilityDetector()

    async def analyze(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Analyze events for all pattern types.

        Args:
            events: List of telemetry events to analyze

        Returns:
            Combined list of all detected patterns
        """
        # Run all analyzers in parallel
        sequence_patterns = await self.sequence_analyzer.analyze(events)
        failure_patterns = await self.failure_analyzer.analyze(events)
        capability_patterns = await self.capability_detector.analyze(events)

        # Combine all patterns
        all_patterns = sequence_patterns + failure_patterns + capability_patterns

        # Sort by priority (frequency/severity)
        all_patterns.sort(key=self._calculate_priority, reverse=True)

        return all_patterns

    async def detect_repeated_sequence(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect repeated tool sequences.

        Args:
            events: List of telemetry events

        Returns:
            List of sequence patterns
        """
        return await self.sequence_analyzer.detect_repeated_sequence(events)

    async def detect_high_failure_rate(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect tools with high failure rates.

        Args:
            events: List of telemetry events

        Returns:
            List of failure patterns
        """
        return await self.failure_analyzer.detect_high_failure_rate(events)

    async def detect_missing_capability(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect requests for missing functionality.

        Args:
            events: List of telemetry events

        Returns:
            List of missing capability patterns
        """
        return await self.capability_detector.detect_missing_capability(events)

    def _calculate_priority(self, pattern: Dict[str, Any]) -> float:
        """Calculate priority score for a pattern.

        Args:
            pattern: Pattern dictionary

        Returns:
            Priority score (higher = more important)
        """
        pattern_type = pattern["pattern_type"]

        if pattern_type == "repeated_sequence":
            # Priority based on frequency and potential improvement
            frequency = pattern.get("frequency", 0)
            improvement = pattern.get("estimated_improvement_pct", 0) / 100
            return frequency * (1 + improvement)

        elif pattern_type == "high_failure_rate":
            # Priority based on failure rate and execution count
            failure_rate = pattern.get("failure_rate", 0)
            executions = pattern.get("total_executions", 0)
            return failure_rate * executions

        elif pattern_type == "missing_capability":
            # Priority based on request count
            return pattern.get("request_count", 0)

        return 0.0
