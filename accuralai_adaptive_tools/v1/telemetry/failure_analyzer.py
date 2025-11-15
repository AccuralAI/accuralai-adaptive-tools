"""Analyzes tool failures to detect opportunities for improvement."""

from collections import Counter, defaultdict
from typing import Any, Dict, List

from ...contracts.models import EventType, TelemetryEvent


class FailureAnalyzer:
    """Detects tools with high failure rates that could be improved."""

    def __init__(self, min_executions: int = 20, failure_threshold: float = 0.2):
        """Initialize failure analyzer.

        Args:
            min_executions: Minimum executions before considering for analysis
            failure_threshold: Minimum failure rate (0.2 = 20%) to flag
        """
        self.min_executions = min_executions
        self.failure_threshold = failure_threshold

    async def analyze(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Analyze events for high-failure tools.

        Args:
            events: List of telemetry events to analyze

        Returns:
            List of detected patterns with improvement opportunities
        """
        # Group by tool
        tool_stats = self._calculate_tool_statistics(events)

        # Find high-failure tools
        patterns = []
        for tool_id, stats in tool_stats.items():
            if stats["total"] < self.min_executions:
                continue

            failure_rate = stats["failures"] / stats["total"]
            if failure_rate >= self.failure_threshold:
                pattern = self._create_failure_pattern(tool_id, stats, failure_rate)
                patterns.append(pattern)

        # Sort by failure rate (worst first)
        patterns.sort(key=lambda p: p["failure_rate"], reverse=True)

        return patterns

    def _calculate_tool_statistics(self, events: List[TelemetryEvent]) -> Dict[str, Dict[str, Any]]:
        """Calculate failure statistics for each tool.

        Args:
            events: List of telemetry events

        Returns:
            Dict mapping tool_id to statistics
        """
        stats = defaultdict(lambda: {
            "total": 0,
            "failures": 0,
            "error_categories": Counter(),
            "failed_events": []
        })

        for event in events:
            if event.event_type != EventType.TOOL_EXECUTED or not event.item_id:
                continue

            tool_id = event.item_id
            stats[tool_id]["total"] += 1

            if not event.success:
                stats[tool_id]["failures"] += 1

                # Categorize error
                error_type = self._categorize_error(event.error_message or "Unknown")
                stats[tool_id]["error_categories"][error_type] += 1

                # Store event for evidence
                stats[tool_id]["failed_events"].append(event.event_id)

        return dict(stats)

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into common types.

        Args:
            error_message: Error message text

        Returns:
            Error category string
        """
        error_lower = error_message.lower()

        # Common error patterns
        if "filenotfound" in error_lower or "no such file" in error_lower:
            return "FileNotFoundError"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "PermissionError"
        elif "unicode" in error_lower or "decode" in error_lower or "encoding" in error_lower:
            return "UnicodeDecodeError"
        elif "timeout" in error_lower:
            return "TimeoutError"
        elif "connection" in error_lower or "network" in error_lower:
            return "ConnectionError"
        elif "value" in error_lower:
            return "ValueError"
        elif "type" in error_lower:
            return "TypeError"
        elif "attribute" in error_lower:
            return "AttributeError"
        elif "key" in error_lower:
            return "KeyError"
        elif "index" in error_lower:
            return "IndexError"
        else:
            return "Other"

    def _create_failure_pattern(
        self,
        tool_id: str,
        stats: Dict[str, Any],
        failure_rate: float
    ) -> Dict[str, Any]:
        """Create pattern metadata from failure statistics.

        Args:
            tool_id: Tool identifier
            stats: Tool statistics
            failure_rate: Calculated failure rate

        Returns:
            Pattern dictionary with metadata
        """
        # Find most common error
        most_common_error, error_count = stats["error_categories"].most_common(1)[0]

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(most_common_error, error_count, stats["failures"])

        return {
            "pattern_type": "high_failure_rate",
            "tool_id": tool_id,
            "total_executions": stats["total"],
            "failures": stats["failures"],
            "failure_rate": failure_rate,
            "most_common_error": most_common_error,
            "error_count": error_count,
            "error_distribution": dict(stats["error_categories"]),
            "suggestions": suggestions,
            "evidence_events": stats["failed_events"][:10],  # First 10 for review
            "suggested_name": f"{tool_id}_improved",
            "description": f"Improved version of {tool_id} with {suggestions[0] if suggestions else 'better error handling'}",
        }

    def _generate_improvement_suggestions(
        self,
        error_type: str,
        error_count: int,
        total_failures: int
    ) -> List[str]:
        """Generate specific improvement suggestions based on error type.

        Args:
            error_type: Type of error
            error_count: Count of this error
            total_failures: Total failures

        Returns:
            List of suggestion strings
        """
        suggestions = []
        error_percentage = (error_count / total_failures) * 100 if total_failures > 0 else 0

        if error_type == "FileNotFoundError":
            suggestions.append("Add file existence check before reading")
            suggestions.append("Provide better error message with file path")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        elif error_type == "UnicodeDecodeError":
            suggestions.append("Add encoding parameter (auto-detect or configurable)")
            suggestions.append("Try multiple encodings (utf-8, latin1, etc.)")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        elif error_type == "PermissionError":
            suggestions.append("Check file permissions before operation")
            suggestions.append("Provide clearer error message about required permissions")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        elif error_type == "TimeoutError":
            suggestions.append("Make timeout configurable")
            suggestions.append("Add retry logic with exponential backoff")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        elif error_type == "ConnectionError":
            suggestions.append("Add retry logic for transient failures")
            suggestions.append("Implement circuit breaker pattern")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        elif error_type in ["ValueError", "TypeError"]:
            suggestions.append("Add input validation")
            suggestions.append("Provide better error messages about expected types")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        else:
            suggestions.append("Add comprehensive error handling")
            suggestions.append("Improve error messages for debugging")
            suggestions.append(f"Affects {error_percentage:.1f}% of failures")

        return suggestions

    async def detect_high_failure_rate(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Public method for high failure rate detection (implements protocol).

        Args:
            events: List of telemetry events

        Returns:
            List of detected failure patterns
        """
        return await self.analyze(events)
