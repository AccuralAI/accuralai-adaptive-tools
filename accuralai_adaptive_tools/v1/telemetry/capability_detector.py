"""Detects missing capabilities based on failed tool lookups."""

from collections import Counter
from typing import Any, Dict, List

from ...contracts.models import EventType, TelemetryEvent


class CapabilityDetector:
    """Detects requests for missing functionality."""

    def __init__(self, min_requests: int = 5):
        """Initialize capability detector.

        Args:
            min_requests: Minimum times a missing tool must be requested
        """
        self.min_requests = min_requests

    async def analyze(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Analyze events for missing tool requests.

        Args:
            events: List of telemetry events to analyze

        Returns:
            List of detected patterns for missing capabilities
        """
        # Find TOOL_NOT_FOUND events
        not_found_events = [
            e for e in events
            if e.event_type == EventType.TOOL_NOT_FOUND and e.item_id
        ]

        # Count requests by tool name
        tool_requests = Counter(e.item_id for e in not_found_events)

        # Create patterns for frequently requested missing tools
        patterns = []
        for tool_name, count in tool_requests.items():
            if count >= self.min_requests:
                pattern = self._create_missing_tool_pattern(tool_name, count, not_found_events)
                patterns.append(pattern)

        # Sort by request count (most requested first)
        patterns.sort(key=lambda p: p["request_count"], reverse=True)

        return patterns

    def _create_missing_tool_pattern(
        self,
        tool_name: str,
        count: int,
        events: List[TelemetryEvent]
    ) -> Dict[str, Any]:
        """Create pattern metadata for missing tool.

        Args:
            tool_name: Name of requested tool
            count: Number of times requested
            events: Original events for context

        Returns:
            Pattern dictionary with metadata
        """
        # Extract event IDs for evidence
        evidence_events = [
            e.event_id for e in events
            if e.item_id == tool_name
        ][:10]  # First 10

        # Try to infer functionality from name
        inferred_description = self._infer_functionality(tool_name)

        return {
            "pattern_type": "missing_capability",
            "tool_name": tool_name,
            "request_count": count,
            "inferred_description": inferred_description,
            "evidence_events": evidence_events,
            "suggested_name": tool_name,  # Use requested name
            "description": f"Implement missing tool: {tool_name} - {inferred_description}",
        }

    def _infer_functionality(self, tool_name: str) -> str:
        """Infer tool functionality from its name.

        Args:
            tool_name: Name of tool

        Returns:
            Human-readable description
        """
        name_lower = tool_name.lower()

        # Common patterns
        if "read" in name_lower or "load" in name_lower:
            return "Reads or loads data from files/sources"
        elif "write" in name_lower or "save" in name_lower:
            return "Writes or saves data to files/destinations"
        elif "list" in name_lower or "dir" in name_lower:
            return "Lists files or directory contents"
        elif "search" in name_lower or "find" in name_lower or "grep" in name_lower:
            return "Searches for patterns or content"
        elif "parse" in name_lower or "extract" in name_lower:
            return "Parses or extracts structured data"
        elif "convert" in name_lower or "transform" in name_lower:
            return "Converts or transforms data between formats"
        elif "validate" in name_lower or "check" in name_lower:
            return "Validates or checks data integrity"
        elif "merge" in name_lower or "combine" in name_lower or "join" in name_lower:
            return "Merges or combines multiple data sources"
        elif "filter" in name_lower or "select" in name_lower:
            return "Filters or selects data based on criteria"
        elif "sort" in name_lower or "order" in name_lower:
            return "Sorts or orders data"
        elif "count" in name_lower or "stat" in name_lower:
            return "Counts or computes statistics"
        elif "format" in name_lower or "pretty" in name_lower:
            return "Formats data for display"
        elif "delete" in name_lower or "remove" in name_lower:
            return "Deletes or removes files/data"
        elif "copy" in name_lower or "duplicate" in name_lower:
            return "Copies or duplicates files/data"
        elif "move" in name_lower or "rename" in name_lower:
            return "Moves or renames files/data"
        elif "compress" in name_lower or "zip" in name_lower:
            return "Compresses or archives data"
        elif "decompress" in name_lower or "unzip" in name_lower:
            return "Decompresses or extracts archives"
        elif "download" in name_lower or "fetch" in name_lower:
            return "Downloads or fetches data from remote sources"
        elif "upload" in name_lower or "push" in name_lower:
            return "Uploads or pushes data to remote destinations"
        elif "diff" in name_lower or "compare" in name_lower:
            return "Compares or diffs data"
        else:
            return f"Tool for {tool_name.replace('_', ' ').replace('.', ' ')}"

    async def detect_missing_capability(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Public method for missing capability detection (implements protocol).

        Args:
            events: List of telemetry events

        Returns:
            List of detected missing capability patterns
        """
        return await self.analyze(events)
