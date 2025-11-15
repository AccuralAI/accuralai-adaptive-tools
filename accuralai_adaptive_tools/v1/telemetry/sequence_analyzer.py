"""Analyzes tool execution sequences to detect repeated patterns."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from ...contracts.models import EventType, TelemetryEvent


class SequenceAnalyzer:
    """Detects repeated tool sequences that could be combined into new tools."""

    def __init__(self, min_occurrences: int = 10, min_sequence_length: int = 3, max_sequence_length: int = 7):
        """Initialize sequence analyzer.

        Args:
            min_occurrences: Minimum times a sequence must occur to be considered
            min_sequence_length: Minimum length of sequences to detect
            max_sequence_length: Maximum length of sequences to detect
        """
        self.min_occurrences = min_occurrences
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    async def analyze(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Analyze events for repeated tool sequences.

        Args:
            events: List of telemetry events to analyze

        Returns:
            List of detected patterns with metadata
        """
        # Extract tool execution sequences
        sequences = self._extract_sequences(events)

        # Find frequent n-grams
        patterns = []
        for n in range(self.min_sequence_length, self.max_sequence_length + 1):
            ngrams = self._find_ngrams(sequences, n)
            for ngram, count in ngrams.items():
                if count >= self.min_occurrences:
                    pattern = self._create_pattern_from_sequence(ngram, count, events)
                    patterns.append(pattern)

        # Sort by frequency and length (longer, more frequent sequences first)
        patterns.sort(key=lambda p: (p["frequency"], len(p["sequence"])), reverse=True)

        return patterns

    def _extract_sequences(self, events: List[TelemetryEvent]) -> List[List[str]]:
        """Extract tool execution sequences from events.

        Groups events by time windows (5-minute gaps separate sequences).
        """
        from datetime import timedelta

        # Filter to successful tool executions only
        tool_events = [
            e for e in events
            if e.event_type == EventType.TOOL_EXECUTED and e.success and e.item_id
        ]

        # Sort by timestamp
        tool_events.sort(key=lambda e: e.timestamp)

        # Group into sequences (gap > 5 minutes = new sequence)
        sequences = []
        current_sequence = []
        last_timestamp = None
        window_size = timedelta(minutes=5)

        for event in tool_events:
            if last_timestamp and (event.timestamp - last_timestamp) > window_size:
                if len(current_sequence) >= self.min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = []

            current_sequence.append(event.item_id)
            last_timestamp = event.timestamp

        # Add final sequence
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append(current_sequence)

        return sequences

    def _find_ngrams(self, sequences: List[List[str]], n: int) -> Counter:
        """Find all n-grams of length n in sequences.

        Args:
            sequences: List of tool sequences
            n: Length of n-gram

        Returns:
            Counter of n-grams with their frequencies
        """
        ngrams = Counter()

        for sequence in sequences:
            if len(sequence) < n:
                continue

            for i in range(len(sequence) - n + 1):
                ngram = tuple(sequence[i:i + n])
                ngrams[ngram] += 1

        return ngrams

    def _create_pattern_from_sequence(
        self,
        sequence: Tuple[str, ...],
        frequency: int,
        events: List[TelemetryEvent]
    ) -> Dict[str, Any]:
        """Create pattern metadata from detected sequence.

        Args:
            sequence: Tuple of tool names
            frequency: Number of times this sequence occurred
            events: Original events for extracting metrics

        Returns:
            Pattern dictionary with metadata
        """
        # Calculate aggregate metrics
        tool_metrics = self._calculate_sequence_metrics(sequence, events)

        # Generate suggested name
        suggested_name = self._generate_tool_name(sequence)

        # Estimate benefit
        baseline_latency = sum(tool_metrics[tool]["avg_latency_ms"] for tool in sequence)
        estimated_improvement = 0.2  # Conservative 20% improvement from composition

        return {
            "pattern_type": "repeated_sequence",
            "sequence": list(sequence),
            "frequency": frequency,
            "suggested_name": suggested_name,
            "description": f"Combines {', '.join(sequence)} into single operation",
            "tool_metrics": tool_metrics,
            "baseline_latency_ms": baseline_latency,
            "estimated_improvement_pct": estimated_improvement * 100,
            "evidence_count": frequency,
        }

    def _calculate_sequence_metrics(
        self,
        sequence: Tuple[str, ...],
        events: List[TelemetryEvent]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for each tool in sequence.

        Args:
            sequence: Tuple of tool names
            events: Original events

        Returns:
            Dict mapping tool name to metrics
        """
        metrics = defaultdict(lambda: {"latencies": [], "costs": [], "count": 0})

        for event in events:
            if event.item_id in sequence and event.success:
                if event.latency_ms is not None:
                    metrics[event.item_id]["latencies"].append(event.latency_ms)
                if event.cost_cents is not None:
                    metrics[event.item_id]["costs"].append(event.cost_cents)
                metrics[event.item_id]["count"] += 1

        # Calculate averages
        result = {}
        for tool, data in metrics.items():
            result[tool] = {
                "count": data["count"],
                "avg_latency_ms": sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0,
                "avg_cost_cents": sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0,
            }

        return result

    def _generate_tool_name(self, sequence: Tuple[str, ...]) -> str:
        """Generate a suggested tool name from sequence.

        Args:
            sequence: Tuple of tool names

        Returns:
            Suggested name (snake_case)
        """
        # Extract key words from tool names
        words = []
        for tool in sequence:
            # Split on dots and underscores
            parts = tool.replace(".", "_").split("_")
            # Take meaningful words (skip generic ones)
            words.extend([p for p in parts if p and len(p) > 2])

        # Deduplicate while preserving order
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        # Limit to 3-4 words
        if len(unique_words) > 4:
            unique_words = unique_words[:2] + [unique_words[-1]]

        return "_".join(unique_words[:4])

    async def detect_repeated_sequence(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Public method for repeated sequence detection (implements protocol).

        Args:
            events: List of telemetry events

        Returns:
            List of detected sequence patterns
        """
        return await self.analyze(events)
