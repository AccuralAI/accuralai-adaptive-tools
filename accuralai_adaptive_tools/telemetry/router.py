"""Routes telemetry events to V1 and V2 systems."""

from typing import Callable, List

from ..contracts.models import EventType, TelemetryEvent


class TelemetryRouter:
    """Routes telemetry events to appropriate systems."""

    def __init__(self):
        self.v1_handlers: List[Callable] = []
        self.v2_handlers: List[Callable] = []

    def register_v1_handler(self, handler: Callable):
        """Register handler for V1 system."""
        self.v1_handlers.append(handler)

    def register_v2_handler(self, handler: Callable):
        """Register handler for V2 system."""
        self.v2_handlers.append(handler)

    async def route(self, event: TelemetryEvent) -> TelemetryEvent:
        """Route event to appropriate handlers and return updated event."""

        # Determine routing
        route_to_v1 = self._should_route_to_v1(event)
        route_to_v2 = self._should_route_to_v2(event)

        # Update event with routing info
        event.routed_to_v1 = route_to_v1
        event.routed_to_v2 = route_to_v2

        # Call handlers
        if route_to_v1:
            for handler in self.v1_handlers:
                await handler(event)

        if route_to_v2:
            for handler in self.v2_handlers:
                await handler(event)

        return event

    def _should_route_to_v1(self, event: TelemetryEvent) -> bool:
        """Determine if event should go to V1."""
        # V1 cares about: sequences, errors, missing features
        v1_event_types = {
            EventType.TOOL_SEQUENCE,
            EventType.TOOL_ERROR,
            EventType.TOOL_NOT_FOUND,
        }

        # Also route successful tool executions for sequence detection
        if event.event_type == EventType.TOOL_EXECUTED:
            return True

        return event.event_type in v1_event_types

    def _should_route_to_v2(self, event: TelemetryEvent) -> bool:
        """Determine if event should go to V2."""
        # V2 cares about: latency, cost, success rate
        v2_event_types = {
            EventType.TOOL_EXECUTED,
            EventType.PLAN_EXECUTED,
            EventType.PLAN_ERROR,
        }

        return event.event_type in v2_event_types
