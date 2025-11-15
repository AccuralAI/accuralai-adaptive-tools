"""Shared telemetry collector."""

from typing import Any, Dict, List

from ..contracts.models import TelemetryEvent
from .router import TelemetryRouter
from .storage import TelemetryStorage


class SharedTelemetry:
    """Unified telemetry collection for V1 and V2."""

    def __init__(self, storage: TelemetryStorage, router: TelemetryRouter):
        self.storage = storage
        self.router = router

    async def record(self, event: TelemetryEvent):
        """Record event and route to appropriate systems."""
        # Route to handlers
        event = await self.router.route(event)

        # Store in database
        await self.storage.insert(event)

    async def query(self, filter: Dict[str, Any]) -> List[TelemetryEvent]:
        """Query historical telemetry."""
        return await self.storage.query(filter)

    async def get_recent(self, hours: int = 24) -> List[TelemetryEvent]:
        """Get events from last N hours."""
        return await self.storage.get_recent(hours)

    async def get_by_item(self, item_id: str, item_type: str) -> List[TelemetryEvent]:
        """Get all events for specific item."""
        return await self.storage.get_by_item(item_id, item_type)

    async def get_tool_sequences(self, min_length: int = 3, hours: int = 24) -> List[List[str]]:
        """Extract sequences of tool executions."""
        return await self.storage.get_tool_sequences(min_length, hours)

    async def get_failure_stats(self, item_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for item."""
        return await self.storage.get_failure_stats(item_id, hours)

    async def get_latency_stats(self, item_id: str, hours: int = 24) -> Dict[str, float]:
        """Get latency statistics for item."""
        return await self.storage.get_latency_stats(item_id, hours)
