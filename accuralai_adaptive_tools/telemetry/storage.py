"""Telemetry storage using SQLite."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from ..contracts.models import EventType, TelemetryEvent


class TelemetryStorage:
    """Stores telemetry events in SQLite database."""

    def __init__(self, db_path: str = "~/.accuralai/adaptive-tools/telemetry.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def initialize(self):
        """Initialize database schema."""
        if self._initialized and str(self.db_path) != ":memory:":
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    item_id TEXT,
                    item_type TEXT,
                    latency_ms REAL,
                    cost_cents REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    routed_to_v1 BOOLEAN DEFAULT FALSE,
                    routed_to_v2 BOOLEAN DEFAULT FALSE,
                    event_data JSON
                )
                """
            )

            # Create indices
            await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry_events(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_item ON telemetry_events(item_id, item_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_routing ON telemetry_events(routed_to_v1, routed_to_v2)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON telemetry_events(event_type)")

            await db.commit()

        self._initialized = True

    async def _ensure_table_exists(self, db: aiosqlite.Connection):
        """Ensure table exists in the given connection (needed for :memory: databases)."""
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                event_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                item_id TEXT,
                item_type TEXT,
                latency_ms REAL,
                cost_cents REAL,
                success BOOLEAN,
                error_message TEXT,
                routed_to_v1 BOOLEAN DEFAULT FALSE,
                routed_to_v2 BOOLEAN DEFAULT FALSE,
                event_data JSON
            )
            """
        )
        # Create indices
        await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry_events(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_item ON telemetry_events(item_id, item_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_routing ON telemetry_events(routed_to_v1, routed_to_v2)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON telemetry_events(event_type)")
        await db.commit()

    async def insert(self, event: TelemetryEvent):
        """Insert telemetry event."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            await db.execute(
                """
                INSERT INTO telemetry_events (
                    event_id, timestamp, event_type, item_id, item_type,
                    latency_ms, cost_cents, success, error_message,
                    routed_to_v1, routed_to_v2, event_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.event_type.value,
                    event.item_id,
                    event.item_type,
                    event.latency_ms,
                    event.cost_cents,
                    event.success,
                    event.error_message,
                    event.routed_to_v1,
                    event.routed_to_v2,
                    json.dumps(event.event_data),
                ),
            )
            await db.commit()

    async def query(self, filter: Dict[str, Any]) -> List[TelemetryEvent]:
        """Query events with filter."""
        await self.initialize()

        # Build query
        conditions = []
        params = []

        if "event_type" in filter:
            conditions.append("event_type = ?")
            params.append(filter["event_type"])

        if "item_id" in filter:
            conditions.append("item_id = ?")
            params.append(filter["item_id"])

        if "item_type" in filter:
            conditions.append("item_type = ?")
            params.append(filter["item_type"])

        if "success" in filter:
            conditions.append("success = ?")
            params.append(filter["success"])

        if "since" in filter:
            conditions.append("timestamp >= ?")
            params.append(filter["since"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM telemetry_events WHERE {where_clause} ORDER BY timestamp DESC"

        if "limit" in filter:
            query += f" LIMIT {filter['limit']}"

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_event(dict(row)) for row in rows]

    async def get_recent(self, hours: int = 24) -> List[TelemetryEvent]:
        """Get events from last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return await self.query({"since": since})

    async def get_by_item(self, item_id: str, item_type: str) -> List[TelemetryEvent]:
        """Get all events for specific item."""
        return await self.query({"item_id": item_id, "item_type": item_type})

    async def get_tool_sequences(self, min_length: int = 3, hours: int = 24) -> List[List[str]]:
        """Extract sequences of tool executions."""
        events = await self.query(
            {"event_type": EventType.TOOL_EXECUTED.value, "since": datetime.utcnow() - timedelta(hours=hours)}
        )

        # Group by time windows (5 minute windows)
        window_size = timedelta(minutes=5)
        sequences = []
        current_sequence = []
        last_timestamp = None

        for event in sorted(events, key=lambda e: e.timestamp):
            if last_timestamp and (event.timestamp - last_timestamp) > window_size:
                if len(current_sequence) >= min_length:
                    sequences.append(current_sequence)
                current_sequence = []

            if event.item_id:
                current_sequence.append(event.item_id)
            last_timestamp = event.timestamp

        if len(current_sequence) >= min_length:
            sequences.append(current_sequence)

        return sequences

    async def get_failure_stats(self, item_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for item."""
        events = await self.query(
            {"item_id": item_id, "since": datetime.utcnow() - timedelta(hours=hours)}
        )

        total = len(events)
        failures = [e for e in events if not e.success]
        failure_rate = len(failures) / total if total > 0 else 0.0

        # Categorize errors
        error_categories = {}
        for event in failures:
            error = event.error_message or "Unknown"
            error_categories[error] = error_categories.get(error, 0) + 1

        return {
            "total_executions": total,
            "failures": len(failures),
            "failure_rate": failure_rate,
            "error_categories": error_categories,
        }

    async def get_latency_stats(self, item_id: str, hours: int = 24) -> Dict[str, float]:
        """Get latency statistics for item."""
        events = await self.query(
            {"item_id": item_id, "success": True, "since": datetime.utcnow() - timedelta(hours=hours)}
        )

        latencies = [e.latency_ms for e in events if e.latency_ms is not None]

        if not latencies:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

        latencies_sorted = sorted(latencies)
        p95_index = int(len(latencies_sorted) * 0.95)

        return {
            "count": len(latencies),
            "avg": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p95": latencies_sorted[p95_index] if latencies_sorted else 0.0,
        }

    async def cleanup_old(self, days: int = 30):
        """Delete events older than N days."""
        await self.initialize()

        cutoff = datetime.utcnow() - timedelta(days=days)
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            await db.execute("DELETE FROM telemetry_events WHERE timestamp < ?", (cutoff,))
            await db.commit()

    def _row_to_event(self, row: Dict[str, Any]) -> TelemetryEvent:
        """Convert database row to TelemetryEvent."""
        return TelemetryEvent(
            event_id=row["event_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if isinstance(row["timestamp"], str) else row["timestamp"],
            event_type=EventType(row["event_type"]),
            item_id=row.get("item_id"),
            item_type=row.get("item_type"),
            latency_ms=row.get("latency_ms"),
            cost_cents=row.get("cost_cents"),
            success=bool(row.get("success", True)),
            error_message=row.get("error_message"),
            routed_to_v1=bool(row.get("routed_to_v1", False)),
            routed_to_v2=bool(row.get("routed_to_v2", False)),
            event_data=json.loads(row["event_data"]) if row.get("event_data") else {},
        )
