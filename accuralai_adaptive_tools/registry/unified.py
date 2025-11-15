"""Unified registry for V1 tools and V2 plans."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from ..contracts.models import ApprovalState, Plan, SystemType


class UnifiedRegistry:
    """Registry for V1 tools and V2 plans."""

    def __init__(self, db_path: str = "~/.accuralai/adaptive-tools/registry.db"):
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
                CREATE TABLE IF NOT EXISTS unified_registry (
                    id TEXT PRIMARY KEY,
                    type TEXT CHECK(type IN ('tool', 'plan')),
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    system TEXT CHECK(system IN ('v1', 'v2', 'builtin')),
                    source_code TEXT,
                    function_schema JSON,
                    plan_yaml TEXT,
                    plan_schema JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    approval_state TEXT,
                    uses_items JSON,
                    used_by_items JSON,
                    metadata JSON
                )
                """
            )

            # Create indices
            await db.execute("CREATE INDEX IF NOT EXISTS idx_type ON unified_registry(type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_system ON unified_registry(system)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_name_version ON unified_registry(name, version)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_approval ON unified_registry(approval_state)")

            await db.commit()

        self._initialized = True

    async def _ensure_table_exists(self, db: aiosqlite.Connection):
        """Ensure table exists in the given connection (needed for :memory: databases)."""
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS unified_registry (
                id TEXT PRIMARY KEY,
                type TEXT CHECK(type IN ('tool', 'plan')),
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                system TEXT CHECK(system IN ('v1', 'v2', 'builtin')),
                source_code TEXT,
                function_schema JSON,
                plan_yaml TEXT,
                plan_schema JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                approval_state TEXT,
                uses_items JSON,
                used_by_items JSON,
                metadata JSON
            )
            """
        )
        # Create indices
        await db.execute("CREATE INDEX IF NOT EXISTS idx_type ON unified_registry(type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_system ON unified_registry(system)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_name_version ON unified_registry(name, version)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_approval ON unified_registry(approval_state)")
        await db.commit()

    async def register_tool(
        self,
        name: str,
        source_code: str,
        function_schema: Dict[str, Any],
        system: SystemType,
        version: str = "1",
        created_by: str = "system",
        approval_state: ApprovalState = ApprovalState.PENDING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register V1-generated or builtin tool."""
        await self.initialize()

        item_id = f"tool_{name}_v{version}"

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            
            # Check if tool already exists
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, version FROM unified_registry WHERE id = ? OR (name = ? AND system = ?)",
                (item_id, name, system.value)
            ) as cursor:
                existing = await cursor.fetchone()
                
            if existing:
                # Tool already exists - return existing ID or update if needed
                existing_id = existing["id"]
                existing_version = existing["version"]
                print(f"DEBUG: Tool '{name}' already exists with ID '{existing_id}', version '{existing_version}'")
                
                # Optionally update if this is a newer version or different source code
                # For now, just return the existing ID
                return existing_id
            
            # Tool doesn't exist, insert it
            await db.execute(
                """
                INSERT INTO unified_registry (
                    id, type, name, version, system,
                    source_code, function_schema,
                    created_at, created_by, approval_state,
                    uses_items, used_by_items, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    "tool",
                    name,
                    version,
                    system.value,
                    source_code,
                    json.dumps(function_schema),
                    datetime.utcnow(),
                    created_by,
                    approval_state.value,
                    json.dumps([]),
                    json.dumps([]),
                    json.dumps(metadata or {}),
                ),
            )
            await db.commit()

        return item_id

    async def register_plan(
        self,
        plan: Plan,
        system: SystemType,
        created_by: str = "system",
        approval_state: ApprovalState = ApprovalState.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register V2-optimized plan."""
        await self.initialize()

        item_id = f"plan_{plan.name}_{plan.version}"

        # Extract tool dependencies from plan
        uses_items = [step.tool for step in plan.steps]

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            await db.execute(
                """
                INSERT INTO unified_registry (
                    id, type, name, version, system,
                    plan_yaml, plan_schema,
                    created_at, created_by, approval_state,
                    uses_items, used_by_items, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    "plan",
                    plan.name,
                    plan.version,
                    system.value,
                    "",  # Will be filled by PlanLang serializer
                    json.dumps(plan.dict()),
                    datetime.utcnow(),
                    created_by,
                    approval_state.value,
                    json.dumps(uses_items),
                    json.dumps([]),
                    json.dumps(metadata or {}),
                ),
            )

            # Update reverse references (tools used by this plan)
            for tool_name in uses_items:
                await self._add_reverse_reference(db, tool_name, item_id)

            await db.commit()

        return item_id

    async def get_tool(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get tool by name (latest version if not specified)."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if version:
                query = "SELECT * FROM unified_registry WHERE type = 'tool' AND name = ? AND version = ?"
                params = (name, version)
            else:
                query = """
                    SELECT * FROM unified_registry
                    WHERE type = 'tool' AND name = ?
                    ORDER BY created_at DESC LIMIT 1
                """
                params = (name,)

            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_plan(self, name: str, version: Optional[str] = None) -> Optional[Plan]:
        """Get plan by name (latest version if not specified)."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if version:
                query = "SELECT * FROM unified_registry WHERE type = 'plan' AND name = ? AND version = ?"
                params = (name, version)
            else:
                query = """
                    SELECT * FROM unified_registry
                    WHERE type = 'plan' AND name = ?
                    ORDER BY created_at DESC LIMIT 1
                """
                params = (name,)

            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row:
                    plan_schema = json.loads(row["plan_schema"])
                    return Plan(**plan_schema)
                return None

    async def list_tools(self, system: Optional[SystemType] = None) -> List[str]:
        """List all tools."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            if system:
                query = "SELECT DISTINCT name FROM unified_registry WHERE type = 'tool' AND system = ?"
                params = (system.value,)
            else:
                query = "SELECT DISTINCT name FROM unified_registry WHERE type = 'tool'"
                params = ()

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def list_plans(self, system: Optional[SystemType] = None) -> List[str]:
        """List all plans."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            if system:
                query = "SELECT DISTINCT name FROM unified_registry WHERE type = 'plan' AND system = ?"
                params = (system.value,)
            else:
                query = "SELECT DISTINCT name FROM unified_registry WHERE type = 'plan'"
                params = ()

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def get_dependencies(self, item_id: str) -> List[str]:
        """Get items this depends on."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT uses_items FROM unified_registry WHERE id = ?", (item_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return []

    async def get_dependents(self, tool_name: str) -> List[str]:
        """Get items that depend on this tool."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM unified_registry WHERE type = 'tool' AND name = ? ORDER BY created_at DESC LIMIT 1",
                (tool_name,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row["used_by_items"])
                return []

    async def update_approval_state(self, item_id: str, state: ApprovalState):
        """Update approval state of item."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE unified_registry SET approval_state = ? WHERE id = ?",
                (state.value, item_id),
            )
            await db.commit()

    async def _add_reverse_reference(self, db: aiosqlite.Connection, tool_name: str, plan_id: str):
        """Add reverse reference from tool to plan."""
        # Get latest version of tool
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM unified_registry WHERE type = 'tool' AND name = ? ORDER BY created_at DESC LIMIT 1",
            (tool_name,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return

            # Update used_by_items
            used_by = json.loads(row["used_by_items"])
            if plan_id not in used_by:
                used_by.append(plan_id)
                await db.execute(
                    "UPDATE unified_registry SET used_by_items = ? WHERE id = ?",
                    (json.dumps(used_by), row["id"]),
                )

    async def get_v1_tools_for_v2(self) -> List[Dict[str, Any]]:
        """Get all V1 tools available for V2 to use."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM unified_registry
                WHERE type = 'tool' AND system = 'v1' AND approval_state = 'active'
                ORDER BY name, created_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
                # Deduplicate by name (keep latest)
                seen = set()
                tools = []
                for row in rows:
                    if row["name"] not in seen:
                        seen.add(row["name"])
                        tools.append(dict(row))
                return tools

    async def sync(self):
        """Synchronize V1 tools available to V2."""
        # This is called by V3 coordinator to ensure V2 sees all V1 tools
        # In this implementation, it's automatic via the database
        # But could trigger cache refresh, etc.
        pass

    async def get_v1_stats(self) -> Dict[str, Any]:
        """Get V1 system statistics."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            
            # Count tools generated by V1
            async with db.execute(
                "SELECT COUNT(*) as count FROM unified_registry WHERE type = 'tool' AND system = 'v1'"
            ) as cursor:
                row = await cursor.fetchone()
                tools_generated = row["count"] if row else 0
                
            # Debug: List all tools to verify
            async with db.execute(
                "SELECT name, system, type FROM unified_registry WHERE type = 'tool'"
            ) as cursor:
                all_tools = await cursor.fetchall()
                if all_tools:
                    print(f"DEBUG: Found {len(all_tools)} tools in registry: {[dict(t) for t in all_tools]}")
                else:
                    print("DEBUG: No tools found in registry")
            
            # Count pending proposals
            async with db.execute(
                "SELECT COUNT(*) as count FROM unified_registry WHERE type = 'tool' AND system = 'v1' AND approval_state = 'pending'"
            ) as cursor:
                row = await cursor.fetchone()
                pending_proposals = row["count"] if row else 0
            
            # Calculate average success rate (placeholder - would need telemetry)
            avg_success_rate = 0.95  # Default placeholder
            
            return {
                "tools_generated": tools_generated,
                "pending_proposals": pending_proposals,
                "avg_success_rate": avg_success_rate,
            }

    async def get_v2_stats(self) -> Dict[str, Any]:
        """Get V2 system statistics."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            
            # Count active plans
            async with db.execute(
                "SELECT COUNT(*) as count FROM unified_registry WHERE type = 'plan' AND system = 'v2' AND approval_state = 'active'"
            ) as cursor:
                row = await cursor.fetchone()
                active_plans = row["count"] if row else 0
            
            # Count optimization runs (placeholder - would need telemetry)
            optimization_runs = 0
            
            # Average improvement (placeholder)
            avg_improvement = 0.0
            
            return {
                "active_plans": active_plans,
                "optimization_runs": optimization_runs,
                "avg_improvement": avg_improvement,
            }

    async def get_cross_system_stats(self) -> Dict[str, Any]:
        """Get cross-system statistics."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            
            # Count V1 tools used in V2 plans
            async with db.execute(
                """
                SELECT COUNT(DISTINCT r1.name) as count
                FROM unified_registry r1
                JOIN unified_registry r2 ON r1.name IN (
                    SELECT json_each.value
                    FROM json_each(json(r2.uses_items))
                )
                WHERE r1.type = 'tool' AND r1.system = 'v1'
                AND r2.type = 'plan' AND r2.system = 'v2'
                """
            ) as cursor:
                row = await cursor.fetchone()
                v1_in_v2 = row["count"] if row else 0
            
            # V2 patterns triggering V1 (placeholder)
            v2_in_v1 = 0
            
            # Compound factor (placeholder)
            compound_factor = 1.0
            
            return {
                "v1_in_v2": v1_in_v2,
                "v2_in_v1": v2_in_v1,
                "compound_factor": compound_factor,
            }

    async def record_improvement(
        self,
        item_id: str,
        improvement_factor: float,
        time_saved_ms: float,
        cost_saved_cents: float,
    ) -> None:
        """Record an improvement metric for an item."""
        await self.initialize()
        
        # Store improvement in metadata
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            
            # Get existing metadata
            async with db.execute(
                "SELECT metadata FROM unified_registry WHERE id = ?", (item_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                else:
                    metadata = {}
            
            # Add improvement record
            if "improvements" not in metadata:
                metadata["improvements"] = []
            
            metadata["improvements"].append({
                "improvement_factor": improvement_factor,
                "time_saved_ms": time_saved_ms,
                "cost_saved_cents": cost_saved_cents,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            # Update metadata
            await db.execute(
                "UPDATE unified_registry SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), item_id),
            )
            await db.commit()

    async def get_improvements_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get all improvements recorded since a given time."""
        await self.initialize()
        
        improvements = []
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_table_exists(db)
            db.row_factory = aiosqlite.Row
            
            async with db.execute("SELECT id, metadata FROM unified_registry") as cursor:
                async for row in cursor:
                    if row["metadata"]:
                        metadata = json.loads(row["metadata"])
                        if "improvements" in metadata:
                            for improvement in metadata["improvements"]:
                                # Parse timestamp and filter
                                timestamp_str = improvement.get("timestamp", "")
                                if timestamp_str:
                                    try:
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                        if timestamp >= since:
                                            improvement["item_id"] = row["id"]
                                            improvements.append(improvement)
                                    except (ValueError, TypeError):
                                        pass
        
        return improvements
