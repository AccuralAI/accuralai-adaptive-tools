"""V3 Coordinator - Unified V1+V2 orchestration system."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..contracts.models import EventType, TelemetryEvent, V1Status, V2Status
from ..registry.unified import UnifiedRegistry
from ..telemetry import TelemetryCollector, TelemetryRouter
from ..v1.system import V1System
from ..v2.system import V2System


@dataclass
class V3Status:
    """Status of V3 unified system (simplified for coordinator)."""

    # V1 status
    v1_tools_generated: int
    v1_pending_proposals: int
    v1_avg_success_rate: float

    # V2 status
    v2_active_plans: int
    v2_optimization_runs: int
    v2_avg_improvement: float

    # V3 coordination
    v1_tools_used_in_v2: int
    v2_patterns_triggering_v1: int
    compound_gain_factor: float

    # System health
    total_events_processed: int
    last_updated: datetime


@dataclass
class CompoundGains:
    """Tracking of compound improvements across V1+V2 (simplified for coordinator)."""

    time_period: str
    individual_gains: List[Dict[str, Any]]
    compound_factor: float
    total_time_saved_ms: float
    total_cost_saved_cents: float


class V3Coordinator:
    """Main coordinator for V3 unified adaptive tools system."""

    def __init__(
        self,
        registry: UnifiedRegistry,
        collector: TelemetryCollector,
        router: TelemetryRouter,
        v1_system: Optional["V1System"] = None,
        v2_system: Optional["V2System"] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize V3 coordinator.

        Args:
            registry: Unified tool/plan registry
            collector: Telemetry collector
            router: Telemetry router
            v1_system: Optional V1 system instance
            v2_system: Optional V2 system instance
            config: Configuration dictionary
        """
        self.registry = registry
        self.collector = collector
        self.router = router
        self.v1 = v1_system
        self.v2 = v2_system
        self.config = config or {}

        # Decision thresholds from config
        self.v1_sequence_threshold = self.config.get("v1_sequence_threshold", 10)
        self.v1_failure_threshold = self.config.get("v1_failure_threshold", 0.2)
        self.v2_latency_threshold_ms = self.config.get("v2_latency_threshold_ms", 500)
        self.v2_cost_threshold_cents = self.config.get("v2_cost_threshold_cents", 10)

        # State tracking
        self.events_processed = 0
        self.v1_triggers = 0
        self.v2_triggers = 0
        self._running = False

    async def start(self) -> None:
        """Initialize and start the V3 coordinator."""
        self._running = True

        # Initialize V1 and V2 if provided
        if self.v1:
            await self.v1.initialize()

        if self.v2:
            await self.v2.initialize()

        # Start event processing loop
        asyncio.create_task(self._event_processing_loop())

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False

    async def _event_processing_loop(self) -> None:
        """Main event processing loop."""
        # Track processed events to avoid reprocessing
        processed_event_ids = set()
        
        while self._running:
            try:
                # Get events from last hour (would use queue in real implementation)
                events = await self.collector.get_recent(hours=1)
                
                # Process only new events
                new_events = [e for e in events if e.event_id not in processed_event_ids]
                
                for event in new_events[:100]:  # Limit to 100 per iteration
                    await self.process_telemetry(event)
                    processed_event_ids.add(event.event_id)
                    self.events_processed += 1
                    
                    # Clean up old event IDs to prevent memory growth
                    if len(processed_event_ids) > 1000:
                        processed_event_ids.clear()

                # Sleep to avoid busy loop
                await asyncio.sleep(1.0)

            except Exception as e:
                # Log error but continue
                print(f"Error in event processing loop: {e}")
                await asyncio.sleep(5.0)

    async def process_telemetry(self, event: TelemetryEvent) -> None:
        """Process telemetry event and route to appropriate system.

        Args:
            event: Telemetry event to process
        """
        # Route event to V1 and/or V2
        routed_event = await self.router.route(event)
        routed_v1 = routed_event.routed_to_v1
        routed_v2 = routed_event.routed_to_v2

        # Handle V1 triggers
        if routed_v1 and self._should_generate_tool(event):
            print(f"DEBUG: V1 trigger condition met for event {event.event_type}, item={event.item_id}")
            await self._v1_generate_flow(event)

        # Handle V2 triggers
        if routed_v2 and self._should_optimize_workflow(event):
            print(f"DEBUG: V2 trigger condition met for event {event.event_type}")
            await self._v2_optimize_flow(event)

    def _should_generate_tool(self, event: TelemetryEvent) -> bool:
        """Determine if V1 tool generation should be triggered.

        Args:
            event: Telemetry event

        Returns:
            True if V1 should be activated
        """
        # Check V1 trigger conditions
        if event.event_type == EventType.TOOL_SEQUENCE:
            # Repeated manual sequences
            sequence_count = event.event_data.get("count", 0)
            should_trigger = sequence_count >= self.v1_sequence_threshold
            if should_trigger:
                print(f"DEBUG: TOOL_SEQUENCE trigger: count={sequence_count}, threshold={self.v1_sequence_threshold}")
            return should_trigger

        elif event.event_type == EventType.TOOL_ERROR:
            # High failure rate
            error_rate = event.event_data.get("error_rate", 0)
            return error_rate >= self.v1_failure_threshold

        elif event.event_type == EventType.TOOL_NOT_FOUND:
            # Missing capability
            return True

        return False

    def _should_optimize_workflow(self, event: TelemetryEvent) -> bool:
        """Determine if V2 workflow optimization should be triggered.

        Args:
            event: Telemetry event

        Returns:
            True if V2 should be activated
        """
        # Check V2 trigger conditions
        if event.event_type == EventType.TOOL_EXECUTED:
            # Latency bottleneck
            if event.latency_ms and event.latency_ms > self.v2_latency_threshold_ms:
                return True

            # Cost inefficiency
            if event.cost_cents and event.cost_cents > self.v2_cost_threshold_cents:
                return True

        elif event.event_type == EventType.PLAN_EXECUTED:
            # Plan performance issues
            avg_latency = event.event_data.get("avg_latency_ms", 0)
            return avg_latency > self.v2_latency_threshold_ms

        return False

    async def _v1_generate_flow(self, event: TelemetryEvent) -> None:
        """Execute V1 tool generation flow.

        Args:
            event: Triggering event
        """
        if not self.v1:
            print("DEBUG: V1 system not initialized")
            return

        try:
            self.v1_triggers += 1
            print(f"DEBUG: V1 generate flow triggered (trigger #{self.v1_triggers})")

            # 1. Pattern analysis
            pattern = await self.v1.analyze_pattern(event)

            if not pattern:
                print("DEBUG: No pattern detected")
                return

            print(f"DEBUG: Pattern detected: {pattern.get('description', 'unknown')[:50]}")

            # 2. Generate tool code
            proposal = await self.v1.generate_proposal(pattern)
            
            if not proposal:
                print("DEBUG: Failed to generate proposal")
                return

            print(f"DEBUG: Proposal generated: {proposal.name}")

            # 3. Safety validation + sandbox
            safety_report = await self.v1.validate_safety(proposal)

            if safety_report.risk_level == "CRITICAL":
                print(f"DEBUG: Critical risk detected, requesting approval")
                # Request human approval
                await self.v1.request_approval(proposal)
                return

            sandbox_result = await self.v1.test_in_sandbox(proposal)

            if not sandbox_result.passed:
                print(f"DEBUG: Sandbox tests failed (passed: {sandbox_result.test_cases_passed}/{sandbox_result.test_cases_run})")
                if sandbox_result.errors:
                    print(f"DEBUG: Sandbox errors: {sandbox_result.errors[:3]}")  # Show first 3 errors
                # Still register tool but mark for review (similar to V1System behavior)
                print(f"DEBUG: Registering tool with review flag due to sandbox failures")
                proposal.function_schema["requires_review"] = True
                proposal.function_schema["sandbox_result"] = {
                    "passed": sandbox_result.passed,
                    "test_cases_run": sandbox_result.test_cases_run,
                    "test_cases_passed": sandbox_result.test_cases_passed,
                    "errors": sandbox_result.errors,
                }
            else:
                print(f"DEBUG: Sandbox tests passed, registering tool")

            # 4. Register tool in unified registry
            from ..contracts.models import SystemType
            await self.registry.register_tool(
                name=proposal.name,
                source_code=proposal.source_code,
                function_schema=proposal.function_schema,
                system=SystemType.V1
            )

            print(f"DEBUG: Tool '{proposal.name}' registered successfully")

            # 5. Notify V2 of new tool
            if self.v2:
                await self.v2.refresh_tool_catalog()

        except Exception as e:
            import traceback
            print(f"Error in V1 generate flow: {e}")
            print(traceback.format_exc())

    async def _v2_optimize_flow(self, event: TelemetryEvent) -> None:
        """Execute V2 workflow optimization flow.

        Args:
            event: Triggering event
        """
        if not self.v2:
            return

        try:
            self.v2_triggers += 1

            # 1. Identify optimization target
            target = await self.v2.identify_bottleneck(event)

            if not target:
                return

            # 2. Generate initial plan (may use V1 tools)
            base_plan = await self.v2.generate_plan(target)

            # 3. Optimize hyperparameters
            optimized_plan = await self.v2.optimize_plan(base_plan)

            # 4. A/B test
            winner = await self.v2.ab_test(base_plan, optimized_plan)

            # 5. Register winning plan
            await self.registry.register_plan(winner, system="v2")

        except Exception as e:
            print(f"Error in V2 optimize flow: {e}")

    async def get_status(self) -> V3Status:
        """Get unified system status.

        Returns:
            Current V3 system status
        """
        # Query V1 stats
        v1_stats = await self.registry.get_v1_stats() if self.v1 else {}
        print(f"DEBUG: V1 stats query result: {v1_stats}")

        # Query V2 stats
        v2_stats = await self.registry.get_v2_stats() if self.v2 else {}

        # Calculate cross-system metrics
        cross_system = await self.registry.get_cross_system_stats()

        return V3Status(
            # V1
            v1_tools_generated=v1_stats.get("tools_generated", 0),
            v1_pending_proposals=v1_stats.get("pending_proposals", 0),
            v1_avg_success_rate=v1_stats.get("avg_success_rate", 0.0),
            # V2
            v2_active_plans=v2_stats.get("active_plans", 0),
            v2_optimization_runs=v2_stats.get("optimization_runs", 0),
            v2_avg_improvement=v2_stats.get("avg_improvement", 0.0),
            # V3
            v1_tools_used_in_v2=cross_system.get("v1_in_v2", 0),
            v2_patterns_triggering_v1=cross_system.get("v2_in_v1", 0),
            compound_gain_factor=cross_system.get("compound_factor", 1.0),
            # System
            total_events_processed=self.events_processed,
            last_updated=datetime.utcnow(),
        )

    async def get_compound_gains(self, since: datetime) -> CompoundGains:
        """Calculate compound gains over time period.

        Args:
            since: Start time for calculation

        Returns:
            Compound gains analysis
        """
        # Query improvement records from registry
        improvements = await self.registry.get_improvements_since(since)

        # Calculate individual and compound gains
        individual_gains = []
        compound_factor = 1.0
        total_time_saved = 0.0
        total_cost_saved = 0.0

        for improvement in improvements:
            # V1 or V2 improvement
            individual_gain = improvement.get("improvement_factor", 1.0)
            compound_factor *= individual_gain

            # Track savings
            total_time_saved += improvement.get("time_saved_ms", 0)
            total_cost_saved += improvement.get("cost_saved_cents", 0)

            individual_gains.append(improvement)

        time_period = f"{since.isoformat()} to {datetime.utcnow().isoformat()}"

        return CompoundGains(
            time_period=time_period,
            individual_gains=individual_gains,
            compound_factor=compound_factor,
            total_time_saved_ms=total_time_saved,
            total_cost_saved_cents=total_cost_saved,
        )

    async def force_sync(self) -> None:
        """Force synchronization between V1 and V2."""
        if self.v2:
            # Refresh V2's view of V1 tools
            await self.v2.refresh_tool_catalog()

        if self.v1:
            # Refresh V1's view of V2 patterns
            await self.v1.refresh_usage_patterns()

    async def shutdown(self) -> None:
        """Gracefully shutdown the coordinator."""
        await self.stop()

        if self.v1:
            await self.v1.shutdown()

        if self.v2:
            await self.v2.shutdown()
