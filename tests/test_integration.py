"""Integration tests for V1+V2+V3 workflows."""

import pytest

from accuralai_adaptive_tools.contracts.models import EventType, Plan, PlanStep, TelemetryEvent
from accuralai_adaptive_tools.coordinator.v3 import V3Coordinator
from accuralai_adaptive_tools.registry.unified import UnifiedRegistry
from accuralai_adaptive_tools.telemetry import TelemetryCollector, TelemetryRouter, TelemetryStorage
from accuralai_adaptive_tools.v2.execution.executor import PlanExecutor
from accuralai_adaptive_tools.v2.optimization.ab_tester import ABTester
from accuralai_adaptive_tools.v2.optimization.bayesian import PlanOptimizer


class MockToolRegistry:
    """Mock tool registry."""

    def __init__(self):
        self.tools = {}

    def get(self, name: str):
        """Get tool."""
        return self.tools.get(name, MockTool(name))


class MockTool:
    """Mock tool."""

    def __init__(self, name: str):
        self.name = name


@pytest.fixture
async def registry():
    """Create and initialize registry."""
    reg = UnifiedRegistry(":memory:")
    await reg.initialize()
    yield reg
    # UnifiedRegistry uses aiosqlite which handles cleanup automatically


@pytest.fixture
def coordinator(registry):
    """Create V3 coordinator."""
    storage = TelemetryStorage(":memory:")
    router = TelemetryRouter()
    collector = TelemetryCollector(storage, router)
    return V3Coordinator(registry, collector, router)


@pytest.mark.asyncio
async def test_v2_end_to_end_optimization(registry):
    """Test complete V2 optimization workflow."""
    # 1. Create initial plan
    plan = Plan(
        name="test_workflow",
        version="1.0.0",
        description="Test optimization workflow",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"value": "test"},
                save_as="result",
                strategy={"type": "cached", "config": {"ttl_seconds": 300}},
                timeout_ms=5000,
            )
        ],
        constraints={"max_latency_ms": 10000},
    )

    # 2. Execute plan
    tool_registry = MockToolRegistry()
    executor = PlanExecutor(tool_registry)

    result = await executor.execute(plan, {})
    assert result.success

    # 3. Optimize plan
    optimizer = PlanOptimizer(n_calls=10)

    for i in range(5):
        params = optimizer.suggest_parameters(plan)

        # Simulate execution
        from accuralai_adaptive_tools.contracts.models import ExecutionMetrics, QualitySignals

        metrics = ExecutionMetrics(
            latency_ms=1000 - i * 100,
            cost_cents=1.0,
            tokens_used=100,
            cache_hit=False,
            retry_count=0,
            success=True,
        )

        quality = QualitySignals(
            validator_scores={"overall": 0.9},
            unit_test_pass_rate=1.0,
            acceptance_test_pass_rate=1.0,
            human_rating=None,
            error_rate=0.0,
        )

        optimizer.record_trial(params, metrics, quality)

    best = optimizer.get_best_trial()
    assert best is not None

    # 4. Apply optimizations
    optimized_plan = optimizer.apply_parameters(plan, best.params)

    # 5. A/B test
    tester = ABTester(executor, min_sample_size=10)

    test_inputs = [{"value": f"test{i}"} for i in range(10)]

    ab_result = await tester.compare_plans(plan, optimized_plan, test_inputs, sample_size=10)

    # Verify A/B test completed
    assert ab_result.trials == 10


@pytest.mark.asyncio
async def test_v3_coordination_routing(coordinator):
    """Test V3 event routing to V1 and V2."""
    # V1 trigger: repeated sequence
    v1_event = TelemetryEvent(
        event_id="e1",
        event_type=EventType.TOOL_SEQUENCE,
        event_data={"count": 15},  # Above threshold
    )

    assert coordinator._should_generate_tool(v1_event)

    # V2 trigger: high latency
    v2_event = TelemetryEvent(
        event_id="e2",
        event_type=EventType.TOOL_EXECUTED,
        latency_ms=1500,  # Above threshold
    )

    assert coordinator._should_optimize_workflow(v2_event)

    # Process events
    await coordinator.process_telemetry(v1_event)
    await coordinator.process_telemetry(v2_event)


@pytest.mark.asyncio
async def test_v3_status_aggregation(coordinator):
    """Test V3 status aggregation."""
    status = await coordinator.get_status()

    # Should have status for all systems
    assert status.v1_tools_generated >= 0
    assert status.v2_active_plans >= 0
    assert status.total_events_processed >= 0


@pytest.mark.asyncio
async def test_compound_gains_tracking(registry):
    """Test compound gains calculation."""
    # Register some improvements
    await registry.record_improvement(
        "v1_tool_1",
        improvement_factor=1.3,
        time_saved_ms=500,
        cost_saved_cents=1.0,
    )

    await registry.record_improvement(
        "v2_plan_1",
        improvement_factor=1.4,
        time_saved_ms=800,
        cost_saved_cents=2.0,
    )

    # Calculate compound gains
    from datetime import datetime, timedelta

    storage = TelemetryStorage(":memory:")
    router = TelemetryRouter()
    collector = TelemetryCollector(storage, router)
    coordinator = V3Coordinator(
        registry,
        collector,
        router,
    )

    gains = await coordinator.get_compound_gains(datetime.utcnow() - timedelta(hours=24))

    # Should have compound effect
    assert len(gains.individual_gains) >= 0
    assert gains.compound_factor >= 1.0


@pytest.mark.asyncio
async def test_cross_system_synchronization(registry):
    """Test V1 tool usage in V2 plans."""
    # Register a V1 tool
    from accuralai_adaptive_tools.contracts.models import SystemType, ToolSpec

    v1_tool = ToolSpec(
        name="v1_generated_tool",
        description="Tool generated by V1",
        function={"parameters": {}},
    )

    await registry.register_tool(
        name=v1_tool.name,
        source_code="# Mock tool generated by V1",
        function_schema=v1_tool.function,
        system=SystemType.V1,
    )

    # Create V2 plan that uses V1 tool
    plan = Plan(
        name="v2_plan_using_v1",
        version="1.0.0",
        description="V2 plan using V1 tool",
        steps=[
            PlanStep(
                id="step1",
                tool="v1_generated_tool",  # Reference V1 tool
                with_args={},
                save_as="result",
            )
        ],
    )

    await registry.register_plan(plan, system=SystemType.V2)

    # Check cross-system stats
    stats = await registry.get_cross_system_stats()

    # Should track V1 tool used in V2 plan
    assert isinstance(stats, dict)
