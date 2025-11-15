"""Unit tests for V2 PlanLang executor."""

import pytest

from accuralai_adaptive_tools.contracts.models import Plan, PlanStep
from accuralai_adaptive_tools.v2.execution.executor import PlanExecutor


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self):
        self.tools = {
            "test.tool": {"name": "test.tool", "execute": self._mock_execute},
            "slow.tool": {"name": "slow.tool", "execute": self._mock_slow},
            "failing.tool": {"name": "failing.tool", "execute": self._mock_fail},
        }

    def get(self, name: str):
        """Get tool by name."""
        return self.tools.get(name)

    async def _mock_execute(self, **kwargs):
        """Mock tool execution."""
        return {"result": "success", "args": kwargs}

    async def _mock_slow(self, **kwargs):
        """Mock slow tool."""
        import asyncio

        await asyncio.sleep(0.1)
        return {"result": "slow"}

    async def _mock_fail(self, **kwargs):
        """Mock failing tool."""
        raise ValueError("Tool failed")


class MockCache:
    """Mock cache for testing."""

    def __init__(self):
        self._data = {}

    async def get(self, key: str):
        """Get from cache."""
        return self._data.get(key)

    async def set(self, key: str, value, ttl: int = 300):
        """Set in cache."""
        self._data[key] = value


@pytest.fixture
def registry():
    """Create mock registry."""
    return MockToolRegistry()


@pytest.fixture
def executor(registry):
    """Create executor."""
    return PlanExecutor(registry)


@pytest.mark.asyncio
async def test_simple_execution(executor):
    """Test simple plan execution."""
    plan = Plan(
        name="simple_test",
        version="1.0.0",
        description="Simple test plan",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"arg1": "value1"},
                save_as="result1",
            )
        ],
    )

    result = await executor.execute(plan, {})

    assert result.success
    assert "result1" in result.context
    assert result.metrics.latency_ms >= 0


@pytest.mark.asyncio
async def test_variable_substitution(executor):
    """Test variable substitution in arguments."""
    plan = Plan(
        name="var_test",
        version="1.0.0",
        description="Variable substitution test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"data": "${inputs.test_data}"},
                save_as="result1",
            )
        ],
        inputs=[{"name": "test_data", "type": "string"}],
    )

    result = await executor.execute(plan, {"test_data": "hello"})

    assert result.success
    assert result.context["result1"]["args"]["data"] == "hello"


@pytest.mark.asyncio
async def test_sequential_dependencies(executor):
    """Test sequential step execution with dependencies."""
    plan = Plan(
        name="sequential_test",
        version="1.0.0",
        description="Sequential dependencies test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"value": "first"},
                save_as="result1",
            ),
            PlanStep(
                id="step2",
                tool="test.tool",
                with_args={"previous": "${result1}"},
                save_as="result2",
                depends_on=["step1"],
            ),
        ],
    )

    result = await executor.execute(plan, {})

    assert result.success
    assert "result1" in result.context
    assert "result2" in result.context


@pytest.mark.asyncio
async def test_cached_execution(executor):
    """Test cached execution strategy."""
    cache = MockCache()
    executor.cache = cache

    plan = Plan(
        name="cache_test",
        version="1.0.0",
        description="Caching test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"value": "cached"},
                save_as="result1",
                strategy={"type": "cached", "config": {"ttl_seconds": 300}},
            )
        ],
    )

    # First execution
    result1 = await executor.execute(plan, {})
    assert result1.success

    # Second execution (should hit cache)
    result2 = await executor.execute(plan, {})
    assert result2.success
    assert result2.metrics.cache_hit


@pytest.mark.asyncio
async def test_retry_strategy(executor):
    """Test retry execution strategy."""
    plan = Plan(
        name="retry_test",
        version="1.0.0",
        description="Retry test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",  # Use working tool for this test
                with_args={"value": "retry"},
                save_as="result1",
                strategy={"type": "retry", "config": {"max_attempts": 3, "initial_delay_ms": 10}},
            )
        ],
    )

    result = await executor.execute(plan, {})
    assert result.success


@pytest.mark.asyncio
async def test_error_handling_continue(executor):
    """Test error handling with continue strategy."""
    plan = Plan(
        name="error_continue_test",
        version="1.0.0",
        description="Error handling test",
        steps=[
            PlanStep(
                id="step1",
                tool="failing.tool",
                with_args={},
                save_as="result1",
                error_handling={"on_failure": "continue"},
            ),
            PlanStep(
                id="step2",
                tool="test.tool",
                with_args={},
                save_as="result2",
            ),
        ],
    )

    result = await executor.execute(plan, {})
    # Plan should complete despite step1 failing
    assert result.success


@pytest.mark.asyncio
async def test_conditional_execution(executor):
    """Test conditional step execution."""
    plan = Plan(
        name="conditional_test",
        version="1.0.0",
        description="Conditional execution test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"flag": True},
                save_as="result1",
            ),
            PlanStep(
                id="step2",
                tool="test.tool",
                with_args={},
                save_as="result2",
                conditional="${result1.flag}",  # Simplified conditional
            ),
        ],
    )

    result = await executor.execute(plan, {})
    assert result.success


@pytest.mark.asyncio
async def test_input_validation(executor):
    """Test input validation."""
    plan = Plan(
        name="validation_test",
        version="1.0.0",
        description="Input validation test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"data": "${inputs.required_field}"},
                save_as="result1",
            )
        ],
        inputs=[{"name": "required_field", "type": "string", "required": True}],
    )

    # Missing required input should fail
    result = await executor.execute(plan, {})
    assert not result.success
    assert "required_field" in result.error


@pytest.mark.asyncio
async def test_execution_metrics(executor):
    """Test execution metrics collection."""
    plan = Plan(
        name="metrics_test",
        version="1.0.0",
        description="Metrics test",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={},
                save_as="result1",
            )
        ],
    )

    result = await executor.execute(plan, {})

    assert result.success
    assert result.metrics.latency_ms >= 0
    assert result.metrics.success is True
    assert result.metrics.retry_count >= 0
