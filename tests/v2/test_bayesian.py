"""Unit tests for Bayesian optimizer."""

import pytest

from accuralai_adaptive_tools.contracts.models import ExecutionMetrics, Plan, PlanStep, QualitySignals
from accuralai_adaptive_tools.v2.optimization.bayesian import Objective, PlanOptimizer, SearchDimension


@pytest.fixture
def optimizer():
    """Create optimizer instance."""
    return PlanOptimizer(n_calls=20, random_state=42)


@pytest.fixture
def test_plan():
    """Create test plan with tunable parameters."""
    return Plan(
        name="test_plan",
        version="1.0.0",
        description="Test plan",
        steps=[
            PlanStep(
                id="step1",
                tool="test.tool",
                with_args={"value": "test"},
                save_as="result1",
                strategy={"type": "cached", "config": {"ttl_seconds": 300}},
                timeout_ms=5000,
            )
        ],
        constraints={"max_latency_ms": 10000, "max_cost_cents": 5},
    )


def test_extract_search_space(optimizer, test_plan):
    """Test search space extraction."""
    dimensions = optimizer.extract_search_space(test_plan)

    assert len(dimensions) > 0

    # Should find timeout parameter
    timeout_dims = [d for d in dimensions if "timeout_ms" in d.name]
    assert len(timeout_dims) > 0

    # Should find cache TTL parameter
    ttl_dims = [d for d in dimensions if "ttl_seconds" in d.name]
    assert len(ttl_dims) > 0


def test_suggest_parameters_random(optimizer, test_plan):
    """Test random parameter suggestion."""
    params = optimizer.suggest_parameters(test_plan)

    assert isinstance(params, dict)
    assert len(params) > 0


def test_suggest_parameters_bayesian(optimizer, test_plan):
    """Test Bayesian parameter suggestion."""
    # Record several trials first
    for i in range(10):
        params = optimizer.suggest_parameters(test_plan)

        metrics = ExecutionMetrics(
            latency_ms=1000 - i * 50,  # Improving
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

    # Now it should use Bayesian optimization
    params = optimizer.suggest_parameters(test_plan)
    assert isinstance(params, dict)


def test_record_trial(optimizer):
    """Test trial recording."""
    params = {"step1.timeout_ms": 5000, "step1.strategy.ttl_seconds": 300}

    metrics = ExecutionMetrics(
        latency_ms=500,
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

    trial = optimizer.record_trial(params, metrics, quality)

    assert trial.trial_id == 0
    assert trial.params == params
    assert trial.score != 0


def test_get_best_trial(optimizer):
    """Test getting best trial."""
    assert optimizer.get_best_trial() is None

    # Record trials with different scores
    for latency in [1000, 800, 600, 900]:
        params = {"test_param": latency}

        metrics = ExecutionMetrics(
            latency_ms=latency,
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
    # Best should have lowest latency (600ms)
    assert best.params["test_param"] == 600


def test_objective_function():
    """Test multi-objective scoring."""
    objective = Objective()

    metrics = ExecutionMetrics(
        latency_ms=500,
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

    score = objective.score(metrics, quality)

    assert isinstance(score, float)
    # Score should be reasonable
    assert -10 < score < 10


def test_optimization_summary(optimizer):
    """Test optimization summary generation."""
    summary = optimizer.get_optimization_summary()

    assert summary["trials"] == 0
    assert summary["best_score"] is None

    # Record some trials
    for i in range(5):
        params = {"param": i * 100}

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

    summary = optimizer.get_optimization_summary()

    assert summary["trials"] == 5
    assert summary["best_score"] is not None
    assert "mean_score" in summary
    assert "improvement" in summary


def test_apply_parameters(optimizer, test_plan):
    """Test applying parameters to plan."""
    params = {"step1.timeout_ms": 8000, "constraints.max_latency_ms": 15000}

    modified_plan = optimizer.apply_parameters(test_plan, params)

    # Verify parameters were applied
    assert modified_plan.steps[0].timeout_ms == 8000
    assert modified_plan.constraints["max_latency_ms"] == 15000
