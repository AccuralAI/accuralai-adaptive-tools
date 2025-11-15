"""A/B testing framework for plan comparison."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats

from ...contracts.models import ExecutionMetrics, Plan, QualitySignals
from ..execution.executor import ExecutionResult, PlanExecutor


@dataclass
class ABTestResult:
    """Result of A/B test comparison."""

    plan_a_name: str
    plan_b_name: str
    trials: int

    # Plan A metrics
    plan_a_latency_mean: float
    plan_a_latency_std: float
    plan_a_success_rate: float
    plan_a_cost_mean: float

    # Plan B metrics
    plan_b_latency_mean: float
    plan_b_latency_std: float
    plan_b_success_rate: float
    plan_b_cost_mean: float

    # Statistical comparison
    latency_p_value: float
    success_p_value: float
    latency_improvement_pct: float
    cost_improvement_pct: float

    # Winner determination
    winner: str  # 'A', 'B', or 'tie'
    confidence: float  # 0.0-1.0
    recommendation: str


class ABTester:
    """Statistical A/B testing for plans."""

    def __init__(
        self,
        executor: PlanExecutor,
        significance_level: float = 0.05,
        min_sample_size: int = 30,
        max_sample_size: int = 100,
    ):
        """Initialize A/B tester.

        Args:
            executor: Plan executor for running tests
            significance_level: Statistical significance threshold (default 0.05)
            min_sample_size: Minimum samples per variant
            max_sample_size: Maximum samples per variant
        """
        self.executor = executor
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size

    async def compare_plans(
        self,
        plan_a: Plan,
        plan_b: Plan,
        test_inputs: List[dict],
        sample_size: Optional[int] = None,
    ) -> ABTestResult:
        """Run A/B test comparing two plans.

        Args:
            plan_a: First plan (baseline)
            plan_b: Second plan (variant)
            test_inputs: List of test input dictionaries
            sample_size: Optional fixed sample size

        Returns:
            A/B test results with winner determination
        """
        n_samples = sample_size or self.min_sample_size

        # Run trials for both plans
        results_a = await self._run_trials(plan_a, test_inputs, n_samples)
        results_b = await self._run_trials(plan_b, test_inputs, n_samples)

        # Extract metrics
        latencies_a = [r.metrics.latency_ms for r in results_a if r.success]
        latencies_b = [r.metrics.latency_ms for r in results_b if r.success]

        costs_a = [r.metrics.cost_cents for r in results_a if r.success]
        costs_b = [r.metrics.cost_cents for r in results_b if r.success]

        success_a = [1 if r.success else 0 for r in results_a]
        success_b = [1 if r.success else 0 for r in results_b]

        # Calculate statistics
        latency_stats = self._compare_continuous(latencies_a, latencies_b)
        cost_stats = self._compare_continuous(costs_a, costs_b)
        success_stats = self._compare_binary(success_a, success_b)

        # Determine winner
        winner, confidence, recommendation = self._determine_winner(
            latency_stats,
            cost_stats,
            success_stats,
        )

        return ABTestResult(
            plan_a_name=plan_a.name,
            plan_b_name=plan_b.name,
            trials=n_samples,
            # Plan A metrics
            plan_a_latency_mean=np.mean(latencies_a) if latencies_a else 0,
            plan_a_latency_std=np.std(latencies_a) if latencies_a else 0,
            plan_a_success_rate=np.mean(success_a),
            plan_a_cost_mean=np.mean(costs_a) if costs_a else 0,
            # Plan B metrics
            plan_b_latency_mean=np.mean(latencies_b) if latencies_b else 0,
            plan_b_latency_std=np.std(latencies_b) if latencies_b else 0,
            plan_b_success_rate=np.mean(success_b),
            plan_b_cost_mean=np.mean(costs_b) if costs_b else 0,
            # Statistical comparison
            latency_p_value=latency_stats["p_value"],
            success_p_value=success_stats["p_value"],
            latency_improvement_pct=latency_stats["improvement_pct"],
            cost_improvement_pct=cost_stats["improvement_pct"],
            # Winner
            winner=winner,
            confidence=confidence,
            recommendation=recommendation,
        )

    async def _run_trials(self, plan: Plan, test_inputs: List[dict], n_samples: int) -> List[ExecutionResult]:
        """Run multiple trials of a plan.

        Args:
            plan: Plan to execute
            test_inputs: Test input sets
            n_samples: Number of samples to run

        Returns:
            List of execution results
        """
        results = []

        for i in range(n_samples):
            # Cycle through test inputs
            inputs = test_inputs[i % len(test_inputs)]

            try:
                result = await self.executor.execute(plan, inputs)
                results.append(result)
            except Exception as e:
                # Record failure
                results.append(
                    ExecutionResult(
                        success=False,
                        context={},
                        metrics=ExecutionMetrics(),
                        validation_results=[],
                        error=str(e),
                    )
                )

        return results

    def _compare_continuous(self, values_a: List[float], values_b: List[float]) -> dict:
        """Compare continuous metrics (latency, cost) using t-test.

        Args:
            values_a: Values from plan A
            values_b: Values from plan B

        Returns:
            Statistical comparison results
        """
        if not values_a or not values_b:
            return {
                "p_value": 1.0,
                "significant": False,
                "improvement_pct": 0.0,
                "mean_a": 0.0,
                "mean_b": 0.0,
            }

        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)

        # Two-sample t-test
        statistic, p_value = stats.ttest_ind(values_a, values_b)

        # Calculate improvement (negative = B is better)
        improvement_pct = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0

        return {
            "p_value": p_value,
            "significant": p_value < self.significance_level,
            "improvement_pct": improvement_pct,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "statistic": statistic,
        }

    def _compare_binary(self, values_a: List[int], values_b: List[int]) -> dict:
        """Compare binary metrics (success/failure) using chi-square test.

        Args:
            values_a: Binary values from plan A
            values_b: Binary values from plan B

        Returns:
            Statistical comparison results
        """
        if not values_a or not values_b:
            return {
                "p_value": 1.0,
                "significant": False,
                "rate_a": 0.0,
                "rate_b": 0.0,
            }

        rate_a = np.mean(values_a)
        rate_b = np.mean(values_b)

        # Contingency table
        successes_a = sum(values_a)
        failures_a = len(values_a) - successes_a
        successes_b = sum(values_b)
        failures_b = len(values_b) - successes_b

        contingency = np.array([[successes_a, failures_a], [successes_b, failures_b]])

        # Check for edge cases that would cause chi-square test to fail
        # If all values are the same (no variation), return not significant
        if np.all(contingency == 0) or (successes_a == len(values_a) and successes_b == len(values_b)) or (failures_a == len(values_a) and failures_b == len(values_b)):
            return {
                "p_value": 1.0,
                "significant": False,
                "rate_a": rate_a,
                "rate_b": rate_b,
                "chi2": 0.0,
            }

        # Check if contingency table has zeros that would cause issues
        # Add small epsilon to avoid zero frequencies
        if np.any(contingency == 0):
            contingency = contingency + 0.5

        # Chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except ValueError:
            # If chi-square test still fails, fall back to not significant
            return {
                "p_value": 1.0,
                "significant": False,
                "rate_a": rate_a,
                "rate_b": rate_b,
                "chi2": 0.0,
            }

        return {
            "p_value": p_value,
            "significant": p_value < self.significance_level,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "chi2": chi2,
        }

    def _determine_winner(
        self,
        latency_stats: dict,
        cost_stats: dict,
        success_stats: dict,
    ) -> tuple:
        """Determine overall winner based on multiple metrics.

        Args:
            latency_stats: Latency comparison results
            cost_stats: Cost comparison results
            success_stats: Success rate comparison results

        Returns:
            Tuple of (winner, confidence, recommendation)
        """
        # Scoring system
        score_b = 0  # Positive score favors B, negative favors A
        confidence = 0.0

        # Success rate (most important)
        if success_stats["significant"]:
            if success_stats["rate_b"] > success_stats["rate_a"]:
                score_b += 3
                confidence += 0.4
            else:
                score_b -= 3
                confidence += 0.4

        # Latency (important)
        if latency_stats["significant"]:
            if latency_stats["improvement_pct"] < 0:  # B is faster
                score_b += 2
                confidence += 0.3
            else:
                score_b -= 2
                confidence += 0.3

        # Cost (less important)
        if cost_stats["significant"]:
            if cost_stats["improvement_pct"] < 0:  # B is cheaper
                score_b += 1
                confidence += 0.2
            else:
                score_b -= 1
                confidence += 0.2

        # Determine winner
        if score_b > 1:
            winner = "B"
            recommendation = f"Deploy {latency_stats.get('mean_b', 0):.0f}ms avg"
        elif score_b < -1:
            winner = "A"
            recommendation = "Keep current version"
        else:
            winner = "tie"
            recommendation = "No clear winner, collect more data"

        return winner, min(confidence, 1.0), recommendation

    def sequential_test(
        self,
        plan_a: Plan,
        plan_b: Plan,
        test_inputs: List[dict],
        early_stop: bool = True,
    ) -> ABTestResult:
        """Run sequential A/B test with early stopping.

        Args:
            plan_a: Baseline plan
            plan_b: Variant plan
            test_inputs: Test inputs
            early_stop: Whether to stop early if winner clear

        Returns:
            A/B test results
        """
        # This would implement sequential probability ratio test (SPRT)
        # For now, just run fixed sample size
        import asyncio

        return asyncio.run(
            self.compare_plans(plan_a, plan_b, test_inputs, sample_size=self.min_sample_size)
        )
