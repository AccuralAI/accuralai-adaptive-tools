"""Bayesian optimization for plan hyperparameter tuning."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from ...contracts.models import ExecutionMetrics, Plan, QualitySignals


@dataclass
class SearchDimension:
    """Definition of a tunable parameter dimension."""

    name: str
    type: str  # 'int', 'float', 'categorical'
    range: Optional[Tuple] = None
    categories: Optional[List] = None
    default: Any = None


@dataclass
class Trial:
    """Single optimization trial."""

    trial_id: int
    params: Dict[str, Any]
    score: float
    metrics: ExecutionMetrics
    quality: QualitySignals


@dataclass
class Objective:
    """Multi-objective optimization function."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "latency": -1.0,  # Minimize
            "cost": -0.5,  # Minimize
            "quality": 2.0,  # Maximize
            "reliability": 1.5,  # Maximize
        }
    )

    def score(self, metrics: ExecutionMetrics, quality: QualitySignals) -> float:
        """Calculate weighted score.

        Args:
            metrics: Execution metrics
            quality: Quality signals

        Returns:
            Composite score (higher is better)
        """
        # Normalize latency (assume 10s is baseline)
        latency_norm = metrics.latency_ms / 10000.0

        # Normalize cost (assume $0.10 is baseline)
        cost_norm = metrics.cost_cents / 10.0

        # Quality score (0-1)
        quality_score = quality.validator_scores.get("overall", 0.5)

        # Reliability (success rate)
        reliability = 1.0 - quality.error_rate

        score = (
            self.weights["latency"] * latency_norm
            + self.weights["cost"] * cost_norm
            + self.weights["quality"] * quality_score
            + self.weights["reliability"] * reliability
        )

        return score


class PlanOptimizer:
    """Bayesian optimizer for plan hyperparameters."""

    def __init__(self, objective: Optional[Objective] = None, n_calls: int = 50, random_state: int = 42):
        """Initialize optimizer.

        Args:
            objective: Optimization objective function
            n_calls: Number of optimization iterations
            random_state: Random seed for reproducibility
        """
        self.objective = objective or Objective()
        self.n_calls = n_calls
        self.random_state = random_state
        self.trials: List[Trial] = []
        self.search_space: Optional[List[SearchDimension]] = None
        self.skopt_space: Optional[List] = None

    def extract_search_space(self, plan: Plan) -> List[SearchDimension]:
        """Extract tunable parameters from plan.

        Args:
            plan: Plan to analyze

        Returns:
            List of tunable dimensions
        """
        dimensions = []

        for step in plan.steps:
            step_prefix = f"{step.id}."

            # Timeout is tunable
            if step.timeout_ms:
                dimensions.append(
                    SearchDimension(
                        name=f"{step_prefix}timeout_ms",
                        type="int",
                        range=(100, 30000),
                        default=step.timeout_ms,
                    )
                )

            # Strategy configs
            if step.strategy and isinstance(step.strategy, dict):
                config = step.strategy.get("config", {})
                strategy_type = step.strategy.get("type")

                if strategy_type == "cached" and "ttl_seconds" in config:
                    dimensions.append(
                        SearchDimension(
                            name=f"{step_prefix}strategy.ttl_seconds",
                            type="int",
                            range=(60, 7200),
                            default=config["ttl_seconds"],
                        )
                    )

                elif strategy_type == "retry":
                    if "max_attempts" in config:
                        dimensions.append(
                            SearchDimension(
                                name=f"{step_prefix}strategy.max_attempts",
                                type="int",
                                range=(1, 10),
                                default=config["max_attempts"],
                            )
                        )

                    if "initial_delay_ms" in config:
                        dimensions.append(
                            SearchDimension(
                                name=f"{step_prefix}strategy.initial_delay_ms",
                                type="int",
                                range=(100, 5000),
                                default=config["initial_delay_ms"],
                            )
                        )

        # Global constraints
        if plan.constraints:
            if "max_latency_ms" in plan.constraints:
                dimensions.append(
                    SearchDimension(
                        name="constraints.max_latency_ms",
                        type="int",
                        range=(1000, 60000),
                        default=plan.constraints["max_latency_ms"],
                    )
                )

        return dimensions

    def _build_skopt_space(self, dimensions: List[SearchDimension]):
        """Convert search dimensions to skopt space.

        Args:
            dimensions: List of search dimensions

        Returns:
            List of skopt dimension objects
        """
        space = []

        for dim in dimensions:
            if dim.type == "int":
                space.append(Integer(dim.range[0], dim.range[1], name=dim.name))
            elif dim.type == "float":
                space.append(Real(dim.range[0], dim.range[1], name=dim.name))
            elif dim.type == "categorical":
                space.append(Categorical(dim.categories, name=dim.name))

        return space

    def suggest_parameters(self, plan: Plan) -> Dict[str, Any]:
        """Suggest next hyperparameters to try.

        Args:
            plan: Plan to optimize

        Returns:
            Dictionary of parameter suggestions
        """
        if not self.search_space:
            self.search_space = self.extract_search_space(plan)
            self.skopt_space = self._build_skopt_space(self.search_space)

        if len(self.trials) < 5:
            # Random exploration phase
            return self._random_sample(self.search_space)
        else:
            # Use Bayesian optimization
            return self._bayesian_suggest()

    def _random_sample(self, dimensions: List[SearchDimension]) -> Dict[str, Any]:
        """Generate random parameter sample.

        Args:
            dimensions: Search dimensions

        Returns:
            Random parameter values
        """
        params = {}

        for dim in dimensions:
            if dim.type == "int":
                params[dim.name] = np.random.randint(dim.range[0], dim.range[1] + 1)
            elif dim.type == "float":
                params[dim.name] = np.random.uniform(dim.range[0], dim.range[1])
            elif dim.type == "categorical":
                params[dim.name] = np.random.choice(dim.categories)

        return params

    def _bayesian_suggest(self) -> Dict[str, Any]:
        """Use Bayesian optimization to suggest parameters.

        Returns:
            Suggested parameter values
        """
        # Extract X (parameters) and y (scores) from trials
        X = []
        y = []

        for trial in self.trials:
            param_vector = []
            for dim in self.search_space:
                param_vector.append(trial.params.get(dim.name, dim.default))
            X.append(param_vector)
            y.append(-trial.score)  # Negate for minimization

        # Create objective function for next suggestion
        def objective_func(*args):
            # Just return a placeholder - we use tell() instead
            return 0

        # Run one iteration to get next suggestion
        result = gp_minimize(
            objective_func,
            self.skopt_space,
            n_calls=1,
            random_state=self.random_state,
            x0=X,
            y0=y,
            n_initial_points=0,
        )

        # Convert result to parameter dict
        next_params = {}
        for i, dim in enumerate(self.search_space):
            next_params[dim.name] = result.x[i]

        return next_params

    def record_trial(
        self,
        params: Dict[str, Any],
        metrics: ExecutionMetrics,
        quality: QualitySignals,
    ) -> Trial:
        """Record trial outcome.

        Args:
            params: Parameters used
            metrics: Execution metrics
            quality: Quality signals

        Returns:
            Trial record
        """
        score = self.objective.score(metrics, quality)

        trial = Trial(
            trial_id=len(self.trials),
            params=params,
            score=score,
            metrics=metrics,
            quality=quality,
        )

        self.trials.append(trial)
        return trial

    def get_best_trial(self) -> Optional[Trial]:
        """Get best trial so far.

        Returns:
            Trial with highest score or None
        """
        if not self.trials:
            return None

        return max(self.trials, key=lambda t: t.score)

    def apply_parameters(self, plan: Plan, params: Dict[str, Any]) -> Plan:
        """Apply parameters to plan.

        Args:
            plan: Base plan
            params: Parameters to apply

        Returns:
            Modified plan
        """
        # Deep copy plan
        plan_dict = json.loads(plan.model_dump_json())

        # Apply parameters
        for param_name, value in params.items():
            self._set_nested_value(plan_dict, param_name, value)

        # Create new plan from modified dict
        return Plan(**plan_dict)

    def _set_nested_value(self, data: Dict, path: str, value: Any) -> None:
        """Set nested dictionary value using dotted path.

        Args:
            data: Dictionary to modify
            path: Dotted path like 'step1.strategy.ttl_seconds'
            value: Value to set
        """
        parts = path.split(".")
        current = data

        # Navigate to parent
        for part in parts[:-1]:
            # Handle step references
            if part in [s["id"] for s in data.get("steps", [])]:
                # Find step by id
                for step in data["steps"]:
                    if step["id"] == part:
                        current = step
                        break
            elif part in current:
                current = current[part]
            else:
                current[part] = {}
                current = current[part]

        # Set final value
        current[parts[-1]] = value

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results.

        Returns:
            Summary statistics
        """
        if not self.trials:
            return {"trials": 0, "best_score": None}

        best = self.get_best_trial()
        scores = [t.score for t in self.trials]

        return {
            "trials": len(self.trials),
            "best_score": best.score,
            "best_params": best.params,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "improvement": (best.score - scores[0]) / abs(scores[0]) if scores[0] != 0 else 0,
        }
