"""Execution context for plan execution."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ExecutionContext:
    """Context for plan execution tracking."""

    # Plan identification
    plan_name: str
    plan_version: str
    execution_id: str

    # Inputs
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Variables (step outputs)
    variables: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)

    # Metrics
    total_latency_ms: float = 0.0
    total_cost_cents: float = 0.0

    # Step-level metrics
    step_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)

    def add_variable(self, name: str, value: Any):
        """Add variable to context.

        Args:
            name: Variable name
            value: Variable value
        """
        self.variables[name] = value

    def get_variable(self, name: str) -> Any:
        """Get variable from context.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        if name == "inputs":
            return self.inputs

        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not defined")

        return self.variables[name]

    def has_variable(self, name: str) -> bool:
        """Check if variable exists.

        Args:
            name: Variable name

        Returns:
            True if exists
        """
        return name in self.variables or name == "inputs"

    def mark_step_completed(self, step_id: str, latency_ms: float, cost_cents: float = 0.0):
        """Mark step as completed.

        Args:
            step_id: Step identifier
            latency_ms: Step latency
            cost_cents: Step cost
        """
        self.completed_steps.append(step_id)
        self.total_latency_ms += latency_ms
        self.total_cost_cents += cost_cents

        self.step_metrics[step_id] = {
            "latency_ms": latency_ms,
            "cost_cents": cost_cents,
            "success": True,
        }

    def mark_step_failed(self, step_id: str, error: str):
        """Mark step as failed.

        Args:
            step_id: Step identifier
            error: Error message
        """
        self.failed_steps.append(step_id)
        self.errors.append(f"Step '{step_id}': {error}")

        self.step_metrics[step_id] = {
            "success": False,
            "error": error,
        }

    def is_step_completed(self, step_id: str) -> bool:
        """Check if step is completed.

        Args:
            step_id: Step identifier

        Returns:
            True if completed
        """
        return step_id in self.completed_steps

    def get_elapsed_time_ms(self) -> float:
        """Get elapsed execution time.

        Returns:
            Elapsed time in milliseconds
        """
        elapsed = datetime.utcnow() - self.start_time
        return elapsed.total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "plan_name": self.plan_name,
            "plan_version": self.plan_version,
            "execution_id": self.execution_id,
            "inputs": self.inputs,
            "variables": self.variables,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_cents": self.total_cost_cents,
            "step_metrics": self.step_metrics,
            "errors": self.errors,
            "elapsed_time_ms": self.get_elapsed_time_ms(),
        }
