"""V2 Execution: Plan execution with strategy support."""

from .context import ExecutionContext
from .executor import PlanExecutor
from .variable_resolver import VariableResolver

# Alias for compatibility
PlanExecutorImpl = PlanExecutor

__all__ = [
    "ExecutionContext",
    "PlanExecutor",
    "PlanExecutorImpl",
    "VariableResolver",
]
