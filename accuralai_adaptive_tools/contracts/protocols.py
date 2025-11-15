"""Protocols for adaptive tools system components."""

from typing import Any, Dict, List, Protocol

from .models import (
    ExecutionResult,
    OptimizationResult,
    Plan,
    SafetyReport,
    SandboxResult,
    TelemetryEvent,
    ToolProposal,
)


# ============================================================================
# V1 Protocols (Tool Generation)
# ============================================================================


class PatternDetector(Protocol):
    """Detects patterns in telemetry data."""

    async def analyze(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Analyze events and return detected patterns."""
        ...

    async def detect_repeated_sequence(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect repeated tool sequences."""
        ...

    async def detect_high_failure_rate(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect tools with high failure rates."""
        ...

    async def detect_missing_capability(self, events: List[TelemetryEvent]) -> List[Dict[str, Any]]:
        """Detect requests for missing functionality."""
        ...


class CodeGenerator(Protocol):
    """Generates tool code from patterns."""

    async def generate(self, pattern: Dict[str, Any]) -> ToolProposal:
        """Generate tool code from detected pattern."""
        ...

    async def improve_existing(self, tool_name: str, issues: List[str]) -> ToolProposal:
        """Generate improved version of existing tool."""
        ...


class SafetyValidator(Protocol):
    """Validates generated code for safety."""

    async def validate(self, source_code: str) -> SafetyReport:
        """Analyze code and return safety report."""
        ...

    def check_forbidden_patterns(self, source_code: str) -> List[str]:
        """Check for forbidden patterns (eval, exec, etc.)."""
        ...

    def check_imports(self, source_code: str) -> List[str]:
        """Check for suspicious imports."""
        ...


class SandboxExecutor(Protocol):
    """Executes code in isolated sandbox."""

    async def test(self, source_code: str, test_cases: List[Dict[str, Any]]) -> SandboxResult:
        """Execute code with test cases in sandbox."""
        ...

    async def cleanup(self):
        """Clean up sandbox resources."""
        ...


# ============================================================================
# V2 Protocols (Plan Optimization)
# ============================================================================


class PlanParser(Protocol):
    """Parses PlanLang YAML/JSON into Plan objects."""

    def parse(self, yaml_content: str) -> Plan:
        """Parse YAML/JSON content into Plan."""
        ...

    def to_yaml(self, plan: Plan) -> str:
        """Convert Plan to YAML."""
        ...


class PlanValidator(Protocol):
    """Validates plans for correctness."""

    async def validate(self, plan: Plan) -> List[str]:
        """Validate plan and return list of errors."""
        ...

    def check_tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists in registry."""
        ...

    def check_variable_dependencies(self, plan: Plan) -> List[str]:
        """Check that all variable dependencies are satisfied."""
        ...


class PlanExecutor(Protocol):
    """Executes plans with strategy support."""

    async def execute(self, plan: Plan, inputs: Dict[str, Any]) -> ExecutionResult:
        """Execute plan with given inputs."""
        ...

    async def execute_step(self, step: Any, context: Dict[str, Any]) -> Any:
        """Execute single plan step."""
        ...


class Optimizer(Protocol):
    """Optimizes plan hyperparameters."""

    async def optimize(self, plan: Plan, trials: int = 50) -> OptimizationResult:
        """Optimize plan hyperparameters using Bayesian optimization."""
        ...

    async def suggest_params(self, plan: Plan) -> Dict[str, Any]:
        """Suggest next set of parameters to try."""
        ...

    async def update(self, params: Dict[str, Any], score: float):
        """Update optimizer with evaluation result."""
        ...


# ============================================================================
# Shared Protocols
# ============================================================================


class TelemetryCollector(Protocol):
    """Collects and stores telemetry events."""

    async def record(self, event: TelemetryEvent):
        """Record telemetry event."""
        ...

    async def query(self, filter: Dict[str, Any]) -> List[TelemetryEvent]:
        """Query historical telemetry."""
        ...

    async def get_recent(self, hours: int = 24) -> List[TelemetryEvent]:
        """Get recent events."""
        ...


class Registry(Protocol):
    """Registry for tools and plans."""

    def register_tool(self, name: str, spec: Any, system: str):
        """Register tool."""
        ...

    def register_plan(self, name: str, plan: Plan, system: str):
        """Register plan."""
        ...

    def get_tool(self, name: str) -> Any:
        """Get tool by name."""
        ...

    def get_plan(self, name: str) -> Plan:
        """Get plan by name."""
        ...

    def list_tools(self) -> List[str]:
        """List all tools."""
        ...

    def list_plans(self) -> List[str]:
        """List all plans."""
        ...
