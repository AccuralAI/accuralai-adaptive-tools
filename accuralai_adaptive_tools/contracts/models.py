"""Data models for adaptive tools system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of telemetry events."""

    TOOL_EXECUTED = "tool_executed"
    TOOL_ERROR = "tool_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_SEQUENCE = "tool_sequence"
    PLAN_EXECUTED = "plan_executed"
    PLAN_ERROR = "plan_error"


class RiskLevel(str, Enum):
    """Risk levels for safety analysis."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ApprovalState(str, Enum):
    """Approval states for proposals."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"


class SystemType(str, Enum):
    """System that created/manages an item."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    BUILTIN = "builtin"


# ============================================================================
# Telemetry Models
# ============================================================================


class TelemetryEvent(BaseModel):
    """Single telemetry event."""

    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType

    # Context
    item_id: Optional[str] = None
    item_type: Optional[str] = None  # 'tool' or 'plan'

    # Metrics
    latency_ms: Optional[float] = None
    cost_cents: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    # Routing
    routed_to_v1: bool = False
    routed_to_v2: bool = False

    # Full event data
    event_data: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# V1 Models (Tool Generation)
# ============================================================================


class SafetyReport(BaseModel):
    """Safety analysis report for generated code."""

    risk_level: RiskLevel
    forbidden_patterns: List[str] = Field(default_factory=list)
    suspicious_imports: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    passed: bool


class ToolProposal(BaseModel):
    """Proposal for a new or improved tool."""

    proposal_id: str
    name: str
    description: str
    source_code: str  # Python function code
    function_schema: Dict[str, Any]  # Function signature
    rationale: str
    evidence: List[str] = Field(default_factory=list)  # Telemetry trace IDs
    safety_analysis: SafetyReport
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "v1"  # 'v1', 'llm:model', or 'human'
    approval_state: ApprovalState = ApprovalState.PENDING


class ToolMetrics(BaseModel):
    """Performance metrics for a tool."""

    tool_name: str
    version: int
    invocations: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_cents: float
    failure_reasons: Dict[str, int] = Field(default_factory=dict)
    used_in_plans: List[str] = Field(default_factory=list)  # Plan IDs


# ============================================================================
# V2 Models (Plan Optimization)
# ============================================================================


class PlanStep(BaseModel):
    """Single step in a plan."""

    id: str
    tool: str
    with_args: Dict[str, Any] = Field(default_factory=dict)
    save_as: str
    depends_on: List[str] = Field(default_factory=list)
    strategy: Optional[Dict[str, Any]] = None
    conditional: Optional[str] = None  # Conditional expression
    timeout_ms: Optional[int] = None  # Step-level timeout
    error_handling: Optional[Dict[str, Any]] = None  # Error handling config


class PlanConstraints(BaseModel):
    """Resource constraints for plan execution."""

    max_latency_ms: Optional[int] = None
    max_cost_cents: Optional[float] = None
    max_steps: Optional[int] = None


class Plan(BaseModel):
    """Execution plan in PlanLang format."""

    name: str
    version: str
    description: str = ""
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[PlanStep]
    constraints: Dict[str, Any] = Field(default_factory=dict)  # Allow Dict for flexibility
    eval_hooks: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanMetrics(BaseModel):
    """Performance metrics for a plan."""

    plan_name: str
    version: str
    executions: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_cents: float
    constraint_violations: int
    optimizer_runs: int
    best_params: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# V3 Models (Coordination)
# ============================================================================


class CompoundGains(BaseModel):
    """Tracking compound gains from V1+V2 interaction."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Source
    v1_tool_id: Optional[str] = None
    v2_plan_id: Optional[str] = None

    # Impact
    baseline_latency_ms: float
    improved_latency_ms: float
    latency_improvement_pct: float

    baseline_cost_cents: float
    improved_cost_cents: float
    cost_improvement_pct: float

    # Cross-system effects
    triggered_v2_optimization: bool = False
    triggered_v1_generation: bool = False
    downstream_improvements: int = 0

    # Compound factor
    individual_gain: float
    compound_gain: float


class V1Status(BaseModel):
    """Status of V1 system."""

    active_tools_generated: int
    pending_proposals: int
    avg_success_rate: float
    total_invocations: int


class V2Status(BaseModel):
    """Status of V2 system."""

    active_plans: int
    optimization_runs: int
    avg_latency_improvement_pct: float
    avg_cost_improvement_pct: float


class V3Status(BaseModel):
    """Overall status of V3 unified system."""

    v1_status: V1Status
    v2_status: V2Status

    # Cross-system metrics
    v1_tools_in_v2_plans: int
    v2_patterns_triggering_v1: int
    compound_gain_factor: float

    # System health
    telemetry_events_collected: int
    avg_time_to_tool_deployment_hours: float
    avg_time_to_optimization_hours: float
    user_intervention_rate: float


# ============================================================================
# Execution Results
# ============================================================================


class SandboxResult(BaseModel):
    """Result from sandbox execution."""

    passed: bool
    test_cases_run: int
    test_cases_passed: int
    errors: List[str] = Field(default_factory=list)
    execution_time_ms: float


class ExecutionResult(BaseModel):
    """Result from plan execution."""

    success: bool
    steps_completed: int
    total_steps: int
    latency_ms: float
    cost_cents: float
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    """Result from optimization process."""

    original_plan: Plan
    optimized_plan: Plan
    improvement_pct: float
    iterations: int
    best_params: Dict[str, Any]
    ab_test_winner: str  # 'original' or 'optimized'


class ExecutionMetrics(BaseModel):
    """Execution metrics for plan/tool runs."""

    latency_ms: float = 0.0
    cost_cents: float = 0.0
    tokens_used: int = 0
    cache_hit: bool = False
    retry_count: int = 0
    success: bool = True


class QualitySignals(BaseModel):
    """Quality signals from execution."""

    validator_scores: Dict[str, float] = Field(default_factory=dict)
    unit_test_pass_rate: float = 1.0
    acceptance_test_pass_rate: float = 1.0
    human_rating: Optional[float] = None
    error_rate: float = 0.0


class ToolSpec(BaseModel):
    """Specification for a tool."""

    name: str
    description: str
    function: Dict[str, Any]  # Function schema
    metadata: Dict[str, Any] = Field(default_factory=dict)
