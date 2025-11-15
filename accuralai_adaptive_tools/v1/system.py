"""V1 System: Complete pattern-based tool generation pipeline."""

from typing import Any, Dict, List, Optional

from ..contracts.models import EventType, SafetyReport, TelemetryEvent, ToolProposal
from ..registry import UnifiedRegistry
from ..telemetry import SharedTelemetry
from .approval import ApprovalWorkflow
from .sandbox import SafetyValidatorImpl, SandboxExecutorImpl
from .synthesis import CodeGeneratorImpl, LLMClient
from .telemetry import PatternDetectorImpl


class V1System:
    """Complete V1 system for pattern-based tool generation."""

    def __init__(
        self,
        telemetry: SharedTelemetry,
        registry: UnifiedRegistry,
        llm_backend_id: str = "google",
        llm_model: str = "gemini-2.5-flash-lite",
        llm_temperature: float = 0.2,
        auto_approve_low_risk: bool = False,
    ):
        """Initialize V1 system.

        Args:
            telemetry: Shared telemetry system
            registry: Unified registry
            llm_backend_id: LLM backend for code generation
            llm_model: LLM model name
            llm_temperature: LLM sampling temperature
            auto_approve_low_risk: Auto-approve LOW risk proposals
        """
        self.telemetry = telemetry
        self.registry = registry

        # Initialize components
        self.pattern_detector = PatternDetectorImpl()
        self.llm_client = LLMClient(
            backend_id=llm_backend_id,
            model=llm_model,
            temperature=llm_temperature
        )
        self.code_generator = CodeGeneratorImpl(self.llm_client)
        self.safety_validator = SafetyValidatorImpl()
        self.sandbox_executor = SandboxExecutorImpl()
        self.approval_workflow = ApprovalWorkflow(
            registry=registry,
            auto_approve_low_risk=auto_approve_low_risk
        )

    async def analyze_and_propose(self, hours: int = 24) -> List[ToolProposal]:
        """Analyze recent telemetry and generate proposals.

        Args:
            hours: Hours of telemetry to analyze

        Returns:
            List of tool proposals
        """
        # Get recent events
        events = await self.telemetry.get_recent(hours=hours)

        # Detect patterns
        patterns = await self.pattern_detector.analyze(events)

        if not patterns:
            return []

        # Generate proposals for top patterns
        proposals = []
        for pattern in patterns[:5]:  # Top 5 patterns
            try:
                proposal = await self._create_proposal_from_pattern(pattern)
                if proposal:
                    proposals.append(proposal)
            except Exception as e:
                # Log error but continue with other patterns
                print(f"Failed to create proposal from pattern: {e}")
                continue

        return proposals

    async def _create_proposal_from_pattern(self, pattern: Dict[str, Any]) -> ToolProposal | None:
        """Create complete proposal from pattern (generate, validate, test).

        Args:
            pattern: Pattern dictionary

        Returns:
            ToolProposal if successful, None otherwise
        """
        # Generate code
        proposal = await self.code_generator.generate(pattern)

        # Safety validation
        safety_report = await self.safety_validator.validate(proposal.source_code)
        proposal.safety_analysis = safety_report

        # If critical risk, reject immediately
        if safety_report.risk_level.value == "CRITICAL":
            return None

        # If validation failed, reject
        if not safety_report.passed:
            return None

        # Sandbox testing
        test_cases = proposal.function_schema.get("test_cases", [])
        sandbox_result = await self.sandbox_executor.test(
            proposal.source_code,
            test_cases
        )

        # Store sandbox results in proposal
        proposal.function_schema["sandbox_result"] = {
            "passed": sandbox_result.passed,
            "test_cases_run": sandbox_result.test_cases_run,
            "test_cases_passed": sandbox_result.test_cases_passed,
            "errors": sandbox_result.errors,
        }

        # If sandbox failed, still return proposal but mark for review
        if not sandbox_result.passed:
            proposal.function_schema["requires_review"] = True

        # Submit to approval workflow
        await self.approval_workflow.submit_proposal(proposal)

        return proposal

    async def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal.

        Args:
            proposal_id: Proposal ID

        Returns:
            True if successful
        """
        success = await self.approval_workflow.approve_proposal(proposal_id)
        if success:
            # Activate the tool
            await self.approval_workflow.activate_tool(proposal_id)
        return success

    async def reject_proposal(self, proposal_id: str, reason: str) -> bool:
        """Reject a proposal.

        Args:
            proposal_id: Proposal ID
            reason: Rejection reason

        Returns:
            True if successful
        """
        return await self.approval_workflow.reject_proposal(proposal_id, reason)

    async def get_pending_proposals(self) -> List[dict]:
        """Get all pending proposals.

        Returns:
            List of pending proposals
        """
        return await self.approval_workflow.get_pending_proposals()

    async def improve_tool(self, tool_name: str, issues: List[str]) -> ToolProposal | None:
        """Generate improved version of existing tool.

        Args:
            tool_name: Name of tool to improve
            issues: List of issues to address

        Returns:
            ToolProposal if successful, None otherwise
        """
        # Generate improved version
        proposal = await self.code_generator.improve_existing(tool_name, issues)

        # Validate and test
        safety_report = await self.safety_validator.validate(proposal.source_code)
        proposal.safety_analysis = safety_report

        if not safety_report.passed:
            return None

        test_cases = proposal.function_schema.get("test_cases", [])
        sandbox_result = await self.sandbox_executor.test(
            proposal.source_code,
            test_cases
        )

        proposal.function_schema["sandbox_result"] = {
            "passed": sandbox_result.passed,
            "test_cases_run": sandbox_result.test_cases_run,
            "test_cases_passed": sandbox_result.test_cases_passed,
        }

        # Submit to approval
        await self.approval_workflow.submit_proposal(proposal)

        return proposal

    # Adapter methods for V3Coordinator compatibility
    async def analyze_pattern(self, event: TelemetryEvent) -> Optional[Dict[str, Any]]:
        """Analyze pattern from event (adapter for V3Coordinator).
        
        Args:
            event: Telemetry event
            
        Returns:
            Pattern dictionary or None
        """
        # If this is a TOOL_SEQUENCE event, create pattern directly from it
        if event.event_type == EventType.TOOL_SEQUENCE:
            sequence_count = event.event_data.get("count", 0)
            pattern_text = event.event_data.get("pattern", "")
            tool_calls = event.event_data.get("tool_calls", 0)
            
            # Generate a suggested tool name from the pattern
            # Extract key words from pattern text and item_id
            name_parts = []
            if event.item_id:
                # Use item_id as base, convert to snake_case
                base_name = event.item_id.lower().replace(" ", "_").replace("-", "_")
                name_parts.append(base_name)
            
            # Extract action words from pattern text
            if "read" in pattern_text.lower():
                name_parts.append("read")
            if "write" in pattern_text.lower() or "save" in pattern_text.lower():
                name_parts.append("write")
            if "parse" in pattern_text.lower() or "process" in pattern_text.lower():
                name_parts.append("process")
            if "convert" in pattern_text.lower() or "transform" in pattern_text.lower():
                name_parts.append("transform")
            
            # Generate suggested name
            if name_parts:
                suggested_name = "_".join(name_parts[:3]) + "_tool"  # Limit to 3 parts
            else:
                suggested_name = f"auto_generated_tool_{hash(pattern_text) % 10000}"
            
            # Create a pattern from the sequence event
            pattern = {
                "pattern_type": "repeated_sequence",
                "description": f"Repeated pattern detected: {pattern_text[:100]}",
                "sequence": [event.item_id] if event.item_id else [],
                "frequency": sequence_count,
                "item_id": event.item_id,
                "item_type": event.item_type,
                "estimated_improvement_pct": 30.0,  # Estimate 30% improvement
                "tool_calls": tool_calls,
                "pattern_text": pattern_text,
                "suggested_name": suggested_name,
                "tool_metrics": {},  # Empty metrics for now
            }
            return pattern
        
        # For other event types, try to analyze recent events
        # Get recent events to analyze
        events = await self.telemetry.get_recent(hours=1)
        if not events:
            return None
            
        # Detect patterns
        patterns = await self.pattern_detector.analyze(events)
        if patterns:
            return patterns[0]  # Return first pattern
        return None

    async def generate_proposal(self, pattern: Dict[str, Any]) -> ToolProposal:
        """Generate tool proposal from pattern (adapter for V3Coordinator).
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            ToolProposal
        """
        return await self._create_proposal_from_pattern(pattern)

    async def validate_safety(self, proposal: ToolProposal) -> SafetyReport:
        """Validate safety of proposal (adapter for V3Coordinator).
        
        Args:
            proposal: Tool proposal
            
        Returns:
            SafetyReport
        """
        return await self.safety_validator.validate(proposal.source_code)

    async def test_in_sandbox(self, proposal: ToolProposal) -> Any:
        """Test proposal in sandbox (adapter for V3Coordinator).
        
        Args:
            proposal: Tool proposal
            
        Returns:
            Sandbox result
        """
        test_cases = proposal.function_schema.get("test_cases", [])
        return await self.sandbox_executor.test(proposal.source_code, test_cases)

    async def request_approval(self, proposal: ToolProposal) -> None:
        """Request approval for proposal (adapter for V3Coordinator).
        
        Args:
            proposal: Tool proposal
        """
        await self.approval_workflow.submit_proposal(proposal)

    async def refresh_usage_patterns(self) -> None:
        """Refresh usage patterns from telemetry (adapter for V3Coordinator)."""
        # V1 doesn't need to refresh patterns - they're analyzed on-demand
        pass

    async def initialize(self) -> None:
        """Initialize V1 system (adapter for V3Coordinator)."""
        # Already initialized in __init__
        pass

    async def shutdown(self) -> None:
        """Shutdown V1 system (adapter for V3Coordinator)."""
        # Cleanup if needed
        pass
