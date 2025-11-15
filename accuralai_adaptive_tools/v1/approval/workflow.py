"""Approval workflow for tool proposals."""

from typing import Optional

from ...contracts.models import ApprovalState, RiskLevel, SystemType, ToolProposal
from ...registry import UnifiedRegistry


class ApprovalWorkflow:
    """Manages approval process for tool proposals."""

    def __init__(
        self,
        registry: UnifiedRegistry,
        auto_approve_low_risk: bool = False,
        min_executions_for_auto_approve: int = 100,
    ):
        """Initialize approval workflow.

        Args:
            registry: Unified registry for storing proposals
            auto_approve_low_risk: Automatically approve LOW risk proposals after testing
            min_executions_for_auto_approve: Minimum successful executions before auto-approve
        """
        self.registry = registry
        self.auto_approve_low_risk = auto_approve_low_risk
        self.min_executions_for_auto_approve = min_executions_for_auto_approve

    async def submit_proposal(self, proposal: ToolProposal) -> str:
        """Submit proposal for approval.

        Args:
            proposal: Tool proposal to submit

        Returns:
            Proposal ID
        """
        # Store proposal in registry
        await self.registry.initialize()

        # Convert proposal to registry format (not activated yet)
        proposal_id = await self.registry.register_tool(
            name=proposal.name,
            source_code=proposal.source_code,
            function_schema=proposal.function_schema,
            system=SystemType.V1,  # V1-generated
            version="1",
            created_by=proposal.created_by,
            approval_state=ApprovalState.PENDING,
            metadata={
                "proposal_id": proposal.proposal_id,
                "rationale": proposal.rationale,
                "evidence": proposal.evidence,
                "safety_analysis": proposal.safety_analysis.dict() if proposal.safety_analysis else None,
            },
        )

        # Check if auto-approval applies
        if self._should_auto_approve(proposal):
            await self.approve_proposal(proposal.proposal_id, approved_by="system:auto")

        return proposal.proposal_id

    async def approve_proposal(self, proposal_id: str, approved_by: str = "human") -> bool:
        """Approve a proposal.

        Args:
            proposal_id: Proposal ID to approve
            approved_by: Who approved it

        Returns:
            True if successful
        """
        await self.registry.initialize()

        # Find proposal by metadata
        tool_id = await self._find_tool_by_proposal_id(proposal_id)
        if not tool_id:
            return False

        # Update state to APPROVED (not yet ACTIVE - that requires deployment)
        await self.registry.update_approval_state(tool_id, ApprovalState.APPROVED)

        return True

    async def reject_proposal(self, proposal_id: str, reason: str, rejected_by: str = "human") -> bool:
        """Reject a proposal.

        Args:
            proposal_id: Proposal ID to reject
            reason: Rejection reason
            rejected_by: Who rejected it

        Returns:
            True if successful
        """
        await self.registry.initialize()

        tool_id = await self._find_tool_by_proposal_id(proposal_id)
        if not tool_id:
            return False

        await self.registry.update_approval_state(tool_id, ApprovalState.REJECTED)

        # Could store rejection reason in metadata here
        return True

    async def activate_tool(self, proposal_id: str) -> bool:
        """Activate an approved tool for production use.

        Args:
            proposal_id: Proposal ID to activate

        Returns:
            True if successful
        """
        await self.registry.initialize()

        tool_id = await self._find_tool_by_proposal_id(proposal_id)
        if not tool_id:
            return False

        # Get current state
        tool = await self.registry.get_tool(tool_id.split("_v")[0])
        if not tool or tool.get("approval_state") != ApprovalState.APPROVED.value:
            return False

        # Activate
        await self.registry.update_approval_state(tool_id, ApprovalState.ACTIVE)

        # Tool is now available in registry for V2 to use
        await self.registry.sync()

        return True

    async def get_pending_proposals(self) -> list[dict]:
        """Get all pending proposals.

        Returns:
            List of pending proposal dictionaries
        """
        await self.registry.initialize()

        # Query registry for PENDING tools
        # This would require additional registry methods in production
        # For now, return empty list as placeholder
        return []

    def _should_auto_approve(self, proposal: ToolProposal) -> bool:
        """Determine if proposal should be auto-approved.

        Args:
            proposal: Tool proposal

        Returns:
            True if should auto-approve
        """
        if not self.auto_approve_low_risk:
            return False

        if not proposal.safety_analysis:
            return False

        # Only auto-approve LOW risk
        if proposal.safety_analysis.risk_level != RiskLevel.LOW:
            return False

        # Must pass all safety checks
        if not proposal.safety_analysis.passed:
            return False

        return True

    def _calculate_required_reviewers(self, proposal: ToolProposal) -> int:
        """Calculate number of required reviewers based on risk.

        Args:
            proposal: Tool proposal

        Returns:
            Number of reviewers required
        """
        if not proposal.safety_analysis:
            return 2

        risk_level = proposal.safety_analysis.risk_level

        if risk_level == RiskLevel.LOW:
            return 1
        elif risk_level == RiskLevel.MEDIUM:
            return 1
        elif risk_level == RiskLevel.HIGH:
            return 2
        elif risk_level == RiskLevel.CRITICAL:
            return 3

        return 2

    async def _find_tool_by_proposal_id(self, proposal_id: str) -> Optional[str]:
        """Find tool ID by proposal ID in metadata.

        Args:
            proposal_id: Proposal ID to search for

        Returns:
            Tool ID if found, None otherwise
        """
        # This would require a registry query method in production
        # For now, return None as placeholder
        # In real implementation:
        # tools = await self.registry.query_by_metadata({"proposal_id": proposal_id})
        # return tools[0]["id"] if tools else None
        return None
