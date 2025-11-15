"""LLM-powered code generator for tools."""

import uuid
from datetime import datetime
from typing import Any, Dict

from ...contracts.models import RiskLevel, SafetyReport, ToolProposal
from ...contracts.protocols import CodeGenerator
from .llm_client import LLMClient
from .templates import ToolTemplate


class CodeGeneratorImpl(CodeGenerator):
    """Generates tool code from patterns using LLM."""

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize code generator.

        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client or LLMClient()
        self.template = ToolTemplate()

    async def generate(self, pattern: Dict[str, Any]) -> ToolProposal:
        """Generate tool code from detected pattern.

        Args:
            pattern: Pattern dictionary from pattern detector

        Returns:
            ToolProposal with generated code

        Raises:
            RuntimeError: If generation fails
        """
        # Initialize LLM client
        await self.llm_client.initialize()

        # Get appropriate prompt based on pattern type
        pattern_type = pattern["pattern_type"]

        if pattern_type == "repeated_sequence":
            user_prompt = self.template.get_sequence_generation_prompt(pattern)
        elif pattern_type == "high_failure_rate":
            user_prompt = self.template.get_failure_improvement_prompt(pattern)
        elif pattern_type == "missing_capability":
            user_prompt = self.template.get_missing_capability_prompt(pattern)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        # Generate code
        system_prompt = self.template.get_system_prompt()
        source_code = await self.llm_client.generate_code(system_prompt, user_prompt)

        # Extract function schema
        function_schema = await self._extract_schema(source_code)

        # Generate test cases
        test_cases = await self._generate_test_cases(source_code, pattern)

        # Create rationale
        rationale = self._create_rationale(pattern)

        # Extract evidence
        evidence = self._extract_evidence(pattern)

        # Create default safety report (will be updated by safety validator)
        default_safety_report = SafetyReport(
            risk_level=RiskLevel.LOW,
            forbidden_patterns=[],
            suspicious_imports=[],
            validation_errors=[],
            recommendations=[],
            passed=True,
        )

        # Create proposal
        proposal = ToolProposal(
            proposal_id=f"prop_v1_{uuid.uuid4().hex[:8]}",
            name=pattern["suggested_name"],
            description=pattern["description"],
            source_code=source_code,
            function_schema=function_schema,
            rationale=rationale,
            evidence=evidence,
            safety_analysis=default_safety_report,  # Will be updated by safety validator
            created_at=datetime.utcnow(),
            created_by=f"v1:llm:{self.llm_client.model}",
        )

        # Store test cases in metadata
        proposal.function_schema["test_cases"] = test_cases

        return proposal

    async def improve_existing(self, tool_name: str, issues: list[str]) -> ToolProposal:
        """Generate improved version of existing tool.

        Args:
            tool_name: Name of tool to improve
            issues: List of issues to address

        Returns:
            ToolProposal with improved code
        """
        # Create synthetic pattern for improvement
        pattern = {
            "pattern_type": "high_failure_rate",
            "tool_id": tool_name,
            "suggested_name": f"{tool_name}_improved",
            "description": f"Improved version of {tool_name}",
            "suggestions": issues,
            "failure_rate": 0.0,  # Placeholder
            "most_common_error": "Various",
        }

        return await self.generate(pattern)

    async def _extract_schema(self, source_code: str) -> Dict[str, Any]:
        """Extract function schema from generated code.

        Args:
            source_code: Generated function code

        Returns:
            Function schema dictionary
        """
        try:
            schema_prompt = self.template.get_function_schema_prompt(source_code)
            schema = await self.llm_client.extract_json(
                system_prompt="Extract function schema as JSON.",
                user_prompt=schema_prompt
            )
            return schema
        except Exception as e:
            # Fallback to basic schema
            return {
                "name": "unknown",
                "description": "Generated tool",
                "args": [],
                "returns": {"type": "ToolResult"}
            }

    async def _generate_test_cases(
        self,
        source_code: str,
        pattern: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        """Generate test cases for the code.

        Args:
            source_code: Generated function code
            pattern: Original pattern

        Returns:
            List of test case dictionaries
        """
        try:
            test_prompt = self.template.get_test_cases_prompt(source_code, pattern)
            test_cases = await self.llm_client.extract_json(
                system_prompt="Generate test cases as JSON array.",
                user_prompt=test_prompt
            )

            # Handle both dict with "test_cases" key and direct list
            if isinstance(test_cases, dict) and "test_cases" in test_cases:
                return test_cases["test_cases"]
            elif isinstance(test_cases, list):
                return test_cases
            else:
                return []

        except Exception as e:
            # Fallback to basic test
            return [{
                "name": "test_basic",
                "args": [],
                "call_args": {},
                "expected_status": "success",
                "description": "Basic functionality test"
            }]

    def _create_rationale(self, pattern: Dict[str, Any]) -> str:
        """Create human-readable rationale for proposal.

        Args:
            pattern: Pattern dictionary

        Returns:
            Rationale string
        """
        pattern_type = pattern["pattern_type"]

        if pattern_type == "repeated_sequence":
            frequency = pattern.get("frequency", 0)
            improvement = pattern.get("estimated_improvement_pct", 0)
            return (
                f"Detected repeated sequence executed {frequency} times. "
                f"Combining into single tool could improve efficiency by ~{improvement:.0f}%. "
                f"Reduces context switching and eliminates redundant operations."
            )

        elif pattern_type == "high_failure_rate":
            failure_rate = pattern.get("failure_rate", 0) * 100
            error_type = pattern.get("most_common_error", "errors")
            return (
                f"Tool has {failure_rate:.1f}% failure rate, primarily due to {error_type}. "
                f"Proposed improvements address root causes and add better error handling."
            )

        elif pattern_type == "missing_capability":
            request_count = pattern.get("request_count", 0)
            return (
                f"Requested {request_count} times but not found. "
                f"Users clearly need this functionality. "
                f"Implementation inferred from tool name and usage context."
            )

        return "Generated based on detected pattern."

    def _extract_evidence(self, pattern: Dict[str, Any]) -> list[str]:
        """Extract evidence trace IDs from pattern.

        Args:
            pattern: Pattern dictionary

        Returns:
            List of event IDs as evidence
        """
        # Try different evidence fields based on pattern type
        evidence_fields = ["evidence_events", "failed_events", "evidence_count"]

        for field in evidence_fields:
            if field in pattern:
                value = pattern[field]
                if isinstance(value, list):
                    return value[:10]  # First 10
                elif isinstance(value, int):
                    return [f"pattern_{field}_{value}"]

        return []
