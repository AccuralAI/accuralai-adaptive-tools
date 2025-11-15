"""Code generation templates for tool synthesis."""

from typing import Any, Dict, List


class ToolTemplate:
    """Templates for generating tool code."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for LLM code generation."""
        return """You are an expert Python developer specializing in creating CLI tools for the AccuralAI ecosystem.

Your task is to generate clean, safe, well-documented Python functions that implement CLI tools.

Requirements:
1. Use async/await syntax (async def)
2. Follow the signature: async def tool_name(state: SessionState, args: List[str], call_args: Dict) -> ToolResult
3. Import only from standard library or explicitly allowed packages
4. Include comprehensive error handling
5. Return ToolResult with status, message, and optional data
6. Add docstrings explaining parameters and return values
7. Never use eval(), exec(), or subprocess without explicit need
8. Validate inputs before processing
9. Provide helpful error messages

Allowed imports:
- Standard library (pathlib, json, csv, datetime, etc.)
- typing
- accuralai_core.cli.tools.models (for ToolResult, SessionState)

Output only the function code, no explanations or markdown."""

    @staticmethod
    def get_sequence_generation_prompt(pattern: Dict[str, Any]) -> str:
        """Generate prompt for sequence pattern.

        Args:
            pattern: Pattern dictionary from SequenceAnalyzer

        Returns:
            Generation prompt for LLM
        """
        sequence = pattern["sequence"]
        suggested_name = pattern["suggested_name"]
        description = pattern["description"]
        tool_metrics = pattern.get("tool_metrics", {})

        # Build metrics context
        metrics_str = ""
        for tool, metrics in tool_metrics.items():
            metrics_str += f"\n- {tool}: avg {metrics['avg_latency_ms']:.1f}ms, {metrics['count']} executions"

        prompt = f"""Generate a Python async function that combines these tools into a single operation:

Tool Sequence:
{' → '.join(sequence)}

Function Name: {suggested_name}

Description: {description}

Performance Context:{metrics_str}

The function should:
1. Combine the functionality of all tools in the sequence
2. Eliminate redundant operations (e.g., reading same file multiple times)
3. Use efficient data structures to minimize overhead
4. Handle errors gracefully
5. Accept command-line args similar to the original tools
6. Return ToolResult with comprehensive status

Example usage:
/tool run {suggested_name} <args>

Generate the complete async function implementation:"""

        return prompt

    @staticmethod
    def get_failure_improvement_prompt(pattern: Dict[str, Any]) -> str:
        """Generate prompt for failure improvement pattern.

        Args:
            pattern: Pattern dictionary from FailureAnalyzer

        Returns:
            Generation prompt for LLM
        """
        tool_id = pattern["tool_id"]
        suggested_name = pattern["suggested_name"]
        failure_rate = pattern["failure_rate"]
        most_common_error = pattern["most_common_error"]
        suggestions = pattern["suggestions"]

        suggestions_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

        prompt = f"""Generate an improved version of an existing tool that has reliability issues:

Original Tool: {tool_id}
Failure Rate: {failure_rate * 100:.1f}%
Most Common Error: {most_common_error}

Improvement Suggestions:
{suggestions_str}

New Function Name: {suggested_name}

The function should:
1. Implement all suggested improvements
2. Add comprehensive error handling for {most_common_error}
3. Provide clear, actionable error messages
4. Validate inputs before processing
5. Include fallback strategies where appropriate
6. Maintain backward-compatible interface if possible
7. Add helpful docstring explaining improvements

Generate the complete async function implementation:"""

        return prompt

    @staticmethod
    def get_missing_capability_prompt(pattern: Dict[str, Any]) -> str:
        """Generate prompt for missing capability pattern.

        Args:
            pattern: Pattern dictionary from CapabilityDetector

        Returns:
            Generation prompt for LLM
        """
        tool_name = pattern["tool_name"]
        inferred_description = pattern["inferred_description"]
        request_count = pattern["request_count"]

        prompt = f"""Generate a new tool that implements requested functionality:

Tool Name: {tool_name}
Inferred Purpose: {inferred_description}
Times Requested: {request_count}

The function should:
1. Implement the most likely interpretation of the tool name
2. Follow common CLI tool patterns
3. Accept flexible arguments
4. Provide helpful usage messages
5. Handle edge cases and errors gracefully
6. Include comprehensive docstring
7. Return structured data when appropriate

If the tool name is ambiguous, implement the most common/useful interpretation.

Generate the complete async function implementation:"""

        return prompt

    @staticmethod
    def get_function_schema_prompt(source_code: str) -> str:
        """Generate prompt to extract function schema.

        Args:
            source_code: Generated function code

        Returns:
            Prompt to extract JSON schema
        """
        prompt = f"""Analyze this Python function and generate a JSON schema describing its interface:

{source_code}

Generate a JSON object with:
{{
    "name": "function_name",
    "description": "brief description",
    "args": [
        {{"name": "arg1", "type": "str", "description": "...", "required": true}},
        ...
    ],
    "returns": {{"type": "ToolResult", "description": "..."}}
}}

Output only the JSON, no explanations:"""

        return prompt

    @staticmethod
    def get_test_cases_prompt(source_code: str, pattern: Dict[str, Any]) -> str:
        """Generate prompt to create test cases.

        Args:
            source_code: Generated function code
            pattern: Original pattern

        Returns:
            Prompt to generate test cases
        """
        prompt = f"""Generate test cases for this function:

{source_code}

Pattern Context:
{pattern.get('description', 'N/A')}

Generate 3-5 test cases covering:
1. Happy path (normal usage)
2. Error cases (invalid inputs, missing files, etc.)
3. Edge cases (empty inputs, large data, special characters)

Output as JSON array:
[
    {{
        "name": "test_normal_usage",
        "args": ["arg1", "arg2"],
        "call_args": {{}},
        "expected_status": "success",
        "description": "Tests normal operation"
    }},
    ...
]

Output only the JSON array:"""

        return prompt
