"""LLM client for code generation."""

import json
from typing import Any, Dict, Optional


class LLMClient:
    """Client for LLM-powered code generation.

    Integrates with AccuralAI core pipeline for backend selection.
    """

    def __init__(
        self,
        backend_id: str = "google",
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ):
        """Initialize LLM client.

        Args:
            backend_id: AccuralAI backend ID
            model: Model name
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.backend_id = backend_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._pipeline = None

    async def initialize(self):
        """Initialize connection to AccuralAI pipeline.

        This will be implemented to integrate with accuralai-core.
        For now, it's a placeholder for the interface.
        """
        # TODO: Import and initialize accuralai-core pipeline
        # from accuralai_core.pipeline import Pipeline
        # self._pipeline = Pipeline(config)
        pass

    async def generate_code(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> str:
        """Generate code using LLM.

        Args:
            system_prompt: System prompt with instructions
            user_prompt: User prompt with specific task
            max_retries: Maximum retry attempts on failure

        Returns:
            Generated code string

        Raises:
            RuntimeError: If generation fails after retries
        """
        for attempt in range(max_retries):
            try:
                # TODO: Use accuralai-core pipeline
                # For now, return mock response for testing
                response = await self._call_llm(system_prompt, user_prompt)

                # Clean response (remove markdown code blocks if present)
                code = self._extract_code(response)

                return code

            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Code generation failed after {max_retries} attempts: {e}")
                # Exponential backoff
                import asyncio
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Code generation failed")

    async def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Extract structured JSON from LLM.

        Args:
            system_prompt: System prompt with instructions
            user_prompt: User prompt with specific task
            max_retries: Maximum retry attempts

        Returns:
            Parsed JSON dictionary

        Raises:
            RuntimeError: If extraction fails after retries
        """
        for attempt in range(max_retries):
            try:
                response = await self._call_llm(system_prompt, user_prompt)

                # Extract JSON from response
                json_str = self._extract_json_string(response)
                data = json.loads(json_str)

                return data

            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"JSON extraction failed after {max_retries} attempts: {e}")
                import asyncio
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("JSON extraction failed")

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM backend.

        Args:
            system_prompt: System instructions
            user_prompt: User prompt

        Returns:
            LLM response text
        """
        # TODO: Implement actual LLM call via accuralai-core
        # For now, return placeholder that indicates structure

        # This is a mock - in real implementation, would call pipeline:
        # result = await self._pipeline.generate(
        #     prompt=user_prompt,
        #     metadata={"system": system_prompt},
        #     parameters={"temperature": self.temperature, "max_tokens": self.max_tokens}
        # )
        # return result.text

        return """```python
async def example_tool(state, args, call_args):
    \"\"\"Example generated tool.\"\"\"
    from accuralai_core.cli.tools.models import ToolResult

    if not args:
        return ToolResult(
            status="error",
            message="Usage: example_tool <input>"
        )

    # Implementation here
    return ToolResult(
        status="success",
        message="Example complete",
        data={"result": "mock"}
    )
```"""

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned code string
        """
        # Remove markdown code blocks
        if "```python" in response:
            # Extract content between ```python and ```
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                code = response[start:end].strip()
            else:
                code = response[start:].strip()
        elif "```" in response:
            # Generic code block
            start = response.find("```") + len("```")
            end = response.find("```", start)
            if end != -1:
                code = response[start:end].strip()
            else:
                code = response[start:].strip()
        else:
            code = response.strip()

        return code

    def _extract_json_string(self, response: str) -> str:
        """Extract JSON string from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            JSON string
        """
        # Try to find JSON object in response
        response = response.strip()

        # Remove markdown if present
        if "```json" in response:
            start = response.find("```json") + len("```json")
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
            else:
                json_str = response[start:].strip()
        elif "```" in response:
            start = response.find("```") + len("```")
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
            else:
                json_str = response[start:].strip()
        else:
            # Try to find { ... } or [ ... ]
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            elif "[" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                json_str = response[start:end]
            else:
                json_str = response

        return json_str
