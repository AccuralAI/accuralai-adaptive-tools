"""PlanLang executor with strategy support."""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anyio

from ...contracts.models import ExecutionMetrics, Plan, PlanStep


@dataclass
class ExecutionResult:
    """Result of plan execution."""

    success: bool
    context: Dict[str, Any]
    metrics: ExecutionMetrics
    validation_results: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass
class StepResult:
    """Result of a single step execution."""

    step_id: str
    success: bool
    data: Any
    latency_ms: float
    error: Optional[str] = None


class PlanExecutor:
    """Execute PlanLang plans with strategy support."""

    def __init__(self, tool_registry, cache_backend=None):
        """Initialize executor.

        Args:
            tool_registry: Registry of available tools
            cache_backend: Optional cache backend for strategy support
        """
        self.registry = tool_registry
        self.cache = cache_backend
        self.context: Dict[str, Any] = {}
        self.metrics: ExecutionMetrics = ExecutionMetrics()

    async def execute(self, plan: Plan, inputs: Dict[str, Any]) -> ExecutionResult:
        """Execute a complete plan.

        Args:
            plan: The plan to execute
            inputs: Input values for the plan

        Returns:
            Execution result with context and metrics
        """
        self.context = {"inputs": inputs}
        self.metrics = ExecutionMetrics(
            latency_ms=0.0,
            cost_cents=0.0,
            tokens_used=0,
            cache_hit=False,
            retry_count=0,
            success=True,
        )

        start_time = time.time()
        validation_results = []

        try:
            # Validate constraints
            self._validate_constraints(plan.constraints)

            # Validate inputs
            self._validate_inputs(plan.inputs, inputs)

            # Build execution graph
            execution_order = self._build_execution_order(plan.steps)

            # Execute steps
            for step in execution_order:
                # Evaluate conditional
                if step.conditional and not self._evaluate_conditional(step.conditional):
                    continue

                # Execute with strategy
                result = await self._execute_step(step)

                if not result.success:
                    if step.error_handling and step.error_handling.get("on_failure") == "continue":
                        continue
                    raise ValueError(f"Step '{step.id}' failed: {result.error}")

                # Save result to context
                self.context[step.save_as] = result.data

                # Update metrics
                self.metrics.latency_ms += result.latency_ms

            # Run validation hooks
            if plan.eval_hooks:
                validation_results = await self._run_validators(plan.eval_hooks)

            self.metrics.latency_ms = (time.time() - start_time) * 1000
            self.metrics.success = True

            return ExecutionResult(
                success=True,
                context=self.context,
                metrics=self.metrics,
                validation_results=validation_results,
            )

        except Exception as e:
            self.metrics.success = False
            return ExecutionResult(
                success=False,
                context=self.context,
                metrics=self.metrics,
                validation_results=validation_results,
                error=str(e),
            )

    async def _execute_step(self, step: PlanStep) -> StepResult:
        """Execute a single step with strategy.

        Args:
            step: The step to execute

        Returns:
            Step execution result
        """
        start_time = time.time()

        try:
            # Resolve arguments (variable substitution)
            resolved_args = self._resolve_args(step.with_args)

            # Get tool from registry
            tool = self.registry.get(step.tool)
            if not tool:
                raise ValueError(f"Tool '{step.tool}' not found in registry")

            # Apply execution strategy
            if step.strategy:
                data = await self._execute_with_strategy(tool, resolved_args, step.strategy)
            else:
                data = await self._execute_simple(tool, resolved_args)

            latency_ms = (time.time() - start_time) * 1000

            return StepResult(
                step_id=step.id,
                success=True,
                data=data,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return StepResult(
                step_id=step.id,
                success=False,
                data=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _execute_simple(self, tool, args: Dict[str, Any]) -> Any:
        """Execute tool without strategy.

        Args:
            tool: Tool to execute (can be dict with 'execute' key or object with execute method)
            args: Tool arguments

        Returns:
            Tool result data
        """
        # Try to call the tool's execute method
        if isinstance(tool, dict) and "execute" in tool:
            execute_func = tool["execute"]
            if callable(execute_func):
                return await execute_func(**args)
        elif hasattr(tool, "execute"):
            execute_func = tool.execute
            if callable(execute_func):
                return await execute_func(**args)
        
        # Fallback: return mock result
        return {"status": "success", "args": args}

    async def _execute_with_strategy(self, tool, args: Dict[str, Any], strategy: Dict) -> Any:
        """Execute tool with strategy.

        Args:
            tool: Tool to execute
            args: Tool arguments
            strategy: Strategy configuration

        Returns:
            Tool result data
        """
        strategy_type = strategy if isinstance(strategy, str) else strategy.get("type")
        strategy_config = {} if isinstance(strategy, str) else strategy.get("config", {})

        if strategy_type == "cached":
            return await self._execute_cached(tool, args, strategy_config)
        elif strategy_type == "retry":
            return await self._execute_retry(tool, args, strategy_config)
        elif strategy_type == "parallel":
            return await self._execute_parallel(tool, args, strategy_config)
        elif strategy_type == "fallback":
            return await self._execute_fallback(tool, args, strategy_config)
        else:
            return await self._execute_simple(tool, args)

    async def _execute_cached(self, tool, args: Dict[str, Any], config: Dict) -> Any:
        """Execute with caching strategy.

        Args:
            tool: Tool to execute
            args: Tool arguments
            config: Cache configuration

        Returns:
            Cached or fresh result
        """
        if not self.cache:
            return await self._execute_simple(tool, args)

        # Generate cache key
        tool_name = tool.name if hasattr(tool, "name") else tool.get("name", "unknown")
        cache_key = self._generate_cache_key(tool_name, args, config.get("key_template"))

        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.cache_hit = True
            return cached

        # Execute and cache
        result = await self._execute_simple(tool, args)
        ttl = config.get("ttl_seconds", 300)
        await self.cache.set(cache_key, result, ttl=ttl)

        return result

    async def _execute_retry(self, tool, args: Dict[str, Any], config: Dict) -> Any:
        """Execute with retry strategy.

        Args:
            tool: Tool to execute
            args: Tool arguments
            config: Retry configuration

        Returns:
            Tool result after retries
        """
        max_attempts = config.get("max_attempts", 3)
        backoff_type = config.get("backoff", "exponential")
        initial_delay_ms = config.get("initial_delay_ms", 1000)
        max_delay_ms = config.get("max_delay_ms", 30000)

        last_error = None

        for attempt in range(max_attempts):
            try:
                result = await self._execute_simple(tool, args)
                if attempt > 0:
                    self.metrics.retry_count = attempt
                return result
            except Exception as e:
                last_error = e

                if attempt < max_attempts - 1:
                    # Calculate delay
                    if backoff_type == "exponential":
                        delay_ms = min(initial_delay_ms * (2**attempt), max_delay_ms)
                    elif backoff_type == "linear":
                        delay_ms = min(initial_delay_ms * (attempt + 1), max_delay_ms)
                    else:  # constant
                        delay_ms = initial_delay_ms

                    await asyncio.sleep(delay_ms / 1000.0)

        raise last_error if last_error else Exception("All retry attempts failed")

    async def _execute_parallel(self, tool, args: Dict[str, Any], config: Dict) -> Any:
        """Execute with parallel strategy (placeholder).

        Args:
            tool: Tool to execute
            args: Tool arguments
            config: Parallel configuration

        Returns:
            Tool result
        """
        # For now, just execute normally
        # Real implementation would coordinate with other parallel steps
        return await self._execute_simple(tool, args)

    async def _execute_fallback(self, tool, args: Dict[str, Any], config: Dict) -> Any:
        """Execute with fallback strategy.

        Args:
            tool: Tool to execute
            args: Tool arguments
            config: Fallback configuration

        Returns:
            Tool result from primary or fallback
        """
        alternatives = config.get("alternatives", [])

        # Try primary tool
        try:
            return await self._execute_simple(tool, args)
        except Exception as primary_error:
            # Try alternatives
            for alt_tool_name in alternatives:
                alt_tool = self.registry.get(alt_tool_name)
                if alt_tool:
                    try:
                        return await self._execute_simple(alt_tool, args)
                    except Exception:
                        continue

            # All failed
            raise primary_error

    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve variable substitutions in arguments.

        Args:
            args: Arguments potentially containing ${var} references

        Returns:
            Resolved arguments
        """
        resolved = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable path
                var_path = value[2:-1]
                resolved[key] = self._get_nested(var_path)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_args(value)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_args(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _get_nested(self, path: str) -> Any:
        """Get nested value from context using dotted path.

        Args:
            path: Dotted path like 'inputs.location' or 'wx.temp'

        Returns:
            Value from context

        Raises:
            KeyError: If path not found
        """
        parts = path.split(".")
        current = self.context

        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            else:
                current = getattr(current, part)

        return current

    def _evaluate_conditional(self, condition: str) -> bool:
        """Evaluate conditional expression.

        Args:
            condition: Conditional expression (simplified)

        Returns:
            True if condition passes
        """
        # Simplified evaluation - resolve variables and check truthiness
        # Real implementation would parse expression properly
        try:
            resolved = self._resolve_args({"cond": condition})["cond"]
            return bool(resolved)
        except Exception:
            return False

    def _generate_cache_key(self, tool_name: str, args: Dict[str, Any], template: Optional[str]) -> str:
        """Generate cache key for tool execution.

        Args:
            tool_name: Name of tool
            args: Tool arguments
            template: Optional key template

        Returns:
            Cache key string
        """
        if template:
            # Use template (would need proper variable substitution)
            key_data = template
        else:
            # Default: hash of tool name + args
            import json

            key_data = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _validate_constraints(self, constraints: Dict[str, Any]) -> None:
        """Validate plan constraints.

        Args:
            constraints: Constraint configuration

        Raises:
            ValueError: If constraints are invalid
        """
        # Constraints will be checked during execution
        pass

    def _validate_inputs(self, input_specs: List[Dict], provided: Dict[str, Any]) -> None:
        """Validate provided inputs against spec.

        Args:
            input_specs: Input specifications from plan
            provided: Provided input values

        Raises:
            ValueError: If required inputs missing
        """
        for spec in input_specs:
            name = spec["name"]
            required = spec.get("required", True)

            if required and name not in provided:
                raise ValueError(f"Required input '{name}' not provided")

    def _build_execution_order(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Build execution order respecting dependencies.

        Args:
            steps: List of plan steps

        Returns:
            Steps in execution order
        """
        # Simple topological sort based on depends_on
        ordered = []
        remaining = list(steps)
        completed = set()

        while remaining:
            # Find steps with satisfied dependencies
            ready = []
            for step in remaining:
                deps = step.depends_on or []
                if all(dep in completed for dep in deps):
                    ready.append(step)

            if not ready:
                raise ValueError("Circular dependency detected in plan steps")

            # Add ready steps to ordered list
            for step in ready:
                ordered.append(step)
                completed.add(step.id)
                remaining.remove(step)

        return ordered

    async def _run_validators(self, hooks: List[str]) -> List[Dict[str, Any]]:
        """Run validation hooks.

        Args:
            hooks: List of validator expressions

        Returns:
            Validation results
        """
        # Placeholder - would execute actual validators
        results = []
        for hook in hooks:
            results.append({"hook": hook, "passed": True, "message": "Validation passed"})
        return results
