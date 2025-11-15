"""PlanLang semantic validator."""

import re
from typing import List, Set

from ...contracts.models import Plan
from ...contracts.protocols import PlanValidator
from ...registry import UnifiedRegistry
from .schema import PlanLangSchema


class PlanLangValidator(PlanValidator):
    """Validates plans for semantic correctness."""

    def __init__(self, registry: UnifiedRegistry | None = None):
        """Initialize validator.

        Args:
            registry: Tool registry for existence checks
        """
        self.registry = registry
        self.schema = PlanLangSchema()

    async def validate(self, plan: Plan) -> List[str]:
        """Validate plan and return list of errors.

        Args:
            plan: Plan to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Basic structure validation
        errors.extend(self._validate_structure(plan))

        # Step validation
        errors.extend(self._validate_steps(plan))

        # Dependency validation
        errors.extend(self._validate_dependencies(plan))

        # Variable validation
        errors.extend(self._validate_variables(plan))

        # Tool existence check (if registry available)
        if self.registry:
            errors.extend(await self._validate_tools_exist(plan))

        # Strategy validation
        errors.extend(self._validate_strategies(plan))

        # Constraints validation
        errors.extend(self._validate_constraints(plan))

        return errors

    def check_tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists in registry.

        Args:
            tool_name: Tool name to check

        Returns:
            True if exists
        """
        # TODO: Implement registry check
        # This would require async registry.get_tool(tool_name)
        # For now, assume all tools exist
        return True

    def check_variable_dependencies(self, plan: Plan) -> List[str]:
        """Check that all variable dependencies are satisfied.

        Args:
            plan: Plan to check

        Returns:
            List of dependency errors
        """
        return self._validate_variables(plan)

    def _validate_structure(self, plan: Plan) -> List[str]:
        """Validate basic plan structure.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []

        # Name format
        if not re.match(r"^[a-z][a-z0-9_]*$", plan.name):
            errors.append(f"Plan name '{plan.name}' must be snake_case (lowercase, underscores)")

        # Version format
        if not re.match(r"^\d+\.\d+\.\d+$", plan.version):
            errors.append(f"Plan version '{plan.version}' must be semantic (X.Y.Z)")

        # Must have at least one step
        if not plan.steps:
            errors.append("Plan must have at least one step")

        # Inputs validation
        for i, input_spec in enumerate(plan.inputs):
            if "name" not in input_spec:
                errors.append(f"Input {i} missing 'name' field")
            elif not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", input_spec["name"]):
                errors.append(f"Input name '{input_spec['name']}' must be valid identifier")

        return errors

    def _validate_steps(self, plan: Plan) -> List[str]:
        """Validate individual steps.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []
        step_ids = set()

        for i, step in enumerate(plan.steps):
            # Step ID format
            if not re.match(r"^[a-z][a-z0-9_]*$", step.id):
                errors.append(f"Step {i} ID '{step.id}' must be snake_case")

            # Duplicate step IDs
            if step.id in step_ids:
                errors.append(f"Duplicate step ID: '{step.id}'")
            step_ids.add(step.id)

            # Save_as format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", step.save_as):
                errors.append(f"Step '{step.id}' save_as '{step.save_as}' must be valid identifier")

            # Tool name not empty
            if not step.tool:
                errors.append(f"Step '{step.id}' missing tool name")

        return errors

    def _validate_dependencies(self, plan: Plan) -> List[str]:
        """Validate step dependencies form valid DAG.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []
        step_ids = {step.id for step in plan.steps}

        for step in plan.steps:
            for dep in step.depends_on:
                # Dependency must reference existing step
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on non-existent step '{dep}'")

                # No self-dependencies
                if dep == step.id:
                    errors.append(f"Step '{step.id}' cannot depend on itself")

        # Check for cycles
        if self._has_cycle(plan):
            errors.append("Plan has circular dependencies")

        return errors

    def _validate_variables(self, plan: Plan) -> List[str]:
        """Validate variable references are defined.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []

        # Available variables at each step
        available_vars = {"inputs"}  # Special 'inputs' variable
        input_names = {inp["name"] for inp in plan.inputs}

        for step in plan.steps:
            # Check variable references in with_args
            for arg_name, arg_value in step.with_args.items():
                if isinstance(arg_value, str):
                    # Find ${var} references
                    matches = re.findall(self.schema.VARIABLE_PATTERN, arg_value)
                    for var_ref in matches:
                        # Split dotted path
                        parts = var_ref.split(".")

                        if parts[0] == "inputs":
                            # Check input name exists
                            if len(parts) > 1 and parts[1] not in input_names:
                                errors.append(
                                    f"Step '{step.id}' references undefined input: '{parts[1]}'"
                                )
                        elif parts[0] not in available_vars:
                            errors.append(
                                f"Step '{step.id}' references undefined variable: '{parts[0]}'"
                            )

            # Add this step's output to available vars
            available_vars.add(step.save_as)

        return errors

    async def _validate_tools_exist(self, plan: Plan) -> List[str]:
        """Validate that all referenced tools exist.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []

        if not self.registry:
            return errors

        # Check each tool
        for step in plan.steps:
            try:
                tool = await self.registry.get_tool(step.tool)
                if not tool:
                    errors.append(f"Step '{step.id}' references non-existent tool: '{step.tool}'")
            except Exception as e:
                # Registry error - skip validation
                pass

        return errors

    def _validate_strategies(self, plan: Plan) -> List[str]:
        """Validate strategy configurations.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []

        for step in plan.steps:
            if not step.strategy:
                continue

            # Get strategy type
            if isinstance(step.strategy, dict):
                strategy_type = step.strategy.get("type")
            else:
                continue

            # Check valid type
            if strategy_type not in self.schema.STRATEGY_TYPES:
                errors.append(
                    f"Step '{step.id}' has invalid strategy type: '{strategy_type}'"
                )
                continue

            # Validate strategy-specific config
            if "config" in step.strategy:
                config = step.strategy["config"]
                strategy_errors = self._validate_strategy_config(
                    strategy_type, config, step.id
                )
                errors.extend(strategy_errors)

        return errors

    def _validate_strategy_config(
        self,
        strategy_type: str,
        config: dict,
        step_id: str
    ) -> List[str]:
        """Validate strategy-specific configuration.

        Args:
            strategy_type: Type of strategy
            config: Configuration dict
            step_id: Step ID for error messages

        Returns:
            List of errors
        """
        errors = []

        if strategy_type == "cached":
            if "ttl_seconds" in config:
                ttl = config["ttl_seconds"]
                if not isinstance(ttl, int) or ttl < 1:
                    errors.append(f"Step '{step_id}' cached strategy: ttl_seconds must be positive integer")

        elif strategy_type == "retry":
            if "max_attempts" in config:
                attempts = config["max_attempts"]
                if not isinstance(attempts, int) or attempts < 1 or attempts > 10:
                    errors.append(f"Step '{step_id}' retry strategy: max_attempts must be 1-10")

            if "backoff" in config:
                backoff = config["backoff"]
                if backoff not in ["linear", "exponential", "constant"]:
                    errors.append(f"Step '{step_id}' retry strategy: invalid backoff type '{backoff}'")

        elif strategy_type == "fallback":
            if "alternatives" not in config:
                errors.append(f"Step '{step_id}' fallback strategy: missing 'alternatives'")
            elif not config["alternatives"]:
                errors.append(f"Step '{step_id}' fallback strategy: 'alternatives' cannot be empty")

        return errors

    def _validate_constraints(self, plan: Plan) -> List[str]:
        """Validate plan constraints.

        Args:
            plan: Plan to validate

        Returns:
            List of errors
        """
        errors = []

        if not plan.constraints:
            return errors

        if plan.constraints.max_latency_ms is not None:
            if plan.constraints.max_latency_ms < 0:
                errors.append("max_latency_ms must be non-negative")

        if plan.constraints.max_cost_cents is not None:
            if plan.constraints.max_cost_cents < 0:
                errors.append("max_cost_cents must be non-negative")

        if plan.constraints.max_steps is not None:
            if plan.constraints.max_steps < 1:
                errors.append("max_steps must be at least 1")
            elif plan.constraints.max_steps < len(plan.steps):
                errors.append(f"max_steps ({plan.constraints.max_steps}) less than actual steps ({len(plan.steps)})")

        return errors

    def _has_cycle(self, plan: Plan) -> bool:
        """Check if plan has circular dependencies.

        Args:
            plan: Plan to check

        Returns:
            True if cycle detected
        """
        # Build adjacency list
        graph = {step.id: step.depends_on for step in plan.steps}

        # Track visited and recursion stack
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check each node
        for step_id in graph:
            if step_id not in visited:
                if visit(step_id):
                    return True

        return False
