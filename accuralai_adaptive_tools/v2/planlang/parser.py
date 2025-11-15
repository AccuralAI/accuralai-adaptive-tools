"""PlanLang YAML/JSON parser."""

import yaml
from typing import Any, Dict

from ...contracts.models import Plan, PlanConstraints, PlanStep
from ...contracts.protocols import PlanParser


class PlanLangParser(PlanParser):
    """Parses PlanLang YAML/JSON into Plan objects."""

    def parse(self, yaml_content: str) -> Plan:
        """Parse YAML/JSON content into Plan.

        Args:
            yaml_content: YAML or JSON string

        Returns:
            Parsed Plan object

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Parse YAML (also handles JSON)
            data = yaml.safe_load(yaml_content)

            if not isinstance(data, dict):
                raise ValueError("Plan must be a YAML object/dictionary")

            # Validate required fields
            self._validate_required_fields(data)

            # Parse steps
            steps = [self._parse_step(step_data) for step_data in data.get("steps", [])]

            # Parse constraints
            constraints = self._parse_constraints(data.get("constraints", {}))

            # Create Plan
            plan = Plan(
                name=data["name"],
                version=data["version"],
                description=data.get("description", ""),
                inputs=data.get("inputs", []),
                steps=steps,
                constraints=constraints,
                eval_hooks=data.get("eval_hooks", []),
                metadata=data.get("metadata", {}),
            )

            return plan

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            raise ValueError(f"Plan parsing error: {e}")

    def to_yaml(self, plan: Plan) -> str:
        """Convert Plan to YAML.

        Args:
            plan: Plan object to serialize

        Returns:
            YAML string
        """
        # Convert to dict
        data = {
            "name": plan.name,
            "version": plan.version,
        }

        if plan.description:
            data["description"] = plan.description

        if plan.inputs:
            data["inputs"] = plan.inputs

        # Convert steps
        data["steps"] = [self._step_to_dict(step) for step in plan.steps]

        # Add constraints if any
        if plan.constraints:
            constraints_dict = {}
            if plan.constraints.max_latency_ms is not None:
                constraints_dict["max_latency_ms"] = plan.constraints.max_latency_ms
            if plan.constraints.max_cost_cents is not None:
                constraints_dict["max_cost_cents"] = plan.constraints.max_cost_cents
            if plan.constraints.max_steps is not None:
                constraints_dict["max_steps"] = plan.constraints.max_steps

            if constraints_dict:
                data["constraints"] = constraints_dict

        if plan.eval_hooks:
            data["eval_hooks"] = plan.eval_hooks

        if plan.metadata:
            data["metadata"] = plan.metadata

        # Convert to YAML
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def _validate_required_fields(self, data: Dict[str, Any]):
        """Validate required fields are present.

        Args:
            data: Parsed YAML data

        Raises:
            ValueError: If required fields missing
        """
        required = ["name", "version", "steps"]
        for field in required:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing")

        if not data["steps"]:
            raise ValueError("Plan must have at least one step")

    def _parse_step(self, step_data: Dict[str, Any]) -> PlanStep:
        """Parse step data into PlanStep.

        Args:
            step_data: Step dictionary

        Returns:
            PlanStep object

        Raises:
            ValueError: If step data invalid
        """
        # Required fields
        required = ["id", "tool", "save_as"]
        for field in required:
            if field not in step_data:
                raise ValueError(f"Step missing required field '{field}'")

        # Parse strategy
        strategy = None
        if "strategy" in step_data:
            strategy_data = step_data["strategy"]
            if isinstance(strategy_data, str):
                # Simple strategy: just type name
                strategy = {"type": strategy_data}
            elif isinstance(strategy_data, dict):
                # Full strategy with config
                strategy = strategy_data
            else:
                raise ValueError(f"Invalid strategy format in step '{step_data['id']}'")

        return PlanStep(
            id=step_data["id"],
            tool=step_data["tool"],
            with_args=step_data.get("with_args", {}),
            save_as=step_data["save_as"],
            depends_on=step_data.get("depends_on", []),
            strategy=strategy,
        )

    def _parse_constraints(self, constraints_data: Dict[str, Any]) -> PlanConstraints:
        """Parse constraints data.

        Args:
            constraints_data: Constraints dictionary

        Returns:
            PlanConstraints object
        """
        return PlanConstraints(
            max_latency_ms=constraints_data.get("max_latency_ms"),
            max_cost_cents=constraints_data.get("max_cost_cents"),
            max_steps=constraints_data.get("max_steps"),
        )

    def _step_to_dict(self, step: PlanStep) -> Dict[str, Any]:
        """Convert PlanStep to dictionary.

        Args:
            step: PlanStep object

        Returns:
            Dictionary representation
        """
        step_dict = {
            "id": step.id,
            "tool": step.tool,
            "with_args": step.with_args,
            "save_as": step.save_as,
        }

        if step.depends_on:
            step_dict["depends_on"] = step.depends_on

        if step.strategy:
            step_dict["strategy"] = step.strategy

        return step_dict
