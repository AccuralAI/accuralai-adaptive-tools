"""PlanLang serializer for converting plans to/from various formats."""

import json
from typing import Any, Dict

from ...contracts.models import Plan
from .parser import PlanLangParser


class PlanLangSerializer:
    """Serializes plans to YAML/JSON formats."""

    def __init__(self):
        """Initialize serializer."""
        self.parser = PlanLangParser()

    def to_yaml(self, plan: Plan) -> str:
        """Convert plan to YAML string.

        Args:
            plan: Plan to serialize

        Returns:
            YAML string
        """
        return self.parser.to_yaml(plan)

    def to_json(self, plan: Plan) -> str:
        """Convert plan to JSON string.

        Args:
            plan: Plan to serialize

        Returns:
            JSON string
        """
        # Use pydantic's json method
        plan_dict = plan.dict(exclude_none=True)
        return json.dumps(plan_dict, indent=2)

    def to_dict(self, plan: Plan) -> Dict[str, Any]:
        """Convert plan to dictionary.

        Args:
            plan: Plan to serialize

        Returns:
            Dictionary representation
        """
        return plan.dict(exclude_none=True)

    def from_dict(self, data: Dict[str, Any]) -> Plan:
        """Create plan from dictionary.

        Args:
            data: Plan dictionary

        Returns:
            Plan object
        """
        return Plan(**data)
