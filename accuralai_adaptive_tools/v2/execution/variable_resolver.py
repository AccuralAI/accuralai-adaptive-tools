"""Variable resolution with ${var} substitution."""

import re
from typing import Any, Dict

from .context import ExecutionContext


class VariableResolver:
    """Resolves ${var} references in strings and data structures."""

    # Pattern for ${var.path.to.field}
    VARIABLE_PATTERN = r"\$\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"

    def resolve(self, value: Any, context: ExecutionContext) -> Any:
        """Resolve variables in value.

        Args:
            value: Value to resolve (string, dict, list, or primitive)
            context: Execution context

        Returns:
            Resolved value
        """
        if isinstance(value, str):
            return self._resolve_string(value, context)
        elif isinstance(value, dict):
            return self._resolve_dict(value, context)
        elif isinstance(value, list):
            return self._resolve_list(value, context)
        else:
            # Primitive types (int, float, bool, None) pass through
            return value

    def _resolve_string(self, text: str, context: ExecutionContext) -> Any:
        """Resolve variables in string.

        Args:
            text: String with ${var} references
            context: Execution context

        Returns:
            Resolved value (may be string or extracted object)
        """
        # Find all ${var} references
        matches = list(re.finditer(self.VARIABLE_PATTERN, text))

        if not matches:
            return text

        # If entire string is a single variable reference, return the actual value
        if len(matches) == 1 and matches[0].group(0) == text:
            var_path = matches[0].group(1)
            return self._resolve_variable_path(var_path, context)

        # Otherwise, do string substitution
        result = text
        for match in matches:
            var_ref = match.group(0)  # Full ${var.path}
            var_path = match.group(1)  # Just var.path

            try:
                value = self._resolve_variable_path(var_path, context)
                # Convert to string for substitution
                result = result.replace(var_ref, str(value))
            except (KeyError, AttributeError) as e:
                # Variable not found - keep placeholder
                pass

        return result

    def _resolve_dict(self, data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Resolve variables in dictionary.

        Args:
            data: Dictionary with potential ${var} references
            context: Execution context

        Returns:
            Dictionary with resolved values
        """
        return {key: self.resolve(value, context) for key, value in data.items()}

    def _resolve_list(self, data: list, context: ExecutionContext) -> list:
        """Resolve variables in list.

        Args:
            data: List with potential ${var} references
            context: Execution context

        Returns:
            List with resolved values
        """
        return [self.resolve(item, context) for item in data]

    def _resolve_variable_path(self, var_path: str, context: ExecutionContext) -> Any:
        """Resolve dotted variable path.

        Args:
            var_path: Variable path (e.g., "inputs.log_directory" or "matches")
            context: Execution context

        Returns:
            Resolved value

        Raises:
            KeyError: If variable not found
            AttributeError: If path invalid
        """
        parts = var_path.split(".")

        # Get root variable
        root_name = parts[0]
        value = context.get_variable(root_name)

        # Navigate path
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise AttributeError(f"Cannot access '{part}' in '{var_path}'")

        return value

    def has_unresolved_variables(self, value: Any) -> bool:
        """Check if value contains unresolved ${var} references.

        Args:
            value: Value to check

        Returns:
            True if has unresolved variables
        """
        if isinstance(value, str):
            return bool(re.search(self.VARIABLE_PATTERN, value))
        elif isinstance(value, dict):
            return any(self.has_unresolved_variables(v) for v in value.values())
        elif isinstance(value, list):
            return any(self.has_unresolved_variables(item) for item in value)
        else:
            return False
