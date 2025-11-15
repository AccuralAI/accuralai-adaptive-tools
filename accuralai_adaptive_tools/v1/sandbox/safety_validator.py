"""Safety validator using AST analysis."""

import ast
import re
from typing import List, Set

from ...contracts.models import RiskLevel, SafetyReport
from ...contracts.protocols import SafetyValidator


class SafetyValidatorImpl(SafetyValidator):
    """Validates generated code for safety using AST analysis."""

    # Forbidden function calls
    FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__",
        "open",  # Use pathlib instead
        "input",  # No interactive input in tools
    }

    # Forbidden subprocess patterns
    FORBIDDEN_SUBPROCESS = {
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.run",
        "os.system",
        "os.popen",
        "os.spawn",
    }

    # Allowed standard library imports
    ALLOWED_STDLIB = {
        "pathlib", "json", "csv", "datetime", "time", "re", "collections",
        "itertools", "functools", "typing", "dataclasses", "enum",
        "math", "statistics", "random", "uuid", "hashlib",
        "base64", "urllib.parse", "html", "xml.etree.ElementTree",
    }

    # Allowed third-party imports (AccuralAI ecosystem)
    ALLOWED_THIRD_PARTY = {
        "accuralai_core.cli.tools.models",
        "accuralai_core.contracts",
    }

    def __init__(
        self,
        allow_subprocess: bool = False,
        allow_network: bool = False,
        custom_allowed_imports: Set[str] | None = None,
    ):
        """Initialize safety validator.

        Args:
            allow_subprocess: Allow subprocess operations
            allow_network: Allow network operations
            custom_allowed_imports: Additional allowed import modules
        """
        self.allow_subprocess = allow_subprocess
        self.allow_network = allow_network
        self.custom_allowed_imports = custom_allowed_imports or set()

    async def validate(self, source_code: str) -> SafetyReport:
        """Analyze code and return safety report.

        Args:
            source_code: Python source code to analyze

        Returns:
            SafetyReport with validation results
        """
        forbidden_patterns = []
        suspicious_imports = []
        validation_errors = []
        recommendations = []

        try:
            # Parse AST
            tree = ast.parse(source_code)

            # Check for forbidden patterns
            forbidden_patterns = self.check_forbidden_patterns(source_code)

            # Check imports
            suspicious_imports = self.check_imports(source_code)

            # Validate AST structure
            validation_errors.extend(self._validate_ast(tree))

            # Add recommendations based on findings
            if forbidden_patterns:
                recommendations.append("Remove or refactor forbidden operations")
            if suspicious_imports:
                recommendations.append("Use only standard library or explicitly allowed modules")

            # Calculate risk level
            risk_level = self._calculate_risk_level(
                forbidden_patterns, suspicious_imports, validation_errors
            )

            # Check if passed
            passed = (
                not forbidden_patterns and
                not suspicious_imports and
                not validation_errors
            )

        except SyntaxError as e:
            validation_errors.append(f"Syntax error: {e}")
            risk_level = RiskLevel.HIGH
            passed = False

        except Exception as e:
            validation_errors.append(f"Analysis error: {e}")
            risk_level = RiskLevel.HIGH
            passed = False

        return SafetyReport(
            risk_level=risk_level,
            forbidden_patterns=forbidden_patterns,
            suspicious_imports=suspicious_imports,
            validation_errors=validation_errors,
            recommendations=recommendations,
            passed=passed,
        )

    def check_forbidden_patterns(self, source_code: str) -> List[str]:
        """Check for forbidden patterns in code.

        Args:
            source_code: Python source code

        Returns:
            List of forbidden patterns found
        """
        patterns = []

        try:
            tree = ast.parse(source_code)

            # Check for forbidden function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node)

                    if func_name in self.FORBIDDEN_CALLS:
                        patterns.append(f"Forbidden call: {func_name}")

                    if not self.allow_subprocess:
                        for forbidden in self.FORBIDDEN_SUBPROCESS:
                            if func_name == forbidden or func_name.endswith(f".{forbidden}"):
                                patterns.append(f"Forbidden subprocess: {func_name}")

        except SyntaxError:
            patterns.append("Code has syntax errors")

        # Additional regex checks for obfuscation attempts
        if re.search(r'__[a-z]+__\s*\(', source_code):
            patterns.append("Suspicious dunder method call")

        return patterns

    def check_imports(self, source_code: str) -> List[str]:
        """Check for suspicious imports.

        Args:
            source_code: Python source code

        Returns:
            List of suspicious imports
        """
        suspicious = []

        try:
            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_allowed_import(alias.name):
                            suspicious.append(f"Import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if not self._is_allowed_import(module):
                        suspicious.append(f"Import from: {module}")

        except SyntaxError:
            pass

        return suspicious

    def _is_allowed_import(self, module_name: str) -> bool:
        """Check if import is allowed.

        Args:
            module_name: Full module name

        Returns:
            True if allowed
        """
        # Check standard library
        if module_name.split(".")[0] in self.ALLOWED_STDLIB:
            return True

        # Check third-party allowed
        if module_name in self.ALLOWED_THIRD_PARTY:
            return True

        # Check custom allowed
        if module_name in self.custom_allowed_imports:
            return True

        # Check if starts with allowed prefix
        for allowed in self.ALLOWED_THIRD_PARTY | self.custom_allowed_imports:
            if module_name.startswith(allowed + "."):
                return True

        return False

    def _validate_ast(self, tree: ast.AST) -> List[str]:
        """Validate AST structure.

        Args:
            tree: Parsed AST

        Returns:
            List of validation errors
        """
        errors = []

        # Must contain at least one async function definition
        has_async_function = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                has_async_function = True
                break

        if not has_async_function:
            errors.append("Must contain at least one async function")

        return errors

    def _get_function_name(self, call_node: ast.Call) -> str:
        """Extract function name from Call node.

        Args:
            call_node: AST Call node

        Returns:
            Function name string
        """
        func = call_node.func

        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # Handle module.function calls
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            return ".".join(parts)

        return ""

    def _calculate_risk_level(
        self,
        forbidden_patterns: List[str],
        suspicious_imports: List[str],
        validation_errors: List[str]
    ) -> RiskLevel:
        """Calculate risk level based on findings.

        Args:
            forbidden_patterns: List of forbidden patterns
            suspicious_imports: List of suspicious imports
            validation_errors: List of validation errors

        Returns:
            RiskLevel enum value
        """
        # CRITICAL: Forbidden patterns (eval, exec, subprocess)
        critical_keywords = ["eval", "exec", "subprocess", "os.system"]
        for pattern in forbidden_patterns:
            pattern_lower = pattern.lower()
            if any(kw in pattern_lower for kw in critical_keywords):
                return RiskLevel.CRITICAL

        # HIGH: Suspicious imports or validation errors
        if suspicious_imports or validation_errors:
            return RiskLevel.HIGH

        # MEDIUM: Other forbidden patterns
        if forbidden_patterns:
            return RiskLevel.MEDIUM

        # LOW: Clean code
        return RiskLevel.LOW
