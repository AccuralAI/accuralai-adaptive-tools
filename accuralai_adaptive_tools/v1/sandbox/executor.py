"""Sandbox executor for safe code testing."""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from ...contracts.models import SandboxResult
from ...contracts.protocols import SandboxExecutor


class SandboxExecutorImpl(SandboxExecutor):
    """Executes code in isolated subprocess sandbox."""

    def __init__(
        self,
        timeout_s: int = 10,
        max_memory_mb: int = 256,
        allowed_paths: List[str] | None = None,
    ):
        """Initialize sandbox executor.

        Args:
            timeout_s: Maximum execution time per test
            max_memory_mb: Maximum memory usage
            allowed_paths: Allowed file system paths for I/O
        """
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb
        self.allowed_paths = allowed_paths or [tempfile.gettempdir()]

    async def test(
        self,
        source_code: str,
        test_cases: List[Dict[str, Any]]
    ) -> SandboxResult:
        """Execute code with test cases in sandbox.

        Args:
            source_code: Python code to test
            test_cases: List of test case dictionaries

        Returns:
            SandboxResult with test outcomes
        """
        start_time = time.time()
        errors = []
        passed_count = 0

        if not test_cases:
            return SandboxResult(
                passed=False,
                test_cases_run=0,
                test_cases_passed=0,
                errors=["No test cases provided"],
                execution_time_ms=0,
            )

        # Create temporary directory for sandbox
        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)

            # Write code to file
            code_file = sandbox_path / "tool.py"
            code_file.write_text(source_code)

            # Run each test case
            for i, test_case in enumerate(test_cases):
                try:
                    result = await self._run_test_case(
                        code_file,
                        test_case,
                        sandbox_path,
                        i
                    )

                    if result["success"]:
                        passed_count += 1
                    else:
                        errors.append(f"Test {i} ({test_case.get('name', 'unnamed')}): {result['error']}")

                except asyncio.TimeoutError:
                    errors.append(f"Test {i} ({test_case.get('name', 'unnamed')}): Timeout after {self.timeout_s}s")

                except Exception as e:
                    errors.append(f"Test {i} ({test_case.get('name', 'unnamed')}): {str(e)}")

        execution_time_ms = (time.time() - start_time) * 1000

        return SandboxResult(
            passed=(passed_count == len(test_cases) and not errors),
            test_cases_run=len(test_cases),
            test_cases_passed=passed_count,
            errors=errors,
            execution_time_ms=execution_time_ms,
        )

    async def _run_test_case(
        self,
        code_file: Path,
        test_case: Dict[str, Any],
        sandbox_path: Path,
        test_index: int
    ) -> Dict[str, Any]:
        """Run single test case.

        Args:
            code_file: Path to code file
            test_case: Test case dictionary
            sandbox_path: Sandbox directory
            test_index: Test index

        Returns:
            Dict with success status and error message
        """
        # Create test script
        test_script = self._create_test_script(code_file, test_case, test_index)
        test_file = sandbox_path / f"test_{test_index}.py"
        test_file.write_text(test_script)

        # Run in subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                str(test_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(sandbox_path),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_s
            )

            if process.returncode == 0:
                return {"success": True, "error": None}
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return {"success": False, "error": error_msg}

        except asyncio.TimeoutError:
            # Kill process
            try:
                process.kill()
                await process.wait()
            except:
                pass
            raise

    def _create_test_script(
        self,
        code_file: Path,
        test_case: Dict[str, Any],
        test_index: int
    ) -> str:
        """Create test script for execution.

        Args:
            code_file: Path to code file
            test_case: Test case dictionary
            test_index: Test index

        Returns:
            Python test script as string
        """
        # Extract test parameters
        args = test_case.get("args", [])
        call_args = test_case.get("call_args", {})
        expected_status = test_case.get("expected_status", "success")

        # Create test script
        script = f"""
import sys
import asyncio
from pathlib import Path

# Import real ToolResult and SessionState (or create compatible mocks)
try:
    from accuralai_core.cli.tools.models import ToolResult, SessionState
except ImportError:
    # Fallback: Create compatible mocks if imports fail
    class ToolResult:
        def __init__(self, status, message, data=None):
            self.status = status
            self.message = message
            self.data = data
    
    class SessionState:
        def __init__(self):
            self.history = []
            self.context = {{}}

# Mock SessionState for testing (use real one if available, otherwise use mock)
MockSessionState = SessionState

# Import the tool code
sys.path.insert(0, str(Path(__file__).parent))
exec(open("{code_file.name}").read(), globals())

# Run test
async def main():
    state = MockSessionState()
    args = {args!r}
    call_args = {call_args!r}

    # Extract function name (first async def)
    import ast
    tree = ast.parse(open("{code_file.name}").read())
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            func_name = node.name
            break

    if func_name is None:
        print("ERROR: No async function found", file=sys.stderr)
        sys.exit(1)

    # Call function
    try:
        func = globals()[func_name]
        result = await func(state, args, call_args)

        # Check result - use duck typing to be more flexible
        if not hasattr(result, 'status') or not hasattr(result, 'message'):
            print(f"ERROR: Expected ToolResult-like object with 'status' and 'message' attributes, got {{type(result)}}", file=sys.stderr)
            sys.exit(1)

        expected_status = "{expected_status}"
        if result.status != expected_status:
            print(f"ERROR: Expected status '{{expected_status}}', got '{{result.status}}'", file=sys.stderr)
            sys.exit(1)

        print(f"SUCCESS: Test passed")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
"""
        return script

    async def cleanup(self):
        """Clean up sandbox resources."""
        # Cleanup is handled by TemporaryDirectory context manager
        pass
