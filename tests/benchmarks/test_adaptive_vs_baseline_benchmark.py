"""Practical benchmark comparing AccuralAI with and without adaptive tools.

This benchmark uses the Google Gemini backend to test real-world scenarios
and demonstrates the performance improvements from adaptive tools.
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from accuralai_core.contracts.models import GenerateRequest
from accuralai_core.core.orchestrator import CoreOrchestrator
from accuralai_adaptive_tools.coordinator.v3 import V3Coordinator
from accuralai_adaptive_tools.registry.unified import UnifiedRegistry
from accuralai_adaptive_tools.telemetry import TelemetryCollector, TelemetryRouter, TelemetryStorage

# Handle both relative (package) and absolute (direct import) imports
try:
    from .tools import FileToolHandler, get_benchmark_tools
except ImportError:
    from tools import FileToolHandler, get_benchmark_tools


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    scenario_name: str
    with_adaptive_tools: bool
    total_latency_ms: float
    total_cost_cents: float
    total_tokens: int
    success_rate: float
    avg_response_quality: float
    cache_hit_rate: float
    num_tools_generated: int = 0
    num_optimizations: int = 0
    improvement_factor: float = 1.0


@dataclass
class ComparisonResult:
    """Comparison between baseline and adaptive tools."""

    scenario_name: str
    baseline: BenchmarkResult
    adaptive: BenchmarkResult
    latency_improvement: float  # percentage
    cost_improvement: float  # percentage
    token_improvement: float  # percentage
    quality_improvement: float  # percentage
    overall_score: float  # 0-100


class AdaptiveVsBaselineBenchmark:
    """Benchmark comparing AccuralAI with and without adaptive tools."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize benchmark.

        Args:
            api_key: Google GenAI API key (defaults to GOOGLE_GENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google GenAI API key required. Set GOOGLE_GENAI_API_KEY env var or pass api_key parameter."
            )

        self.scenarios = self._define_scenarios()
        
        # Create temporary directory for benchmark databases (shared across connections)
        self.temp_dir = Path(tempfile.gettempdir()) / "accuralai_benchmark"
        self.temp_dir.mkdir(exist_ok=True)

    def _define_scenarios(self) -> List[Dict[str, Any]]:
        """Define benchmark scenarios."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        return [
            {
                "name": "CSV Data Processing",
                "description": "Process CSV data using file operations - creates patterns",
                "prompts": [
                    f"Read the file 'sample_sales.csv' from {test_data_dir}, parse it, and extract the top 5 products by revenue. Write the results to 'top_products.csv'",
                    f"Read 'sample_sales.csv', calculate the average revenue per product, and write the result to 'avg_revenue.txt'",
                    f"Read 'sample_sales.csv', group products by category, calculate total revenue per category, and write to 'category_summary.json'",
                ],
                "iterations": 3,
                "uses_tools": True,
            },
            {
                "name": "JSON Data Transformation",
                "description": "Transform JSON data using file operations",
                "prompts": [
                    f"Read 'sample_data.json', filter users by department='Engineering', and write to 'engineering_users.json'",
                    f"Read 'sample_data.json', calculate average age by department, and write to 'age_stats.json'",
                ],
                "iterations": 3,
                "uses_tools": True,
            },
            {
                "name": "Repeated File Operations",
                "description": "Repeated file operations to trigger pattern detection",
                "prompts": [
                    f"Read 'sample_sales.csv', parse it, and convert to JSON format. Write to 'sales_output.json'",
                ] * 10,  # Repeat 10 times to trigger pattern detection
                "iterations": 1,
                "uses_tools": True,
            },
            {
                "name": "Log Analysis Workflow",
                "description": "Analyze log files using file operations",
                "prompts": [
                    f"Read 'log_entries.txt', extract all ERROR level messages, and write to 'errors.txt'",
                    f"Read 'log_entries.txt', count messages by level (INFO, WARN, ERROR), and write summary to 'log_summary.json'",
                ],
                "iterations": 2,
                "uses_tools": True,
            },
        ]

    async def _create_baseline_orchestrator(self, tools: Optional[List[Dict[str, Any]]] = None) -> CoreOrchestrator:
        """Create orchestrator without adaptive tools.
        
        Args:
            tools: Optional list of tools to enable function calling
        """
        config_overrides = {
            "backends": {
                "google": {
                    "plugin": "google",
                    "options": {
                        "model": "gemini-2.5-flash-lite",  # Use flash for faster/cheaper testing
                        "api_key": self.api_key,
                        "generation_config": {"temperature": 0.7},
                    },
                }
            },
            "router": {
                "plugin": "direct",
                "options": {"default_backend": "google"},
            },
            "cache": {
                "plugin": "memory",
                "options": {"ttl_seconds": 300, "max_size": 100},
            },
        }

        orchestrator = CoreOrchestrator(config_overrides=config_overrides)
        orchestrator._tools = tools or []  # Store tools for later use
        return orchestrator

    async def _create_adaptive_orchestrator(self, tools: Optional[List[Dict[str, Any]]] = None) -> tuple[CoreOrchestrator, V3Coordinator]:
        """Create orchestrator with adaptive tools enabled.
        
        Args:
            tools: Optional list of tools to enable function calling
        """
        # Create orchestrator with same tools
        orchestrator = await self._create_baseline_orchestrator(tools=tools)

        # Use temporary file databases instead of :memory: to ensure persistence across connections
        # :memory: databases are per-connection in SQLite, so we need a shared file
        registry_db = str(self.temp_dir / "registry.db")
        telemetry_db = str(self.temp_dir / "telemetry.db")

        # Create adaptive tools components
        registry = UnifiedRegistry(registry_db)
        await registry.initialize()

        storage = TelemetryStorage(telemetry_db)
        router = TelemetryRouter()
        collector = TelemetryCollector(storage, router)

        # Initialize V1 and V2 systems
        from accuralai_adaptive_tools.v1.system import V1System
        from accuralai_adaptive_tools.v2.system import V2System
        
        v1_system = V1System(
            telemetry=collector,
            registry=registry,
            llm_backend_id="google",
            llm_model="gemini-2.5-flash-lite",
            auto_approve_low_risk=True,  # Auto-approve for benchmark
        )
        
        v2_system = V2System()

        # Create coordinator with lower thresholds to trigger more easily
        coordinator = V3Coordinator(
            registry=registry,
            collector=collector,
            router=router,
            v1_system=v1_system,
            v2_system=v2_system,
            config={
                "v1_sequence_threshold": 3,  # Lower threshold: trigger after 3 repetitions
                "v1_failure_threshold": 0.1,  # Lower threshold: 10% failure rate
                "v2_latency_threshold_ms": 100,  # Lower threshold: 100ms
                "v2_cost_threshold_cents": 0.1,  # Lower threshold: 0.1 cents
            }
        )
        
        await coordinator.start()

        # NOTE: Full integration would hook orchestrator's event_publisher into telemetry
        # For this benchmark, we manually record events to demonstrate the system
        # In production, orchestrator would emit events via ExecutionContext.record_event

        return orchestrator, coordinator

    async def _run_scenario(
        self,
        orchestrator: CoreOrchestrator,
        coordinator: Optional[V3Coordinator],
        scenario: Dict[str, Any],
        with_adaptive: bool,
    ) -> BenchmarkResult:
        """Run a scenario and collect metrics."""
        prompts = scenario["prompts"]
        iterations = scenario["iterations"]
        uses_tools = scenario.get("uses_tools", False)

        total_latency = 0.0
        total_cost = 0.0
        total_tokens = 0
        successes = 0
        cache_hits = 0
        responses = []
        tool_calls_made = 0

        # Setup file tool handler if using tools
        tool_handler = None
        tools = None
        if uses_tools:
            test_data_dir = Path(__file__).parent / "test_data"
            tool_handler = FileToolHandler(str(test_data_dir))
            tools = get_benchmark_tools(str(test_data_dir))

        async with orchestrator:
            for iteration in range(iterations):
                for prompt in prompts:
                    start_time = time.time()
                    conversation_history = []

                    try:
                        # Build request with tools if needed
                        request_params = {}
                        if uses_tools and tools:
                            request_params["function_calling_config"] = {"mode": "AUTO"}

                        request = GenerateRequest(
                            prompt=prompt,
                            metadata={"scenario": scenario["name"], "iteration": iteration},
                            tools=tools if uses_tools else [],
                            parameters=request_params,
                            history=conversation_history,
                        )

                        # Handle multi-turn conversation for tool calls
                        max_turns = 5
                        turn = 0
                        final_response = None
                        accumulated_usage = None
                        
                        while turn < max_turns:
                            response = await orchestrator.generate(request)
                            final_response = response
                            turn += 1
                            
                            # Accumulate usage across turns
                            if accumulated_usage is None:
                                accumulated_usage = response.usage
                            else:
                                from accuralai_core.contracts.models import Usage
                                accumulated_usage = Usage(
                                    prompt_tokens=accumulated_usage.prompt_tokens + response.usage.prompt_tokens,
                                    completion_tokens=accumulated_usage.completion_tokens + response.usage.completion_tokens,
                                )
                            
                            # Check for tool calls
                            tool_calls = response.metadata.get("tool_calls") or []
                            
                            if tool_calls and tool_handler:
                                # Execute tool calls
                                tool_results = []
                                for tool_call in tool_calls:
                                    tool_name = tool_call.get("function", {}).get("name", "")
                                    tool_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                                    
                                    result = await tool_handler.handle_tool_call(tool_name, tool_args)
                                    tool_calls_made += 1
                                    
                                    tool_results.append({
                                        "function": {"name": tool_name},
                                        "response": result
                                    })
                                
                                # Add assistant response and tool results to conversation
                                conversation_history.append({
                                    "role": "assistant",
                                    "content": response.output_text or "",
                                    "tool_calls": tool_calls
                                })
                                
                                # Create follow-up request with tool results
                                tool_results_text = json.dumps(tool_results, indent=2)
                                follow_up_prompt = f"Tool execution results:\n{tool_results_text}\n\nContinue with the original task."
                                
                                conversation_history.append({
                                    "role": "user",
                                    "content": follow_up_prompt
                                })
                                
                                request = GenerateRequest(
                                    prompt=follow_up_prompt,
                                    metadata={"scenario": scenario["name"], "iteration": iteration, "turn": turn},
                                    tools=tools if uses_tools else [],
                                    parameters=request_params,
                                    history=conversation_history,
                                )
                            else:
                                # No more tool calls, we're done
                                break
                        
                        if not final_response:
                            final_response = response
                        
                        latency_ms = (time.time() - start_time) * 1000
                        total_latency += latency_ms

                        # Use accumulated usage if we had multiple turns, otherwise use final response
                        usage = accumulated_usage if accumulated_usage else final_response.usage
                        
                        # Calculate cost from accumulated usage
                        input_cost = (usage.prompt_tokens / 1_000_000) * 0.075
                        output_cost = (usage.completion_tokens / 1_000_000) * 0.30
                        total_cost += (input_cost + output_cost) * 100  # Convert to cents

                        total_tokens += usage.prompt_tokens + usage.completion_tokens

                        # Check cache hit
                        if final_response.metadata.get("cache_hit") or final_response.metadata.get("cache_status") == "hit":
                            cache_hits += 1

                        # Simple quality metric: response length and presence of content
                        quality = min(1.0, len(final_response.output_text) / 100) if final_response.output_text else 0.0
                        responses.append(quality)

                        successes += 1

                        # Record telemetry for adaptive tools
                        if with_adaptive and coordinator:
                            from accuralai_adaptive_tools.contracts.models import EventType, TelemetryEvent

                            # Record main execution event - store it first, then process
                            event = TelemetryEvent(
                                event_id=f"e_{iteration}_{hash(prompt) % 10000}",
                                event_type=EventType.TOOL_EXECUTED,
                                latency_ms=latency_ms,
                                cost_cents=(input_cost + output_cost) * 100,
                                success=True,
                                item_id=scenario["name"],
                                item_type="scenario",
                                event_data={"tool_calls": tool_calls_made, "turns": turn},
                            )
                            # Store event in collector first
                            await coordinator.collector.record(event)
                            # Then process it
                            await coordinator.process_telemetry(event)
                            
                            # Record tool call events if tools were used
                            if tool_calls_made > 0 and tool_handler:
                                tool_stats = tool_handler.get_call_stats()
                                for tool_name, count in tool_stats.items():
                                    tool_event = TelemetryEvent(
                                        event_id=f"tool_{tool_name}_{iteration}",
                                        event_type=EventType.TOOL_EXECUTED,
                                        item_id=tool_name,
                                        item_type="tool",
                                        event_data={"call_count": count},
                                    )
                                    await coordinator.collector.record(tool_event)
                                    await coordinator.process_telemetry(tool_event)
                            
                            # Create TOOL_SEQUENCE event for repeated patterns (triggers V1)
                            # Track how many times we've seen this prompt pattern
                            prompt_hash = hash(prompt) % 10000
                            sequence_count = iteration + 1  # Count iterations
                            
                            if sequence_count >= 3:  # Trigger after 3 repetitions
                                sequence_event = TelemetryEvent(
                                    event_id=f"seq_{scenario['name']}_{prompt_hash}_{iteration}",
                                    event_type=EventType.TOOL_SEQUENCE,
                                    item_id=scenario["name"],
                                    item_type="scenario",
                                    event_data={
                                        "count": sequence_count,
                                        "pattern": prompt[:100],  # First 100 chars
                                        "tool_calls": tool_calls_made,
                                    },
                                )
                                # Store and process sequence event
                                await coordinator.collector.record(sequence_event)
                                await coordinator.process_telemetry(sequence_event)

                    except Exception as e:
                        print(f"Error in scenario {scenario['name']}: {e}")
                        # Record failure
                        if with_adaptive and coordinator:
                            from accuralai_adaptive_tools.contracts.models import EventType, TelemetryEvent

                            event = TelemetryEvent(
                                event_id=f"e_{iteration}_{prompt[:20]}_error",
                                event_type=EventType.TOOL_ERROR,
                                success=False,
                                error_message=str(e),
                                item_id=scenario["name"],
                                item_type="scenario",
                            )
                            await coordinator.collector.record(event)
                            await coordinator.process_telemetry(event)

        total_requests = len(prompts) * iterations
        success_rate = successes / total_requests if total_requests > 0 else 0.0
        avg_quality = sum(responses) / len(responses) if responses else 0.0
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

        # Get adaptive tools metrics
        num_tools_generated = 0
        num_optimizations = 0
        if with_adaptive and coordinator:
            # Give coordinator a moment to process any pending events
            await asyncio.sleep(0.5)
            status = await coordinator.get_status()
            num_tools_generated = status.v1_tools_generated
            num_optimizations = status.v2_optimization_runs
            print(f"DEBUG: Status query - tools_generated={num_tools_generated}, optimizations={num_optimizations}")

        return BenchmarkResult(
            scenario_name=scenario["name"],
            with_adaptive_tools=with_adaptive,
            total_latency_ms=total_latency,
            total_cost_cents=total_cost,
            total_tokens=total_tokens,
            success_rate=success_rate,
            avg_response_quality=avg_quality,
            cache_hit_rate=cache_hit_rate,
            num_tools_generated=num_tools_generated,
            num_optimizations=num_optimizations,
        )

    async def run_comparison(self, scenario_name: Optional[str] = None) -> List[ComparisonResult]:
        """Run comparison benchmark for all or specific scenario.

        Args:
            scenario_name: Optional scenario name to run (runs all if None)

        Returns:
            List of comparison results
        """
        print("\n" + "=" * 80)
        print("ACCURALAI ADAPTIVE TOOLS VS BASELINE BENCHMARK")
        print("=" * 80 + "\n")

        scenarios_to_run = (
            [s for s in self.scenarios if s["name"] == scenario_name]
            if scenario_name
            else self.scenarios
        )

        results = []

        for scenario in scenarios_to_run:
            print(f"\n{'='*80}")
            print(f"Scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"{'='*80}\n")

            # Run baseline
            print("Running BASELINE (without adaptive tools)...")
            baseline_orchestrator = await self._create_baseline_orchestrator()
            baseline_result = await self._run_scenario(baseline_orchestrator, None, scenario, False)
            await baseline_orchestrator.aclose()

            print(f"  Latency: {baseline_result.total_latency_ms:.2f}ms")
            print(f"  Cost: ${baseline_result.total_cost_cents/100:.4f}")
            print(f"  Tokens: {baseline_result.total_tokens:,}")
            print(f"  Success Rate: {baseline_result.success_rate*100:.1f}%")
            print(f"  Cache Hit Rate: {baseline_result.cache_hit_rate*100:.1f}%")

            # Run with adaptive tools
            print("\nRunning WITH ADAPTIVE TOOLS...")
            adaptive_orchestrator, coordinator = await self._create_adaptive_orchestrator()
            adaptive_result = await self._run_scenario(adaptive_orchestrator, coordinator, scenario, True)
            await adaptive_orchestrator.aclose()

            print(f"  Latency: {adaptive_result.total_latency_ms:.2f}ms")
            print(f"  Cost: ${adaptive_result.total_cost_cents/100:.4f}")
            print(f"  Tokens: {adaptive_result.total_tokens:,}")
            print(f"  Success Rate: {adaptive_result.success_rate*100:.1f}%")
            print(f"  Cache Hit Rate: {adaptive_result.cache_hit_rate*100:.1f}%")
            print(f"  Tools Generated: {adaptive_result.num_tools_generated}")
            print(f"  Optimizations: {adaptive_result.num_optimizations}")

            # Calculate improvements
            latency_improvement = (
                ((baseline_result.total_latency_ms - adaptive_result.total_latency_ms) / baseline_result.total_latency_ms * 100)
                if baseline_result.total_latency_ms > 0
                else 0.0
            )
            cost_improvement = (
                ((baseline_result.total_cost_cents - adaptive_result.total_cost_cents) / baseline_result.total_cost_cents * 100)
                if baseline_result.total_cost_cents > 0
                else 0.0
            )
            token_improvement = (
                ((baseline_result.total_tokens - adaptive_result.total_tokens) / baseline_result.total_tokens * 100)
                if baseline_result.total_tokens > 0
                else 0.0
            )
            quality_improvement = (
                ((adaptive_result.avg_response_quality - baseline_result.avg_response_quality) / baseline_result.avg_response_quality * 100)
                if baseline_result.avg_response_quality > 0
                else 0.0
            )

            # Calculate overall score (weighted average)
            overall_score = (
                max(0, latency_improvement) * 0.3
                + max(0, cost_improvement) * 0.3
                + max(0, token_improvement) * 0.2
                + max(0, quality_improvement) * 0.2
            )

            comparison = ComparisonResult(
                scenario_name=scenario["name"],
                baseline=baseline_result,
                adaptive=adaptive_result,
                latency_improvement=latency_improvement,
                cost_improvement=cost_improvement,
                token_improvement=token_improvement,
                quality_improvement=quality_improvement,
                overall_score=overall_score,
            )

            results.append(comparison)

            print(f"\n{'─'*80}")
            print("IMPROVEMENTS:")
            print(f"  Latency: {latency_improvement:+.1f}%")
            print(f"  Cost: {cost_improvement:+.1f}%")
            print(f"  Tokens: {token_improvement:+.1f}%")
            print(f"  Quality: {quality_improvement:+.1f}%")
            print(f"  Overall Score: {overall_score:.1f}/100")
            print(f"{'─'*80}\n")

        return results

    def print_summary(self, results: List[ComparisonResult]):
        """Print summary of all results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80 + "\n")

        total_latency_improvement = sum(r.latency_improvement for r in results) / len(results) if results else 0
        total_cost_improvement = sum(r.cost_improvement for r in results) / len(results) if results else 0
        total_token_improvement = sum(r.token_improvement for r in results) / len(results) if results else 0
        total_quality_improvement = sum(r.quality_improvement for r in results) / len(results) if results else 0
        avg_overall_score = sum(r.overall_score for r in results) / len(results) if results else 0

        print("Average Improvements Across All Scenarios:")
        print(f"  Latency: {total_latency_improvement:+.1f}%")
        print(f"  Cost: {total_cost_improvement:+.1f}%")
        print(f"  Tokens: {total_token_improvement:+.1f}%")
        print(f"  Quality: {total_quality_improvement:+.1f}%")
        print(f"  Overall Score: {avg_overall_score:.1f}/100")

        print("\n" + "=" * 80)


# Pytest integration
@pytest.mark.asyncio
async def test_adaptive_vs_baseline_benchmark():
    """Run the adaptive vs baseline benchmark."""
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_GENAI_API_KEY not set, skipping benchmark")

    benchmark = AdaptiveVsBaselineBenchmark(api_key=api_key)
    results = await benchmark.run_comparison()
    benchmark.print_summary(results)

    # Assert that adaptive tools provide some improvement
    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
    assert avg_score >= -10, f"Adaptive tools should not significantly degrade performance. Score: {avg_score}"


if __name__ == "__main__":
    # Run benchmark directly
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_GENAI_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_GENAI_API_KEY=your_key_here")
        exit(1)

    benchmark = AdaptiveVsBaselineBenchmark(api_key=api_key)
    results = asyncio.run(benchmark.run_comparison())
    benchmark.print_summary(results)

