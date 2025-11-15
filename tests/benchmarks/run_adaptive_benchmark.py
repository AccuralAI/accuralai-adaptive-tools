#!/usr/bin/env python3
"""Standalone script to run the adaptive tools benchmark.

Usage:
    export GOOGLE_GENAI_API_KEY=your_key_here
    python run_adaptive_benchmark.py [scenario_name]

Examples:
    # Run all scenarios
    python run_adaptive_benchmark.py

    # Run specific scenario
    python run_adaptive_benchmark.py "Multi-Step Data Processing"
"""

import asyncio
import os
import sys

# Add benchmarks directory to path so imports work
benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmarks_dir)

from test_adaptive_vs_baseline_benchmark import AdaptiveVsBaselineBenchmark


async def main():
    """Main entry point."""
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_GENAI_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export GOOGLE_GENAI_API_KEY=your_key_here")
        print("\nOr on Windows:")
        print("  set GOOGLE_GENAI_API_KEY=your_key_here")
        sys.exit(1)

    scenario_name = sys.argv[1] if len(sys.argv) > 1 else None

    print("Starting Adaptive Tools Benchmark...")
    print(f"API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else '****'}")
    print()

    benchmark = AdaptiveVsBaselineBenchmark(api_key=api_key)
    results = await benchmark.run_comparison(scenario_name=scenario_name)
    benchmark.print_summary(results)

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
    
    if avg_score > 20:
        print("✅ Adaptive tools show significant improvement!")
        print("   Consider enabling adaptive tools in production.")
    elif avg_score > 0:
        print("✅ Adaptive tools show modest improvement.")
        print("   Consider enabling for workloads with repeated patterns.")
    elif avg_score > -10:
        print("⚠️  Adaptive tools show neutral performance.")
        print("   May be beneficial for long-running workloads.")
    else:
        print("❌ Adaptive tools show performance degradation.")
        print("   Review configuration and telemetry collection overhead.")


if __name__ == "__main__":
    asyncio.run(main())

