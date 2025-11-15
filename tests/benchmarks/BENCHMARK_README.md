# Adaptive Tools Benchmark Suite

This directory contains benchmarks for testing the AccuralAI adaptive tools system.

## Benchmarks

### 1. Adaptive vs Baseline Benchmark (`test_adaptive_vs_baseline_benchmark.py`)

**Purpose**: Compare AccuralAI performance with and without adaptive tools using real Google Gemini API calls.

**What it does**:
- Runs practical scenarios (data processing, code generation, research)
- Tests baseline performance (standard AccuralAI)
- Tests with adaptive tools enabled
- Compares metrics: latency, cost, tokens, quality, cache hits
- Shows improvements from tool generation and optimization

**Requirements**:
- `GOOGLE_GENAI_API_KEY` environment variable set
- `accuralai-core` and `accuralai-google` packages installed
- Internet connection for API calls

**Usage**:
```bash
export GOOGLE_GENAI_API_KEY=your_key_here

# Run all scenarios
python tests/benchmarks/run_adaptive_benchmark.py

# Run specific scenario
python tests/benchmarks/run_adaptive_benchmark.py "Multi-Step Data Processing"

# Via pytest
pytest tests/benchmarks/test_adaptive_vs_baseline_benchmark.py -v -s
```

**Expected Results**:
- **First run**: May show neutral or slightly negative performance (adaptive tools overhead)
- **Repeated patterns**: Should show 10-30% improvements in:
  - Latency (through caching and optimization)
  - Cost (through token efficiency)
  - Quality (through learned optimizations)

### 2. System Benchmark (`test_adaptive_system_benchmark.py`)

**Purpose**: Comprehensive testing of all subsystems (V1, V2, V3) with mock data.

**What it does**:
- Tests V2 plan execution performance
- Tests V2 Bayesian optimization
- Tests V2 A/B testing framework
- Tests V3 coordination efficiency
- Tests compound gains calculation
- Tests telemetry throughput
- Tests end-to-end integration

**Usage**:
```bash
python tests/benchmarks/test_adaptive_system_benchmark.py
```

**Scoring**:
- Each category scored 0-100
- Overall grade: A+ (90+), A (85+), A- (80+), B+ (75+), B (70+), C (60+), D (<60)

## Interpreting Results

### Adaptive vs Baseline Benchmark

**Good signs**:
- ✅ Latency improvement > 10% in repeated scenarios
- ✅ Cost reduction > 10% in repeated scenarios
- ✅ Tools generated > 0 for pattern-heavy scenarios
- ✅ Cache hit rate increases over time

**Expected patterns**:
- First scenario run: Neutral performance (setup overhead)
- Repeated scenarios: Gradual improvement as tools are generated
- Pattern-heavy scenarios: Significant improvements (20-30%)

**Troubleshooting**:
- If all scenarios show negative performance: Check API key, network connectivity
- If no tools generated: Check telemetry collection, pattern thresholds
- If improvements are minimal: May need more iterations or different scenarios

### System Benchmark

**Good signs**:
- ✅ Overall score > 80 (Grade A- or better)
- ✅ V2 Execution score > 70 (good performance)
- ✅ V2 Optimization score > 60 (optimization working)
- ✅ V3 Coordination score > 80 (routing working)

**Troubleshooting**:
- Low execution scores: Check executor implementation
- Low optimization scores: Check Bayesian optimizer configuration
- Low coordination scores: Check telemetry routing logic

## Customizing Benchmarks

### Adding New Scenarios

Edit `test_adaptive_vs_baseline_benchmark.py` and add to `_define_scenarios()`:

```python
{
    "name": "Your Scenario Name",
    "description": "What it tests",
    "prompts": [
        "First prompt",
        "Second prompt",
    ],
    "iterations": 5,
}
```

### Adjusting Metrics

Modify the `_run_scenario()` method to:
- Add custom metrics
- Change quality scoring
- Adjust cost calculations
- Add domain-specific measurements

### Changing Backend

Modify `_create_baseline_orchestrator()` to use a different backend:

```python
config_overrides = {
    "backends": {
        "your_backend": {
            "plugin": "your_plugin",
            "options": {...}
        }
    },
    ...
}
```

## Cost Considerations

**Google Gemini API Costs** (as of 2024):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Benchmark costs**:
- Full benchmark suite: ~$0.10 - $0.50 (depending on scenarios)
- Single scenario: ~$0.01 - $0.10

**Tips**:
- Use `gemini-2.5-flash-lite` for faster/cheaper testing
- Use `gemini-1.5-pro` for quality-focused benchmarks
- Reduce iterations for cost savings
- Cache results when possible

## Future Enhancements

- [ ] Real-time telemetry integration with orchestrator
- [ ] More diverse scenarios (multi-modal, long context)
- [ ] Cost tracking and budgeting
- [ ] Performance regression detection
- [ ] Automated benchmark runs in CI/CD
- [ ] Comparison with other LLM providers

