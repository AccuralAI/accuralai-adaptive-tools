# Benchmark Explanation: What Actually Happens

This document explains exactly what the adaptive vs baseline benchmark does, step by step.

## Overview

The benchmark compares two configurations:
1. **Baseline**: Standard AccuralAI with Google Gemini backend
2. **Adaptive**: Same setup + adaptive tools system (V3Coordinator) that can learn and optimize

## What Gets Sent to Google Gemini

### Example: "Multi-Step Data Processing" Scenario

For the baseline run, here's what happens:

#### Scenario Configuration
```python
{
    "name": "Multi-Step Data Processing",
    "prompts": [
        "Parse this CSV data and extract the top 10 rows by revenue",
        "Calculate the average revenue from the parsed data", 
        "Generate a summary report of the findings",
    ],
    "iterations": 5  # Run this sequence 5 times
}
```

#### What Actually Gets Sent

**For EACH prompt, the benchmark:**

1. **Creates a GenerateRequest**:
   ```python
   GenerateRequest(
       prompt="Parse this CSV data and extract the top 10 rows by revenue",
       metadata={"scenario": "Multi-Step Data Processing", "iteration": 0},
       # No system_prompt, history, tools, etc. - just the raw prompt
   )
   ```

2. **AccuralAI Pipeline Processes It**:
   - **Canonicalization**: Normalizes tags, metadata (minimal in this case)
   - **Cache Check**: Looks for cached response (won't find one on first run)
   - **Router**: Selects "google" backend (direct routing)
   - **Backend Call**: Sends to Google Gemini API

3. **What Google Gemini Actually Receives**:
   ```json
   {
     "model": "gemini-2.5-flash-lite",
     "contents": [
       {
         "role": "user",
         "parts": [{"text": "Parse this CSV data and extract the top 10 rows by revenue"}]
       }
     ],
     "generationConfig": {
       "temperature": 0.7
     }
   }
   ```

4. **Google Responds** with generated text (e.g., Python code or explanation)

5. **AccuralAI Processes Response**:
   - Validates response
   - Caches it (for future identical requests)
   - Returns `GenerateResponse` with:
     - `output_text`: The generated text
     - `usage`: Token counts (prompt_tokens, completion_tokens)
     - `latency_ms`: Time taken
     - `metadata`: Model info, safety ratings, etc.

## Baseline vs Adaptive: The Difference

### Baseline Mode

**Configuration:**
```python
{
    "backends": {
        "google": {
            "plugin": "google",
            "options": {
                "model": "gemini-2.5-flash-lite",
                "api_key": "...",
                "generation_config": {"temperature": 0.7}
            }
        }
    },
    "router": {"plugin": "direct", "options": {"default_backend": "google"}},
    "cache": {"plugin": "memory", "options": {"ttl_seconds": 300}}
}
```

**What happens:**
- Each prompt → Google API → Response
- Responses are cached (so repeated identical prompts are faster)
- No learning or optimization
- Same behavior every time

**Example Flow:**
```
Iteration 1:
  Prompt 1 → Google API (100ms, 500 tokens) → Cache miss → Store in cache
  Prompt 2 → Google API (120ms, 600 tokens) → Cache miss → Store in cache
  Prompt 3 → Google API (110ms, 550 tokens) → Cache miss → Store in cache

Iteration 2:
  Prompt 1 → Cache hit (5ms) ✅
  Prompt 2 → Cache hit (5ms) ✅
  Prompt 3 → Cache hit (5ms) ✅

Iterations 3-5: Same as iteration 2 (all cache hits)
```

### Adaptive Mode

**Configuration:**
- Same as baseline PLUS:
  - `UnifiedRegistry`: Tracks tools and plans
  - `TelemetryStorage`: Stores execution events
  - `TelemetryRouter`: Routes events to V1/V2 systems
  - `V3Coordinator`: Coordinates adaptive behavior

**What happens:**
- Each prompt → Google API → Response (same as baseline)
- **PLUS**: Telemetry events are recorded:
  ```python
  TelemetryEvent(
      event_type=EventType.TOOL_EXECUTED,
      latency_ms=100,
      cost_cents=0.05,
      success=True,
      item_id="Multi-Step Data Processing",
      item_type="scenario"
  )
  ```

- **V3Coordinator analyzes events**:
  - If same prompt repeated 10+ times → V1 might generate a tool
  - If latency > 500ms → V2 might optimize the workflow
  - Patterns detected → Tools generated → Future requests use tools

**Example Flow (with adaptive tools):**
```
Iteration 1:
  Prompt 1 → Google API (100ms) → Record telemetry
  Prompt 2 → Google API (120ms) → Record telemetry
  Prompt 3 → Google API (110ms) → Record telemetry
  → V3Coordinator: "Pattern detected: repeated data processing"

Iteration 2:
  Prompt 1 → Cache hit (5ms) ✅
  Prompt 2 → Cache hit (5ms) ✅
  Prompt 3 → Cache hit (5ms) ✅
  → V3Coordinator: "10+ repetitions detected, generating tool..."

Iteration 3:
  Prompt 1 → Cache hit (5ms) ✅
  Prompt 2 → Cache hit (5ms) ✅
  Prompt 3 → Cache hit (5ms) ✅
  → V1: Tool generated! "csv_parser_tool" now available
  → Future requests could use this tool instead of LLM

Iterations 4-5:
  → V2: Optimized workflow using generated tool
  → Faster execution, lower cost
```

## Metrics Collected

For each scenario run, the benchmark tracks:

### Per-Request Metrics
- **Latency**: Time from request start to response received (ms)
- **Cost**: Calculated from token usage:
  - Input: `(prompt_tokens / 1M) * $0.075`
  - Output: `(completion_tokens / 1M) * $0.30`
- **Tokens**: Total tokens used (prompt + completion)
- **Cache Hit**: Whether response came from cache
- **Quality**: Simple metric based on response length

### Scenario-Level Metrics
- **Total Latency**: Sum of all request latencies
- **Total Cost**: Sum of all request costs
- **Total Tokens**: Sum of all tokens
- **Success Rate**: Percentage of successful requests
- **Cache Hit Rate**: Percentage of cached responses
- **Average Quality**: Average response quality score

### Adaptive Tools Metrics (adaptive mode only)
- **Tools Generated**: Number of tools created by V1 system
- **Optimizations**: Number of workflow optimizations by V2 system

## Real Example: "Multi-Step Data Processing"

### Baseline Run

**What gets sent to Google (15 total requests = 3 prompts × 5 iterations):**

1. "Parse this CSV data and extract the top 10 rows by revenue" (5 times)
2. "Calculate the average revenue from the parsed data" (5 times)
3. "Generate a summary report of the findings" (5 times)

**First iteration (all cache misses):**
- Request 1: ~100ms, ~500 tokens → Google generates Python code
- Request 2: ~120ms, ~600 tokens → Google generates calculation code
- Request 3: ~110ms, ~550 tokens → Google generates report

**Subsequent iterations (cache hits):**
- Requests 4-15: ~5ms each (served from cache)

**Total**: ~330ms latency, ~1650 tokens, $0.0005 cost

### Adaptive Run

**Same requests sent to Google**, but:

**Additionally:**
- Each request creates a telemetry event
- V3Coordinator analyzes patterns
- After 10+ repetitions of similar patterns:
  - V1 might generate: `csv_data_processor_tool`
  - V2 might optimize: Combine all 3 steps into one optimized plan
  - Future requests use the tool/plan instead of raw LLM calls

**Result**: 
- First few iterations: Same as baseline (learning phase)
- Later iterations: Faster execution using generated tools
- Overall: Lower latency, lower cost, better quality

## Key Differences Summary

| Aspect | Baseline | Adaptive |
|--------|----------|----------|
| **What gets sent** | Same prompts to Google | Same prompts to Google |
| **Caching** | ✅ Yes (memory cache) | ✅ Yes (memory cache) |
| **Learning** | ❌ No | ✅ Yes (pattern detection) |
| **Tool Generation** | ❌ No | ✅ Yes (V1 system) |
| **Optimization** | ❌ No | ✅ Yes (V2 system) |
| **First Run** | Same performance | Same performance |
| **Repeated Runs** | Cache hits only | Cache hits + generated tools |
| **Long-term** | Static behavior | Improving behavior |

## Why This Matters

The benchmark demonstrates that:

1. **Short-term**: Adaptive tools have overhead (telemetry collection)
2. **Medium-term**: Adaptive tools break even (caching helps both)
3. **Long-term**: Adaptive tools improve (generated tools optimize workflows)

The "Repeated Pattern Detection" scenario is specifically designed to show this:
- Same prompt repeated 10 times
- Adaptive tools detect the pattern
- Generate a tool to handle it
- Future requests use the tool (faster, cheaper)

## Current Limitations

**Note**: The current benchmark implementation:
- Manually records telemetry (not fully integrated with orchestrator events)
- Doesn't actually generate tools yet (V1 synthesis not fully implemented)
- Doesn't optimize workflows yet (V2 optimizer not fully integrated)
- **But**: It demonstrates the framework and shows where improvements would come from

In production, the adaptive tools would:
- Hook into orchestrator's event system automatically
- Generate tools from detected patterns
- Optimize workflows based on metrics
- Show measurable improvements over time

