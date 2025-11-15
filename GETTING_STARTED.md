# Getting Started with AccuralAI Adaptive Tools V3

**Welcome!** This guide will help you understand and start using the V3 unified adaptive tools system.

## What is V3?

V3 is a **self-improving AI harness** that combines two complementary approaches:

### V1: Exploration (Tool Generation)
- **What it does**: Creates new tools from usage patterns
- **When it triggers**: Repeated manual sequences, high failure rates, missing capabilities
- **Output**: Python functions (atomic tools)
- **Example**: User repeatedly runs `list.directory` → `grep.text` → `write.file`, V1 generates `extract_errors` tool

### V2: Exploitation (Workflow Optimization)
- **What it does**: Optimizes existing workflows
- **When it triggers**: Latency bottlenecks, cost inefficiency, composition opportunities
- **Output**: PlanLang YAML (orchestration recipes)
- **Example**: V2 adds caching to `extract_errors`, parallelizes steps, reduces latency 40%

### V3: Coordination (Compound Gains)
- **What it does**: Bridges V1 and V2 for compound effects
- **Key insight**: V1 tools become building blocks for V2 optimizations
- **Result**: 1.6x compound improvement over baseline

## The Compound Loop

```
Day 1: User performs tasks manually
    ↓
Day 2: V1 detects pattern → generates tool
    ↓
Day 3: User adopts tool → performance improves 30%
    ↓
Day 7: V2 detects tool usage → optimizes workflow
    ↓
Day 8: Combined improvement: 1.3× (V1) × 1.4× (V2) = 1.82× total
    ↓
Loop continues: V1 creates → V2 optimizes → gains compound
```

## Current Status (Phase 1 Complete)

### ✅ What's Implemented

1. **Foundation Infrastructure** (~1100 lines)
   - Shared telemetry system (SQLite-based)
   - Unified registry (tools + plans)
   - Data models (15 models, 10 protocols)
   - Telemetry routing (V1 vs V2 decisions)

2. **Database Schemas**
   - `telemetry_events` table with indices
   - `unified_registry` table with cross-references
   - Analytics queries (sequences, failures, latency)

3. **Core Components**
   - `TelemetryStorage` - Event persistence
   - `TelemetryRouter` - Smart event routing
   - `SharedTelemetry` - Unified collection
   - `UnifiedRegistry` - Tool/plan management

### 🚧 What's Next (Phases 2-6)

1. **Phase 2: V1 Implementation** (~1800 lines)
   - Pattern detection
   - LLM code synthesis
   - Sandbox evaluation
   - Approval workflows

2. **Phase 3: V2 Implementation** (~2400 lines)
   - PlanLang parser
   - Plan executor with strategies
   - Bayesian optimizer
   - A/B testing

3. **Phase 4: V3 Coordination** (~700 lines)
   - Cross-system bridge
   - Decision logic (generate vs optimize)
   - Compound gains tracking

4. **Phase 5: CLI Integration** (~900 lines)
   - All CLI commands
   - Output formatting
   - Interactive REPL

5. **Phase 6: Testing** (~2500 lines)
   - Unit tests
   - Integration tests
   - End-to-end scenarios

## How to Explore the Codebase

### Start Here

1. **Read the V3 Specification**
   ```bash
   cat plan/accuralai-adaptive-tools-v3-spec.md
   ```

2. **Check Implementation Status**
   ```bash
   cat IMPLEMENTATION_STATUS.md
   ```

3. **Explore the Package**
   ```bash
   cd packages/accuralai-adaptive-tools
   tree accuralai_adaptive_tools/
   ```

### Key Files to Understand

#### Data Models (`contracts/models.py`)
- `TelemetryEvent` - What gets tracked
- `ToolProposal` - V1 output
- `Plan` - V2 output
- `CompoundGains` - V3 tracking

#### Telemetry System
- `telemetry/storage.py` - SQLite persistence (306 lines)
- `telemetry/router.py` - V1/V2 routing logic (57 lines)
- `telemetry/collector.py` - Main interface (43 lines)

#### Registry
- `registry/unified.py` - Tool/plan storage (305 lines)

### Database Schema

```sql
-- Telemetry events
CREATE TABLE telemetry_events (
    event_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP,
    event_type TEXT,          -- TOOL_EXECUTED, PLAN_EXECUTED, etc.
    item_id TEXT,              -- Which tool/plan
    latency_ms REAL,
    cost_cents REAL,
    success BOOLEAN,
    routed_to_v1 BOOLEAN,     -- Did this trigger V1?
    routed_to_v2 BOOLEAN      -- Did this trigger V2?
);

-- Unified registry
CREATE TABLE unified_registry (
    id TEXT PRIMARY KEY,
    type TEXT,                 -- 'tool' or 'plan'
    system TEXT,               -- 'v1', 'v2', or 'builtin'
    source_code TEXT,          -- V1 tools
    plan_yaml TEXT,            -- V2 plans
    uses_items JSON,           -- Dependencies
    used_by_items JSON         -- Reverse references
);
```

## Example Scenarios

### Scenario 1: Tool Generation (V1)

```python
# User performs actions (tracked by telemetry)
await telemetry.record(TelemetryEvent(
    event_type=EventType.TOOL_EXECUTED,
    item_id="list.directory",
    latency_ms=50
))
await telemetry.record(TelemetryEvent(
    event_type=EventType.TOOL_EXECUTED,
    item_id="grep.text",
    latency_ms=120
))
await telemetry.record(TelemetryEvent(
    event_type=EventType.TOOL_EXECUTED,
    item_id="write.file",
    latency_ms=30
))

# After 10+ repetitions, V1 pattern detector triggers
sequences = await telemetry.get_tool_sequences(min_length=3)
# Returns: [["list.directory", "grep.text", "write.file"], ...]

# V1 generates proposal (future implementation)
proposal = await v1.synthesizer.generate(pattern)

# Tool registered
await registry.register_tool(
    name="extract_errors",
    source_code=proposal.source_code,
    system=SystemType.V1
)
```

### Scenario 2: Workflow Optimization (V2)

```python
# V2 analyzes tool performance
stats = await telemetry.get_latency_stats("extract_errors")
# Returns: {"avg": 200ms, "p95": 450ms}

# V2 generates optimized plan (future implementation)
plan = await v2.planner.generate("extract_errors_optimized")

# Plan uses V1 tool with caching
plan = Plan(
    name="extract_errors_optimized",
    steps=[
        PlanStep(
            tool="extract_errors",  # V1 tool!
            strategy={"type": "cached", "ttl_seconds": 300}
        )
    ]
)

# Plan registered
await registry.register_plan(plan, SystemType.V2)

# Automatic reverse reference
dependents = await registry.get_dependents("extract_errors")
# Returns: ["plan_extract_errors_optimized_1.0.0"]
```

### Scenario 3: Cross-System Effects (V3)

```python
# V3 coordinator detects compound opportunity
v1_tools = await registry.get_v1_tools_for_v2()
# V2 can now use all V1 tools in optimizations

# Track compound gains
gain = CompoundGains(
    v1_tool_id="tool_extract_errors_v1",
    v2_plan_id="plan_extract_errors_optimized_1.0",
    baseline_latency_ms=200,
    improved_latency_ms=120,
    latency_improvement_pct=40.0,
    individual_gain=1.4,  # V2 optimization
    compound_gain=1.82    # V1 (1.3x) × V2 (1.4x)
)
```

## Configuration Example

```toml
[adaptive_tools]
enabled = true
mode = "v3"  # Unified mode

[adaptive_tools.v3]
auto_coordinate = true
v1_sequence_threshold = 10      # Repetitions before proposing
v2_latency_threshold_ms = 500   # Trigger optimization

[adaptive_tools.telemetry]
storage_path = "~/.accuralai/adaptive-tools/telemetry.db"
retention_days = 30
```

## Development Workflow

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e packages/accuralai-adaptive-tools[dev]

# Run tests (when available)
pytest packages/accuralai-adaptive-tools/tests -v

# Lint code
ruff check packages/accuralai-adaptive-tools/
```

### Adding New Features

1. **Define contracts** in `contracts/models.py` or `contracts/protocols.py`
2. **Implement component** in appropriate subdirectory (`v1/`, `v2/`, `coordinator/`)
3. **Add tests** in `tests/` mirroring the structure
4. **Update documentation** in this file and `/IMPLEMENTATION_STATUS.md`

### Testing Telemetry System

```python
import asyncio
from datetime import datetime
from accuralai_adaptive_tools import (
    SharedTelemetry,
    TelemetryRouter,
    TelemetryStorage,
    TelemetryEvent,
    EventType,
)

async def test_telemetry():
    # Initialize
    storage = TelemetryStorage("test.db")
    router = TelemetryRouter()
    telemetry = SharedTelemetry(storage, router)

    # Record event
    event = TelemetryEvent(
        event_id="test_001",
        event_type=EventType.TOOL_EXECUTED,
        item_id="read.file",
        latency_ms=45.5,
        success=True
    )
    await telemetry.record(event)

    # Query
    recent = await telemetry.get_recent(hours=1)
    print(f"Events: {len(recent)}")

    # Analytics
    sequences = await telemetry.get_tool_sequences()
    print(f"Sequences: {sequences}")

asyncio.run(test_telemetry())
```

### Testing Registry

```python
import asyncio
from accuralai_adaptive_tools import UnifiedRegistry, SystemType, Plan, PlanStep

async def test_registry():
    registry = UnifiedRegistry("test_registry.db")

    # Register V1 tool
    tool_id = await registry.register_tool(
        name="test_tool",
        source_code="async def test_tool(): pass",
        function_schema={"args": []},
        system=SystemType.V1
    )
    print(f"Registered: {tool_id}")

    # Register V2 plan that uses the tool
    plan = Plan(
        name="test_plan",
        version="1.0.0",
        steps=[PlanStep(id="step1", tool="test_tool", save_as="result")]
    )
    plan_id = await registry.register_plan(plan, SystemType.V2)

    # Check cross-references
    dependents = await registry.get_dependents("test_tool")
    print(f"Plans using test_tool: {dependents}")

asyncio.run(test_registry())
```

## Next Steps

### For Users (When Phase 2-6 Complete)

1. Install package: `pip install accuralai-adaptive-tools`
2. Configure: Create `~/.accuralai/config.toml`
3. Use CLI: Run `accuralai` and start using `/tool` and `/plan` commands
4. Monitor: Check `/adaptive status` to see compound gains

### For Contributors (Now)

1. **Review specifications** in `/plan/` directory
2. **Study Phase 1 implementation** (telemetry, registry)
3. **Pick a component** from Phases 2-4
4. **Implement + test** following the patterns established
5. **Submit PR** with tests and documentation

### For Researchers

1. **Study the architecture**: How V1+V2+V3 interact
2. **Analyze compound gains**: Mathematical models in V3 spec
3. **Propose improvements**: Better optimization algorithms, pattern detection
4. **Benchmark**: Compare against baselines

## Resources

### Documentation
- `/plan/accuralai-adaptive-tools-v3-spec.md` - Complete architecture
- `/plan/adaptive-tools-comparison.md` - V1 vs V2 vs V3
- `/IMPLEMENTATION_STATUS.md` - Current progress
- `/plan/QUICKSTART-ADAPTIVE-TOOLS.md` - User quick start

### Code
- `packages/accuralai-adaptive-tools/` - Main package
- `packages/accuralai-adaptive-tools/tests/` - Test suite (future)

### Community
- GitHub Issues - Bug reports and feature requests
- GitHub Discussions - Questions and ideas

## FAQ

### Q: Is V3 production-ready?
**A**: Not yet. Phase 1 (foundation) is complete, but V1, V2, and V3 coordination are still in development. Estimated completion: 6-12 weeks.

### Q: Can I use just V1 or just V2?
**A**: Yes! The architecture is modular. Set `mode = "v1"` or `mode = "v2"` in config.

### Q: How is this different from Langchain agents?
**A**: V3 focuses on **compounding improvements over time** rather than single-shot agentic workflows. It learns from telemetry, generates tools, and optimizes compositions automatically.

### Q: What LLMs are supported for code generation?
**A**: Any LLM backend in AccuralAI (Google Gemini, OpenAI, Anthropic, etc.). Configurable via `adaptive_tools.v1.synthesis.backend_id`.

### Q: How safe is generated code?
**A**: Multiple layers: AST analysis (forbids `eval`/`exec`), import whitelist, sandbox execution, human approval for high-risk code.

### Q: Can V2 plans call external APIs?
**A**: Yes, if you have tools that wrap those APIs. V2 orchestrates existing tools, doesn't create new ones (that's V1's job).

---

**Ready to contribute?** See `/IMPLEMENTATION_STATUS.md` for current tasks, or explore the codebase and pick a component to implement!

**Questions?** Open an issue on GitHub or check the `/plan/` directory for detailed specifications.
