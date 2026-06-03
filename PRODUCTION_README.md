# AnonyLLM - World Model SLM 2026

**Creator:** Aryan Chavan  
**License:** MIT  
**Version:** 2.0 (Production)

A production-ready Small Language Model (SLM) with world model reasoning, multi-agent feedback, RL self-improvement, and extensible plugin system. Runs on laptops (4GB GPU) and mobiles.

## Quick Start
```bash
# UI (recommended)
python main.py --ui

# API Server
python main.py --server

# Training
python main.py --train
```

## System Capabilities
- **World Model**: Simulates scenarios before responding
- **GRPO RL**: Self-improves through experience
- **Multi-Agent**: Generator + Critic + Optimizer + Planner
- **Hallucination Reduction**: Causal + Physics + Self-evaluation
- **Extensible Tools**: Voice, calculator, web search, code exec, plugins
- **RAG Memory**: FAISS vector database for document grounding
- **Auto Complexity**: Dynamically adjusts thinking depth

## Under 50GB
Total project: ~1.5 GB. Model: 360M params (~250MB fp16, ~140MB int4).

## Extending with Tools
Drop a `.py` file in `plugins/` with a `register(registry)` function:
```python
from core.plugin_system import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "What it does"
    async def execute(self, **params) -> ToolResult:
        return ToolResult(success=True, result="done")

def register(registry):
    registry.register(MyTool())
```

## Mobile Deployment
```bash
python scripts/optimize_for_mobile.py
export SLM_MODE=mobile && python main.py --server
```
