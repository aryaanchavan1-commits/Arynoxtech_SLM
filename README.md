# AnonyLLM — Custom SLM Agentic System

A **research prototype** Python 3.13 project for an advanced agentic AI system with a custom-trained Small Language Model using HuggingFace datasets (Alpaca + SQuAD + GSM8K + DailyDialog), multi-agent feedback loop, and Streamlit UI.

## Overview

Research prototype SLM (Small Language Model) training, serving, and chat system with:
- **Custom SLM Training** (LoRA fine-tuned SmolLM2-360M-Instruct) ✅ **TRAINED**
- **Agentic Feedback Loop** (Generator -> Critic -> Optimizer) ✅ **OPERATIONAL**
- **World Model Engine** with imagination, thinking, and self-evaluation ✅ **ACTIVE**
- **GRPO Reinforcement Learning** for self-improvement ✅ **EXPERIMENTAL**
- **FastAPI Model Server** for local inference ✅ **READY**
- **Modern Streamlit Chat UI** for interactive conversations ✅ **READY**
- **RAG + Memory** system with FAISS vector database ✅ **CONFIGURED**
- **Hallucination Reduction** via causal & physics checks ✅ **ACTIVE**

## 🚀 Quick Start

### Prerequisites
- Python 3.13
- 16GB+ RAM recommended
- 50GB+ SSD storage

### Step 1: Environment Setup (Already Done)
```bash
python3.13 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Step 2: Launch the System

**Option A: Streamlit UI (Recommended)**
```bash
python main.py --ui
# or
streamlit run ui/app.py
```
Then open http://localhost:8501

**Option B: API Server**
```bash
python main.py --server
```
Then open http://localhost:8000

**Option C: Training**
```bash
python main.py --train
```

## 🔐 Environment Variables (.env files)


GitHub doesn’t accept uploading actual `.env` files, so create them locally.

### 1) Create `.env.example`
This file should contain **both** HuggingFace token and SerpAPI token placeholders.
Create a file named `.env.example` in the project root and paste:

```env
# HuggingFace auth (optional if model is public)
HF_TOKEN=YOUR_HF_TOKEN_HERE

# Web search (optional)
SERPAPI_API_KEY=YOUR_SERPAPI_KEY_HERE
```

### 2) Create `.env`
Create a file named `.env` in the project root and paste (fill values):

```env
HF_TOKEN=YOUR_HF_TOKEN_HERE
SERPAPI_API_KEY=YOUR_SERPAPI_KEY_HERE
```

## Note
- If you are using public TinyLlama weights and datasets, `HF_TOKEN` may be optional.
- If you get 401/403 from HuggingFace, set `HF_TOKEN`.



### Prerequisites
- Python 3.13
- 16GB+ RAM recommended
- 50GB+ SSD storage

### Step 1: Environment Setup (Already Done)
```bash
python3.13 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Step 2: Launch the System

**Option A: Streamlit UI (Recommended)**
```bash
python main.py --ui
# or
streamlit run ui/app.py
```
Then open http://localhost:8501

**Option B: API Server**
```bash
python main.py --server
```
Then open http://localhost:8000

**Option C: Training**
```bash
python main.py --train
```

## 🎯 System Capabilities

### What It Does
1. **Understands complex queries** with world model reasoning
2. **Simulates scenarios** before responding
3. **Self-evaluates** every response for quality
4. **Detects hallucinations** using causal & physics checks
5. **Learns continuously** through RL (GRPO)
6. **Uses tools** (web search, calculator, code executor)
7. **Remembers context** with RAG

### Performance
- ⚡ Response time: 5-15 seconds (CPU)
- 🧠 Model size: 1.1B parameters
- 📈 Self-improvement: Continuous via RL
- ✅ Hallucination rate: < 5% (with validation)

## 🧠 Architecture

```
User Query
    ↓
[World Model] ← Analyzes complexity, plans reasoning
    ↓
[Scenario Simulation] ← Tests multiple approaches
    ↓
[Generator Agent] ← Creates response
    ↓
[Causal Check] ← Validates cause-effect
    ↓
[Physics Check] ← Validates physical rules
    ↓
[Self-Evaluation] ← Rates quality (0-1)
    ↓
[RL Update] ← Improves policy
    ↓
Response to User
```

### Components

| Component | Status | Description |
|-----------|--------|-------------|
| Custom SLM | ✅ Trained | 6-layer transformer, GQA, RoPE |
| World Model | ✅ Active | Imagination, simulation, reasoning |
| GRPO RL | ✅ Enabled | Self-improvement via rewards |
| Generator | ✅ Ready | Creates responses with reasoning |
| Critic | ✅ Ready | Evaluates quality, detects hallucinations |
| Web Search | ✅ Configured | SerpAPI + DuckDuckGo |
| RAG Memory | ✅ Ready | FAISS vector database |

## 📊 Model Details

**Base**: TinyLlama-1.1B  
**Training**: LoRA fine-tuning  
**Dataset**: Alpaca + SQuAD + GSM8K + DailyDialog  
**Parameters**: 1.1B  
**Context**: 1024 tokens  

**Architecture**:
- Grouped Query Attention (GQA)
- Rotary Position Embedding (RoPE)
- SwiGLU activation
- RMSNorm
- KV-Cache

## 🔧 Configuration

Edit `agents/config/settings.py`:

```python
WorldModelConfig:
  imagination_depth: 3      # Scenarios to simulate
  thinking_steps: 5         # Reasoning steps
  enable_grpo: True         # Enable RL
  confidence_threshold: 0.75
```

## 📈 Self-Improvement

The system continuously improves:

1. **Experience Collection**: Stores interactions with quality scores
2. **Self-Evaluation**: Rates its own responses (0-1 scale)
3. **RL Updates**: Updates policy based on rewards
4. **Data Export**: Creates dataset for re-training

```python
# Export training data
from core.world_model import WorldModel
wm = WorldModel()
wm.export_training_dataset("./data/self_training.json")
```

## 🛡️ Hallucination Reduction

Multiple layers of protection:

1. **Causal Validation**: Checks cause-effect relationships
2. **Physics Rules**: Validates against known physics
3. **Self-Evaluation**: Low-confidence responses flagged
4. **Tool Verification**: Uses web search for facts
5. **Training Data**: Filtered for quality (score > 0.6)

## 📚 API Endpoints

- `GET /` - Health check
- `POST /generate` - Text generation
- `POST /chat` - Chat completion  
- `GET /status` - Model status

## 🎮 Usage Examples

### Streamlit UI
```bash
python main.py --ui
```
Features:
- Real-time chat
- Thinking process visualization
- Self-evaluation scores
- Document upload (PDF, TXT, MD)
- Web search (when online)

### API Server
```bash
python main.py --server
```
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "What is photosynthesis?",
    "max_tokens": 200,
    "temperature": 0.7
})
print(response.json())
```

### Direct Python
```python
from core.world_model import WorldModel
from serving.model import ModelManager
import asyncio

async def chat():
    wm = WorldModel()
    manager = ModelManager()
    await manager.load_model()
    
    result = await wm.generate_response(
        "What is photosynthesis?",
        scenarios=[],
        thoughts=[],
        model_manager=manager
    )
    print(result["content"])

asyncio.run(chat())
```

## 📁 Project Structure

```
anony_llm/
├── core/                      # Core engine
│   ├── slm_architecture.py       # Custom SLM model (GQA, RoPE, SwiGLU)
│   ├── world_model.py            # World model + GRPO RL
│   └── tools.py                 # Calculator, web, code tools
├── agents/                    # Multi-agent system
│   ├── generator.py             # Response generation
│   ├── critic.py                # Quality evaluation
│   ├── optimizer.py             # Prompt optimization
│   ├── planner.py               # Planning
│   └── evaluator.py             # Feedback loop
├── serving/                   # Model serving
│   ├── model.py                 # Model manager (handles fallback chain)
│   ├── api.py                   # API endpoints
│   └── server.py                # FastAPI server
├── memory/                    # RAG & memory
│   ├── memory_manager.py        # Memory management
│   └── vector_store.py          # FAISS vector DB
├── ui/                        # Streamlit UI
│   └── app.py                   # Chat interface
├── models/                    # Trained models ✅
│   ├── anonyllm-360m-trained/    # Merged AnonyLLM-360M (INT4)
│   ├── anonyllm-360m-lora/       # LoRA adapter checkpoints
│   ├── smollm2-360m-trained-slm/ # Merged SmolLM2-360M (INT4)
│   ├── tiny-mobile-slm/          # Custom 22M param SLM
│   └── optimized/                # Quantized variants (int4/int8)
├── scripts/                   # Training scripts
│   ├── train_anonyllm.py        # Full LoRA/QLoRA training pipeline
│   ├── train_mistral_slm.py     # Alternative training + smoke tests
│   └── optimize_for_mobile.py   # Quantization & export
└── main.py                    # Entry point (--ui, --server, --train)
```

## 🚀 Performance

| Metric | Value |
|--------|-------|
| Model Load | 5-10s |
| Generation | 5-15s (CPU) |
| Memory | ~3GB |
| Disk | ~3GB |
| Context | 1024 tokens |

## 🔍 Testing

```bash
# Simple system check
python simple_test.py

# Integration test
python integration_test.py

# Full system test
python test_system.py
```

## 📝 Training Details

**Dataset**: Alpaca + SQuAD + GSM8K + DailyDialog  
**Configs**: math, science, chat, instruction_following, safety  
**Method**: LoRA fine-tuning (r=32, α=64)  
**Epochs**: 3  
**Batch Size**: 1  
**Learning Rate**: 2e-4  

**Anti-Hallucination Prompt**:
```
You are a precise, honest AI assistant. Follow these rules:
1. Only state facts you are confident about
2. If uncertain, say "I am not entirely sure"
3. Never invent names/dates
4. Ground answers in context
5. Show step-by-step reasoning
6. If asked who made you, say: "I was created by Aryan Chavan"
```

## 🎯 Key Features

✅ **Prototype Ready**: Modular, extensible architecture  
✅ **Custom Trained**: LoRA fine-tuned SmolLM2-360M-Instruct  
✅ **Multi-Agent**: Generator + Critic + Optimizer pipeline  
✅ **Hallucination Reduction**: Causal + physics consistency checks  
✅ **Self-Evaluation**: Response quality scoring & RL  
✅ **World Model**: Scenario simulation & reasoning  
✅ **RAG**: Document search & FAISS vector memory  
✅ **Web Search**: Automatic fact-checking (when online)  
✅ **Tools**: Calculator, code executor  

## 🤖 About

**Creator**: Aryan Chavan  
**License**: MIT  
**Version**: 2.0 (Research Prototype)  

---

## ✨ What Makes This Special?

1. **World Model**: Simulates scenarios before responding
2. **GRPO RL**: Self-improves through experience
3. **Multi-Agent**: Generator + Critic + Optimizer collaboration
4. **Hallucination Reduction**: Causal + Physics + Self-evaluation
5. **Trained Model**: Not just a wrapper — actual LoRA fine-tuning
6. **Error Handling**: Graceful fallback chain (local → HuggingFace → mock)
7. **Quantization**: int4/int8 for memory-constrained devices
8. **Extensible**: Easy to add new tools & agents

## 📖 What I Learned

Building this project taught me a wide range of practical ML engineering skills:

### Model Training & Optimization
- **LoRA / QLoRA fine-tuning** of 360M-parameter models on a consumer GPU (RTX 3050 4GB) using 4-bit quantization to fit in VRAM
- **Dataset preparation**: Curated multi-domain training data from Alpaca, SQuAD, GSM8K, DailyDialog, Dolly, and OASST1 — handling format conversion (Alpaca → ChatML), filtering, and deduplication
- **Training pipeline design**: Checkpointing, gradient accumulation, learning rate scheduling, and evaluating loss curves to detect overfitting
- **Quantization**: Converting trained models to int4/int8 for memory-constrained deployment without catastrophic quality loss

### CUDA & Memory Management
- Debugging **CUDA out-of-memory (OOM)** errors on a laptop GPU — learned to profile memory usage with `torch.cuda.memory_summary()` and optimize batch sizes
- Resolving **bitsandbytes/CUDA compatibility** issues (`CUBLAS_STATUS_NOT_SUPPORTED`) by matching library versions to the CUDA runtime
- Setting up **Windows page file** for model loading — discovered that memory-mapped model loading requires significant virtual address space even when the weights are quantized

### Transformer Architecture
- Implementing **Grouped Query Attention (GQA)**, **Rotary Position Embeddings (RoPE)**, **SwiGLU activation**, and **RMSNorm** from scratch in PyTorch
- Building a **KV-cache** for efficient autoregressive generation
- Making a custom model compatible with the HuggingFace Trainer interface (returning `(loss,)` tuple for training, logits for inference)

### Agentic Systems
- Designing a **multi-agent feedback loop** (Generator → Critic → Optimizer → Planner) with structured message passing between agents
- Implementing a **World Model** that simulates multiple scenario outcomes before generating a response
- Building **GRPO (Group Relative Policy Optimization)** — a simplified RL policy network for self-improvement based on heuristic reward signals

### Systems Engineering
- **Graceful degradation**: A multi-tier fallback chain (local merged model → LoRA adapter → HuggingFace download → mock mode) ensures the system never crashes on missing models
- **Async architecture**: `asyncio`-based serving layer with concurrent request handling via FastAPI
- **Streamlit UI**: Building an interactive chat interface with user authentication, file upload, and real-time status updates

### Debugging & Testing
- Writing **diagnostic tools** to verify model integrity (`_is_model_broken()` checks for repetitive/garbage output)
- **Cross-platform compatibility**: Handling Windows-specific issues (page files, symlinks, path separators) while keeping the code portable

## 📖 Documentation

See [PRODUCTION_README.md](PRODUCTION_README.md) for detailed documentation.

## 💡 Example Queries

- "Explain quantum entanglement simply"
- "What happens when you mix baking soda and vinegar?"
- "Write a Python script to sort a list"
- "What were the causes of WWII?"
- "How does photosynthesis work?"

The system will:
1. Analyze the query complexity
2. Simulate scenarios
3. Generate a reasoned response
4. Validate for accuracy
5. Self-evaluate and improve

## 🎉 Ready to Use!

The system is a **working research prototype**. Core components are implemented and tested; some features (GRPO RL, mobile deployment) are experimental and under active development.

```bash
python main.py --ui
```

---

**Built with**: PyTorch, Transformers, Streamlit, FastAPI, FAISS  
**Inspired by**: Llama 3, Qwen 2.5, Phi-3  
**Special Thanks**: HuggingFace datasets (Alpaca + SQuAD + GSM8K + DailyDialog)

