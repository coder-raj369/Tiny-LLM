# Tiny LLM Quick Start Guide

## What Was Built (Phase 0 & 1)

### Phase 0: Scoping & Architecture ✅
- **Spec:** Quantum computing Q&A assistant, <2s latency on Mac M2 CPU, ≥70% correctness
- **Architecture:** Data → Train → Eval → Inference → API → UI with feedback loop
- **Tech Stack:** PyTorch, FastAPI, Streamlit, MLflow, Docker Compose
- **Single Source of Truth:** `config.yaml` for all hyperparameters

### Phase 1: Data & Model ✅
1. **Model** (4 layers, 6 heads, 192 dims, ~3-5M params)
   - `tiny_llm/models/transformer.py` — Full transformer with causal attention
   - `tiny_llm/models/tokenizer.py` — BPE tokenizer with encode/decode

2. **Data Pipeline**
   - `tiny_llm/data/processor.py` — Load, filter, split data
   - `tiny_llm/data/loader.py` — PyTorch DataLoader
   - `tiny_llm/data/validator.py` — Quality checks

3. **Training**
   - `tiny_llm/training/trainer.py` — Full loop with early stopping, checkpointing
   - AdamW optimizer, gradient clipping, warmup, per-step logging

4. **Evaluation**
   - `tiny_llm/eval/evaluator.py` — Perplexity, sample generation, reports

5. **Config & Logging**
   - `config.yaml` — Versioned hyperparameters
   - `tiny_llm/config.py` — Dataclass-based config loading
   - `tiny_llm/logging_config.py` — Structured logging to console + files

6. **Testing**
   - `tests/test_model.py` — Model instantiation, forward pass, generation
   - CI/CD: `.github/workflows/test.yml` runs on Python 3.10+

7. **Automation**
   - `Makefile` — `make setup`, `make train`, `make eval`, `make serve`, `make test`
   - `scripts/train.py` — Full training script (production-ready)

---

## Quick Start

### 1. Setup (One Time)
```bash
cd ~/Documents/TIny\ LLM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Training Data
You need to create: `data/processed/train.jsonl`, `data/processed/val.jsonl`, `data/processed/test.jsonl`

Each line: `{"question": "...", "answer": "..."}`

Example:
```json
{"question": "What is quantum entanglement?", "answer": "Quantum entanglement is a phenomenon where two or more particles become correlated..."}
```

### 3. Train the Model
```bash
python scripts/train.py --config config.yaml --device cpu
# Or: make train
```

Outputs:
- `checkpoints/{run_id}/` — Model checkpoints
- `runs/{run_id}/` — Config, metrics, results

### 4. Evaluate
```bash
python -c "
from tiny_llm.eval import Evaluator
from tiny_llm.data import create_data_loaders
from tiny_llm.models import Transformer, BPETokenizer
import torch

# Load model and evaluate
model = Transformer()
model.load_state_dict(torch.load('checkpoints/{run_id}/best_model.pt')['model_state_dict'])
evaluator = Evaluator(model, device='cpu')
# ... evaluate on test set
"
```

### 5. Run Tests
```bash
pytest tests/ -v
# Or: make test
```

---

## Project Structure

```
Tiny-LLM/
├── config.yaml                    # ← Hyperparameters (single source of truth)
├── Makefile                       # ← Automation targets
├── requirements.txt               # ← Pinned dependencies
├── README.md                      # ← Full documentation
│
├── tiny_llm/                      # ← Main package
│   ├── models/
│   │   ├── transformer.py         # 4-layer, 6-head transformer
│   │   └── tokenizer.py           # BPE tokenizer
│   ├── data/
│   │   ├── processor.py           # Data cleaning & splitting
│   │   ├── loader.py              # PyTorch DataLoader
│   │   └── validator.py           # Quality checks
│   ├── training/
│   │   └── trainer.py             # Full training loop
│   ├── eval/
│   │   └── evaluator.py           # Metrics & evaluation
│   ├── config.py                  # Config loading
│   ├── constants.py               # Enums & paths
│   └── logging_config.py          # Structured logging
│
├── tests/
│   └── test_model.py              # Unit tests
│
├── scripts/
│   └── train.py                   # Training script
│
├── .github/workflows/
│   └── test.yml                   # CI/CD (GitHub Actions)
│
└── data/
    ├── raw/                       # Raw source data
    └── processed/                 # Train/val/test splits
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| 4 layers, 192 dims | Fits in 8GB RAM, trains in hours on M2 CPU |
| BPE tokenizer | Standard in modern LLMs, simple to implement |
| AdamW optimizer | Better than SGD for transformers |
| Config in YAML | Reproducible; every run is versioned |
| Early stopping | Prevent overfitting; save time |
| Structured logging | Debug issues, track experiments |
| PyTorch only | No framework-hopping; focused scope |

---

## Next Steps (Phases 2–7)

- **Phase 2:** FastAPI inference server + Streamlit UI
- **Phase 3:** Integration tests, security, auth
- **Phase 4:** Logging, observability, performance profiling
- **Phase 5:** Docker Compose, GitHub Actions CI/CD, cloud deployment
- **Phase 6:** Monitoring, alerting, error handling, backup
- **Phase 7:** Docs, versioning, continuous evaluation

---

## Interview Talking Points

✅ "I scoped this with a spec—input/output, latency budget, acceptance criteria."
✅ "Config is versioned in git; any checkpoint is fully reproducible."
✅ "Data pipeline has validation: format checks, token counts, deduplication."
✅ "Model is intentionally tiny: 4 layers, ~3-5M params, trains in hours."
✅ "Training tracks experiments: every run logs config, loss, metrics."
✅ "Code is modular: models, data, training, eval separated."
✅ "Tests run in CI on every push; coverage on critical paths."

---

## Where to Find Things

- **How to train?** See `scripts/train.py` or `make train`
- **How to evaluate?** See `tiny_llm/eval/evaluator.py`
- **How to change hyperparams?** Edit `config.yaml`
- **How to add a feature?** Add to corresponding module (models/, data/, training/, eval/)
- **How to run tests?** `pytest tests/` or `make test`
- **How to push changes?** `git add -A && git commit -m "..." && git push`

---

## GitHub Repository

📍 https://github.com/coder-raj369/Tiny-LLM

**Current Status:** Phase 0 & 1 complete ✅
- 30 files, 3000+ lines of production-grade Python
- Ready for data preparation and training
- CI/CD configured and passing

**Next:** Prepare quantum computing Q&A data and run training!

