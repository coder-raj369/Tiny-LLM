# Tiny LLM: Production-Grade Quantum Computing Q&A Assistant

A deliberately small, domain-specialized transformer LLM trained to answer quantum computing questions. This project demonstrates end-to-end ML systems engineering: from data collection through production deployment.

**Target Stack:** Mac M2 / 8GB RAM | **Solo Dev Timeline:** 8–12 weeks | **Portfolio Focus:** Systems & MLOps maturity, not model scale

---

## Project Overview

### What This Is
- A 4-layer, 6-head transformer (~3–5MB parameters) trained on curated quantum computing Q&A data.
- A complete ML pipeline: data versioning, training with experiment tracking, evaluation with custom metrics, and API serving.
- Production-ready infrastructure: Docker, CI/CD, structured logging, reproducible configs.

### What This Isn't
- A large language model; by design, this is tiny.
- A general-purpose chatbot; specialized in quantum computing.
- A research project; engineered for clarity and iteration, not novelty.

### Design Philosophy
*Small scope, shipping-focused, production-adjacent.*
- Model size: Fits in ~500MB container, runs inference in <200ms on CPU.
- Infrastructure: Docker Compose (not Kubernetes), GitHub Actions (not enterprise CI), SQLite-friendly.
- Success metric: Show an interviewer a working system, not a paper.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Training Pipeline                  │
│  Data → Preprocess → Train → Eval → Experiment Log  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓ (best checkpoint)
            ┌──────────────┐
            │ Model Registry
            │ (GitHub Releases)
            └──────┬───────┘
                   │
                   ↓
      ┌────────────────────────┐
      │  Inference Server      │
      │  (FastAPI)             │
      │  ✓ Load checkpoint     │
      │  ✓ Cache in memory     │
      │  ✓ Structure logging   │
      └────────┬───────────────┘
               │
      ┌────────┴──────────┐
      │                   │
      ↓                   ↓
   REST API          Streamlit UI
   (FastAPI)        (Frontend Demo)
   
   Feedback → Evaluation → Re-train
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip / venv
- Git
- Docker (optional, for containerization)

### Quick Start

```bash
# Clone and navigate
git clone https://github.com/coder-raj369/Tiny-LLM.git
cd Tiny-LLM

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
make help
```

### Full Setup (All Extras)
```bash
make install-dev
```

---

## Workflow

### 1. Data Preparation
```bash
# Download raw training data (from curated sources)
make data-download

# Process, validate, and split (train/val/test)
make data-process

# Check: data/processed/{train,val,test}.jsonl created
ls data/processed/
```

### 2. Training
```bash
# Train with default config (config.yaml)
make train

# Logs: runs/{timestamp}/ contains config, metrics, checkpoint

# View runs
ls runs/
```

### 3. Evaluation
```bash
# Evaluate on test set
make eval

# Output: eval/eval_report.json with metrics + qualitative scores
```

### 4. API Serving
```bash
# Start FastAPI server
make serve

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Frontend Demo
```bash
# Start Streamlit UI (separate terminal)
make frontend

# UI available at http://localhost:8501
```

---

## Project Structure

```
Tiny-LLM/
├── config.yaml                      # Single source of truth for hyperparams
├── Makefile                         # Task automation
├── README.md                        # This file
├── LICENSE                          # MIT
├── requirements.txt                 # Dependencies (pinned versions)
├── pyproject.toml                   # Project metadata + extras
│
├── tiny_llm/                        # Main package
│   ├── __init__.py
│   ├── config.py                    # Config loading & dataclasses
│   ├── constants.py                 # Domain constants, paths
│   │
│   ├── models/                      # Model definitions
│   │   ├── __init__.py
│   │   ├── transformer.py           # Core transformer architecture
│   │   ├── tokenizer.py             # BPE tokenizer
│   │   └── utils.py                 # Model utilities (param counting, etc.)
│   │
│   ├── data/                        # Data pipeline
│   │   ├── __init__.py
│   │   ├── downloader.py            # Fetch raw data from sources
│   │   ├── processor.py             # Clean, validate, split data
│   │   ├── loader.py                # PyTorch DataLoader
│   │   └── validator.py             # Data quality checks
│   │
│   ├── training/                    # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── optimizer.py             # Optimizer setup
│   │   └── checkpointing.py         # Save/load checkpoints
│   │
│   ├── eval/                        # Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py               # Perplexity, BLEU, semantic sim, etc.
│   │   ├── evaluator.py             # Run eval pipeline
│   │   └── qualitative.py           # Manual scoring utils
│   │
│   ├── inference/                   # Inference engine
│   │   ├── __init__.py
│   │   ├── model_loader.py          # Load checkpoint + cache
│   │   ├── generator.py             # Generate text (sampling, etc.)
│   │   └── utils.py                 # Tokenization, decoding
│   │
│   ├── api/                         # REST API
│   │   ├── __init__.py
│   │   ├── server.py                # FastAPI app
│   │   ├── schemas.py               # Pydantic models
│   │   └── middleware.py            # Logging, error handling
│   │
│   └── logging_config.py            # Structured logging setup
│
├── frontend/                        # Streamlit UI
│   ├── __init__.py
│   └── app.py                       # Main UI code
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_model.py                # Model tests
│   ├── test_training.py             # Training pipeline tests
│   ├── test_data.py                 # Data loading tests
│   ├── test_inference.py            # Inference tests
│   └── test_api.py                  # API endpoint tests
│
├── data/
│   ├── raw/                         # Raw source data (gitignored)
│   ├── processed/                   # Processed splits (gitignored, versioned with DVC)
│   └── validation_report.json       # Data quality report
│
├── checkpoints/                     # Model checkpoints (gitignored)
│   ├── best_model.pt
│   └── {run_id}/epoch_*.pt
│
├── runs/                            # Training run logs (gitignored, versioned)
│   ├── {timestamp}_config.json
│   ├── {timestamp}_metrics.json
│   └── {timestamp}_eval_report.json
│
├── experiments/                     # Experiment tracking
│   ├── results_summary.csv          # Run comparison table
│   └── mlruns/                      # MLflow artifacts (if using MLflow)
│
├── logs/                            # Application logs
│   ├── train.log
│   └── inference.log
│
├── .github/workflows/
│   ├── test.yml                     # CI: Run tests on every push
│   └── lint.yml                     # Linting checks
│
└── .gitignore                       # Standard Python + custom patterns
```

---

## Key Components (Phase 0–1)

### Phase 0: Scoping & Architecture

**Spec:**
- **Input:** Quantum computing question (≤256 tokens)
- **Output:** Coherent, factually correct answer (≤512 tokens)
- **Acceptance:** ≥70% correctness on 50–100 held-out Q&A pairs
- **Deployment:** REST API + Streamlit UI, <2s latency on CPU
- **Model:** 4 layers, 6 heads, 192 dims, 256 context, ~3–5MB

**See:** [config.yaml](config.yaml) (single source of truth for hyperparams)

---

### Phase 1: Data & Model

#### Data Pipeline
- **Sources:** Textbooks, arXiv abstracts, Stack Exchange, curated Q&A pairs, licensed datasets
- **Licensing:** All data verified for CC-BY or similar; sources documented
- **Versioning:** DVC (optional) or Git LFS for reproducibility
- **Splits:** 70% train, 15% val, 15% test (fixed seed)
- **Validation:** Format checks, token counts, deduplication, OOV analysis

**Files:**
- [tiny_llm/data/downloader.py](tiny_llm/data/downloader.py) — Fetch data
- [tiny_llm/data/processor.py](tiny_llm/data/processor.py) — Clean & split
- [tiny_llm/data/validator.py](tiny_llm/data/validator.py) — Quality assurance

#### Model
- **Architecture:** 4 transformer layers, 6 attention heads, 192 embedding dims
- **Context:** 256 tokens
- **Vocab:** BPE tokenizer, 2000 tokens
- **Parameters:** ~3–5MB
- **Training:** Adam, warmup, gradient clipping, early stopping

**Files:**
- [tiny_llm/models/transformer.py](tiny_llm/models/transformer.py) — Transformer definition
- [tiny_llm/models/tokenizer.py](tiny_llm/models/tokenizer.py) — BPE tokenizer

#### Training Pipeline
- **Config:** [config.yaml](config.yaml) (versioned, reproducible)
- **Checkpointing:** Epoch-based + best model tracking
- **Experiment Tracking:** MLflow (local, free) or JSON logging
- **Logging:** Structured, per-step metrics

**Files:**
- [tiny_llm/training/trainer.py](tiny_llm/training/trainer.py) — Main loop
- [tiny_llm/training/optimizer.py](tiny_llm/training/optimizer.py) — AdamW setup
- [tiny_llm/training/checkpointing.py](tiny_llm/training/checkpointing.py) — Save/load

#### Evaluation
- **Metrics:** Perplexity, semantic similarity (if reference answers), custom domain metrics
- **Qualitative:** Hand-scored correctness/clarity/coherence rubric (~20–50 examples)
- **Failure Analysis:** Root-cause categorization of errors
- **Baseline Comparison:** vs. DistilGPT-2 or 2-layer baseline

**Files:**
- [tiny_llm/eval/metrics.py](tiny_llm/eval/metrics.py) — Metric implementations
- [tiny_llm/eval/evaluator.py](tiny_llm/eval/evaluator.py) — Eval pipeline
- [eval/eval_report.json](eval/eval_report.json) — Results (example)

#### Experiment Tracking
- **Backend:** MLflow (or JSON if preferred)
- **Logged:** Config, hyperparams, train/val/test loss, eval metrics, git commit hash
- **Comparison:** results_summary.csv (all runs ranked)
- **Best Model:** Pinned for inference

**Files:**
- [experiments/results_summary.csv](experiments/results_summary.csv) — Run comparison

---

## Reproducibility

Every training run is reproducible:

1. **Config is versioned:** [config.yaml](config.yaml) in git
2. **Data split is seeded:** `seed: 42` in config
3. **Checkpoint includes config:** Load any checkpoint, inspect its exact hyperparams
4. **Git commit hash logged:** Know exactly what code produced it

Example:
```bash
# Load run from 3 weeks ago
git checkout abc123
python -m tiny_llm.training.train

# Or load a specific checkpoint
python -c "import torch; ckpt = torch.load('checkpoints/run_001/epoch_15.pt'); print(ckpt['config'])"
```

---

## Running Tests

```bash
# Unit tests
make test

# Linting & type checking
make lint

# Format code
make format
```

**Coverage Target:** 70%+ on critical modules (model, training, inference)

---

## Deployment (Phase 5+)

### Local Dev
```bash
make serve  # API on localhost:8000
make frontend  # UI on localhost:8501
```

### Docker
```bash
docker build -t tiny-llm .
docker run -p 8000:8000 tiny-llm
```

### Docker Compose (API + Streamlit)
```bash
docker-compose up
```

---

## Interview Talking Points

1. **"I scoped the project in a spec."** One-pager: exact input/output, acceptance criteria, latency budget, deployment target. No surprises later.

2. **"Data is the bottleneck."** Sourced from 3 licensed sources, deduplicated, validated, split with fixed seed. Can reproduce exact dataset from git.

3. **"Config is king."** Hyperparams live in `config.yaml`, not code. Every run logs its config; any checkpoint is reproducible.

4. **"I tracked experiments systematically."** 5+ runs logged (baseline, different LR, different depth). Comparison table shows which was best and why.

5. **"Model is intentionally tiny."** 4 layers, 192 dims—I measured it fits in 8GB RAM, trains in hours. Chose PyTorch because I know it; no framework-hopping for resume.

6. **"Eval is more than loss numbers."** Perplexity + semantic similarity + manual rubric (correctness/clarity/coherence). Failure analysis: 3 hallucinations, 2 OOD. Not just "loss = 2.5."

7. **"API & serving are tested."** FastAPI with auto-docs, health checks, structured logging. Tests run in CI on every commit.

8. **"Reproducibility is built in."** Fixed seed, pinned versions in requirements.txt, checkpoints include config. Interviewer can clone and train in one command.

---

## Next Steps (Phase 2–7)

- **Phase 2:** Inference server, API, Streamlit frontend
- **Phase 3:** Testing, security, config management
- **Phase 4:** Logging, observability
- **Phase 5:** Docker Compose, CI/CD, cloud deployment (DigitalOcean)
- **Phase 6:** Monitoring, alerting, error handling
- **Phase 7:** Versioning, docs, continuous evaluation

---

## Contributing & Development

See [DEVELOPMENT.md](DEVELOPMENT.md) (forthcoming) for contributor guidelines.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Questions?

- **Model questions:** See [models/README.md](models/README.md)
- **Data questions:** See [data/README.md](data/README.md)
- **API questions:** See [api/README.md](api/README.md)

---

**Built with production rigor on a shoestring (M2 + 8GB + free tools).**
