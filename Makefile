.PHONY: help setup install install-dev lint format test train eval serve clean data-download data-process

help:
	@echo "Tiny LLM - Production ML Project"
	@echo "================================"
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup              - Create virtual environment and install dependencies"
	@echo "  make install            - Install core dependencies"
	@echo "  make install-dev        - Install all dependencies (dev + extras)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint               - Run linting checks (ruff + mypy)"
	@echo "  make format             - Format code with black"
	@echo "  make test               - Run pytest suite"
	@echo ""
	@echo "Data & Training:"
	@echo "  make data-download      - Download raw training data"
	@echo "  make data-process       - Process and validate data"
	@echo "  make train              - Train the model (full pipeline)"
	@echo "  make eval               - Evaluate model on test set"
	@echo ""
	@echo "Serving & Demo:"
	@echo "  make serve              - Start FastAPI server"
	@echo "  make frontend           - Start Streamlit UI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove cache files, logs, temp files"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete! Run: source venv/bin/activate"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,api,frontend,mlops,data]"

lint:
	python -m ruff check .
	python -m mypy tiny_llm --ignore-missing-imports

format:
	python -m black .

test:
	python -m pytest tests/ -v --cov=tiny_llm --cov-report=term-missing

data-download:
	python -m tiny_llm.data.downloader

data-process:
	python -m tiny_llm.data.processor

train:
	python -m tiny_llm.training.train

eval:
	python -m tiny_llm.eval.evaluator

serve:
	uvicorn tiny_llm.api.server:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf tmp/ temp/ *.tmp
	@echo "Clean complete"

.DEFAULT_GOAL := help
