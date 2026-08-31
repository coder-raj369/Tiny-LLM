#!/usr/bin/env python
"""
Sample training script for Tiny LLM.
This demonstrates how to train the model end-to-end.

Usage:
    python scripts/train.py [--config config.yaml]

Make sure you have prepared training data in:
    data/processed/train.jsonl
    data/processed/val.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from tiny_llm.config import load_config
from tiny_llm.constants import CHECKPOINT_DIR, PROCESSED_DATA_DIR
from tiny_llm.models import Transformer, BPETokenizer
from tiny_llm.data import TinyLLMDataset, create_data_loaders
from tiny_llm.training import Trainer
from tiny_llm.logging_config import TRAIN_LOGGER


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Tiny LLM")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    TRAIN_LOGGER.info(f"Loaded config from {args.config}")
    TRAIN_LOGGER.info(f"Model: {config.model.num_layers} layers, {config.model.embedding_dim} dims")
    TRAIN_LOGGER.info(f"Training: {config.training.epochs} epochs, lr={config.training.learning_rate}")
    
    # Determine device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
        TRAIN_LOGGER.info("Using Apple Silicon (MPS)")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        TRAIN_LOGGER.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        TRAIN_LOGGER.info("Using CPU")
    
    # Initialize tokenizer
    TRAIN_LOGGER.info("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.model.vocab_size)
    
    # TODO: Train tokenizer on actual data
    # For now, use dummy training
    tokenizer.train(
        ["quantum computing"] * 10,
        num_merges=config.model.vocab_size - 100
    )
    TRAIN_LOGGER.info(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Check for data files
    train_path = Path(config.data.processed_dir) / "train.jsonl"
    val_path = Path(config.data.processed_dir) / "val.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        TRAIN_LOGGER.error(f"Data files not found!")
        TRAIN_LOGGER.error(f"  Expected: {train_path} and {val_path}")
        TRAIN_LOGGER.error(f"  Please run 'make data-process' first")
        return
    
    # Create data loaders
    TRAIN_LOGGER.info("Creating data loaders...")
    train_loader, val_loader, _ = create_data_loaders(
        str(train_path),
        str(val_path),
        str(Path(config.data.processed_dir) / "test.jsonl"),
        tokenizer,
        batch_size=config.training.batch_size,
        context_length=config.model.context_length,
    )
    
    TRAIN_LOGGER.info(f"Train batches: {len(train_loader)}")
    TRAIN_LOGGER.info(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    TRAIN_LOGGER.info("Initializing model...")
    model = Transformer(
        vocab_size=config.model.vocab_size,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        context_length=config.model.context_length,
        dropout=config.model.dropout,
    )
    
    model_info = model.get_model_info()
    TRAIN_LOGGER.info(f"Model parameters: {model_info['total_parameters_millions']:.2f}M")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=device,
    )
    
    # Train
    results = trainer.train()
    
    TRAIN_LOGGER.info("Training complete!")
    TRAIN_LOGGER.info(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
