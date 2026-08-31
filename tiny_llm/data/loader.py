"""
Data loading utilities for Tiny LLM.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TinyLLMDataset(Dataset):
    """
    Dataset for Tiny LLM training.
    Loads Q&A pairs from JSONL format.
    """
    
    def __init__(
        self,
        filepath: str,
        tokenizer,
        context_length: int = 256,
        max_length: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            filepath: Path to JSONL file with Q&A pairs
            tokenizer: Tokenizer instance with encode() method
            context_length: Maximum sequence length
            max_length: Limit number of samples (for debugging)
        """
        self.filepath = Path(filepath)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.samples = []
        
        # Load data
        self._load_data(max_length)
    
    def _load_data(self, max_length: Optional[int] = None):
        """Load JSONL data from file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        with open(self.filepath, "r") as f:
            for i, line in enumerate(f):
                if max_length and i >= max_length:
                    break
                
                try:
                    data = json.loads(line)
                    self.samples.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {i}")
                    continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict with 'input_ids' and 'labels' tensors
        """
        sample = self.samples[idx]
        
        # Combine question and answer
        if "question" in sample and "answer" in sample:
            text = f"Q: {sample['question']}\nA: {sample['answer']}"
        elif "text" in sample:
            text = sample["text"]
        else:
            raise ValueError(f"Sample missing text fields: {sample.keys()}")
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(token_ids) > self.context_length:
            token_ids = token_ids[:self.context_length]
        else:
            token_ids = token_ids + [self.tokenizer.special_tokens["<pad>"]] * (
                self.context_length - len(token_ids)
            )
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (don't compute loss on padding)
        labels[input_ids == self.tokenizer.special_tokens["<pad>"]] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 32,
    context_length: int = 256,
    num_workers: int = 0,
) -> tuple:
    """
    Create train, val, and test dataloaders.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        test_path: Path to test JSONL file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        context_length: Maximum sequence length
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TinyLLMDataset(train_path, tokenizer, context_length)
    val_dataset = TinyLLMDataset(val_path, tokenizer, context_length)
    test_dataset = TinyLLMDataset(test_path, tokenizer, context_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
