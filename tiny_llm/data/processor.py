"""
Data processing utilities for Tiny LLM.
Handles data cleaning, splitting, and preparation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from tiny_llm.logging_config import DATA_LOGGER


class DataProcessor:
    """Process and prepare data for training."""
    
    def __init__(self, seed: int = 42):
        """Initialize processor with random seed."""
        self.seed = seed
        random.seed(seed)
    
    @staticmethod
    def load_qa_pairs(filepath: str) -> List[Dict[str, str]]:
        """
        Load Q&A pairs from JSONL file.
        
        Expected format: {"question": "...", "answer": "..."}
        """
        pairs = []
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if "question" in data and "answer" in data:
                        pairs.append(data)
                except json.JSONDecodeError as e:
                    DATA_LOGGER.warning(f"Line {line_num}: Could not parse JSON: {e}")
        
        DATA_LOGGER.info(f"Loaded {len(pairs)} Q&A pairs from {filepath}")
        return pairs
    
    @staticmethod
    def filter_invalid_pairs(
        pairs: List[Dict[str, str]],
        min_length: int = 5,
        max_question_length: int = 256,
        max_answer_length: int = 512,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Filter out invalid Q&A pairs.
        
        Returns:
            (valid_pairs, invalid_pairs)
        """
        valid = []
        invalid = []
        
        for pair in pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            
            reasons = []
            
            if len(q) < min_length:
                reasons.append(f"Q too short ({len(q)} < {min_length})")
            if len(a) < min_length:
                reasons.append(f"A too short ({len(a)} < {min_length})")
            if len(q) > max_question_length:
                reasons.append(f"Q too long ({len(q)} > {max_question_length})")
            if len(a) > max_answer_length:
                reasons.append(f"A too long ({len(a)} > {max_answer_length})")
            
            if reasons:
                pair["invalid_reasons"] = "; ".join(reasons)
                invalid.append(pair)
            else:
                valid.append(pair)
        
        DATA_LOGGER.info(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        return valid, invalid
    
    @staticmethod
    def remove_duplicates(pairs: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
        """
        Remove duplicate Q&A pairs.
        
        Returns:
            (unique_pairs, num_duplicates_removed)
        """
        seen = set()
        unique = []
        duplicates = 0
        
        for pair in pairs:
            key = (pair["question"].lower(), pair["answer"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(pair)
            else:
                duplicates += 1
        
        DATA_LOGGER.info(f"Removed {duplicates} duplicate pairs. Remaining: {len(unique)}")
        return unique, duplicates
    
    @staticmethod
    def split_data(
        pairs: List[Dict[str, str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/val/test.
        
        Args:
            pairs: List of Q&A pairs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
        
        Returns:
            (train_pairs, val_pairs, test_pairs)
        """
        random.seed(seed)
        random.shuffle(pairs)
        
        n = len(pairs)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train = pairs[:train_size]
        val = pairs[train_size:train_size + val_size]
        test = pairs[train_size + val_size:]
        
        DATA_LOGGER.info(
            f"Split: train={len(train)} ({100*len(train)/n:.1f}%), "
            f"val={len(val)} ({100*len(val)/n:.1f}%), "
            f"test={len(test)} ({100*len(test)/n:.1f}%)"
        )
        
        return train, val, test
    
    @staticmethod
    def save_qa_pairs(pairs: List[Dict[str, str]], output_path: str):
        """Save Q&A pairs to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        
        DATA_LOGGER.info(f"Saved {len(pairs)} pairs to {output_path}")
    
    @staticmethod
    def compute_statistics(pairs: List[Dict[str, str]]) -> Dict:
        """Compute statistics on Q&A pairs."""
        q_lengths = [len(p["question"].split()) for p in pairs]
        a_lengths = [len(p["answer"].split()) for p in pairs]
        
        stats = {
            "num_pairs": len(pairs),
            "avg_question_words": sum(q_lengths) / len(q_lengths) if q_lengths else 0,
            "max_question_words": max(q_lengths) if q_lengths else 0,
            "min_question_words": min(q_lengths) if q_lengths else 0,
            "avg_answer_words": sum(a_lengths) / len(a_lengths) if a_lengths else 0,
            "max_answer_words": max(a_lengths) if a_lengths else 0,
            "min_answer_words": min(a_lengths) if a_lengths else 0,
        }
        
        return stats
    
    @staticmethod
    def print_statistics(pairs: List[Dict[str, str]], name: str = "Data"):
        """Print statistics on Q&A pairs."""
        stats = DataProcessor.compute_statistics(pairs)
        
        DATA_LOGGER.info(f"\n{name} Statistics:")
        DATA_LOGGER.info(f"  Pairs: {stats['num_pairs']}")
        DATA_LOGGER.info(f"  Question words: avg={stats['avg_question_words']:.1f}, "
                        f"min={stats['min_question_words']}, max={stats['max_question_words']}")
        DATA_LOGGER.info(f"  Answer words: avg={stats['avg_answer_words']:.1f}, "
                        f"min={stats['min_answer_words']}, max={stats['max_answer_words']}")
        
        return stats
