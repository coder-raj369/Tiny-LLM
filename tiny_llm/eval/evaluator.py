"""
Evaluation utilities for Tiny LLM.
Computes metrics and generates evaluation reports.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_llm.logging_config import EVAL_LOGGER
from tiny_llm.constants import PROCESSED_DATA_DIR


class Evaluator:
    """Evaluator for model performance."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    @torch.no_grad()
    def compute_loss_and_perplexity(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Compute loss and perplexity on a dataset.
        
        Args:
            dataloader: Data loader to evaluate on
        
        Returns:
            (loss, perplexity)
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    @torch.no_grad()
    def generate_sample_predictions(
        self,
        tokenizer,
        test_prompts: List[str],
        max_tokens: int = 50,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """
        Generate sample predictions on test prompts.
        
        Args:
            tokenizer: Tokenizer instance
            test_prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            List of (prompt, generated_text) pairs
        """
        samples = []
        
        for prompt in test_prompts:
            # Encode prompt
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            token_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            # Generate
            try:
                generated_ids = self.model.generate(
                    token_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=40,
                    eos_token_id=3,
                )
                
                # Decode
                generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
                
                samples.append({
                    "prompt": prompt,
                    "generated": generated_text,
                })
            except Exception as e:
                EVAL_LOGGER.warning(f"Generation failed for prompt '{prompt}': {e}")
        
        return samples
    
    def generate_report(
        self,
        test_dataloader: DataLoader,
        tokenizer,
        test_prompts: Optional[List[str]] = None,
        output_path: str = "eval/eval_report.json",
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_dataloader: Test set data loader
            tokenizer: Tokenizer instance
            test_prompts: Optional list of prompts for sample generation
            output_path: Path to save report
        
        Returns:
            Evaluation report dictionary
        """
        EVAL_LOGGER.info("Generating evaluation report...")
        
        # Compute metrics
        test_loss, test_perplexity = self.compute_loss_and_perplexity(test_dataloader)
        
        report = {
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "timestamp": str(Path(__file__).parent),
        }
        
        # Generate samples
        if test_prompts:
            samples = self.generate_sample_predictions(tokenizer, test_prompts)
            report["sample_predictions"] = samples
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        EVAL_LOGGER.info(f"Evaluation report saved to {output_path}")
        EVAL_LOGGER.info(f"Test Loss: {test_loss:.4f}")
        EVAL_LOGGER.info(f"Test Perplexity: {test_perplexity:.4f}")
        
        return report
