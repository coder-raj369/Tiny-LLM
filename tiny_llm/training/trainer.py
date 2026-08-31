"""
Main training loop for Tiny LLM.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tiny_llm.config import TrainConfig, ModelConfig
from tiny_llm.logging_config import TRAIN_LOGGER
from tiny_llm.constants import CHECKPOINT_DIR, RUNS_DIR


class Trainer:
    """Trainer class for Tiny LLM."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: str = "cpu",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = CHECKPOINT_DIR / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
        TRAIN_LOGGER.info(f"Trainer initialized. Run ID: {self.run_id}")
    
    def _warmup_learning_rate(self, step: int) -> float:
        """Linear warmup scheduler."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / self.config.warmup_steps)
        return self.config.learning_rate
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Warmup learning rate
            step = epoch * len(self.train_loader) + batch_idx
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._warmup_learning_rate(step)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % self.config.log_metrics_every == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_loss = total_loss / num_batches
                TRAIN_LOGGER.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f}"
                )
        
        avg_train_loss = total_loss / num_batches
        self.train_losses.append(avg_train_loss)
        return avg_train_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:02d}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "val_loss": self.val_losses[-1] if self.val_losses else None,
            "config": self.config.__dict__,
        }
        
        torch.save(checkpoint, checkpoint_path)
        TRAIN_LOGGER.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            TRAIN_LOGGER.info(f"Best model saved: {best_path}")
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            Training results dictionary
        """
        TRAIN_LOGGER.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            TRAIN_LOGGER.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                TRAIN_LOGGER.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        elapsed_time = time.time() - start_time
        
        # Save training results
        results = {
            "run_id": self.run_id,
            "epochs_trained": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_hours": elapsed_time / 3600,
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint": str(self.checkpoint_dir / "best_model.pt"),
        }
        
        # Save config and results
        config_path = self.run_dir / f"{self.run_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        results_path = self.run_dir / f"{self.run_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        TRAIN_LOGGER.info(f"Training completed in {elapsed_time/3600:.2f} hours")
        TRAIN_LOGGER.info(f"Best validation loss: {self.best_val_loss:.4f}")
        TRAIN_LOGGER.info(f"Results saved to {results_path}")
        
        return results
