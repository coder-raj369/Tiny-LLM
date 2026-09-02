"""Checkpoint loading for the inference service."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from tiny_llm.config import ModelConfig
from tiny_llm.models import BPETokenizer, Transformer


class ModelLoadError(RuntimeError):
    """Raised when a model or tokenizer artifact cannot be loaded."""


class ModelLoader:
    """Load a model and tokenizer once, then reuse them for requests."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        model_config: Optional[ModelConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.model_config = model_config or ModelConfig()
        self.device = self._resolve_device(device)
        self.model: Optional[Transformer] = None
        self.tokenizer: Optional[BPETokenizer] = None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> Tuple[Transformer, BPETokenizer]:
        """Load artifacts and put the model in evaluation mode."""
        if self.is_loaded:
            return self.model, self.tokenizer  # type: ignore[return-value]
        if not self.checkpoint_path.exists():
            raise ModelLoadError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.tokenizer_path.exists():
            raise ModelLoadError(f"Tokenizer not found: {self.tokenizer_path}")

        checkpoint: Dict[str, Any] = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model = Transformer(
            vocab_size=self.model_config.vocab_size,
            embedding_dim=self.model_config.embedding_dim,
            num_layers=self.model_config.num_layers,
            num_heads=self.model_config.num_heads,
            context_length=self.model_config.context_length,
            dropout=self.model_config.dropout,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BPETokenizer(vocab_size=self.model_config.vocab_size)
        self.tokenizer.load(str(self.tokenizer_path))
        return self.model, self.tokenizer
