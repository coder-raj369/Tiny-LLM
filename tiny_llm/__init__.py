"""
Tiny LLM: Production-grade domain-specific LLM system.
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Raj Pandit"

# Package-level imports (optional, for convenience)
from tiny_llm.config import load_config, TrainConfig, ModelConfig

__all__ = ["load_config", "TrainConfig", "ModelConfig"]
