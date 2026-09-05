"""
Configuration management for Tiny LLM.
Loads and validates all hyperparameters from config.yaml.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embedding_dim: int = 192
    num_layers: int = 4
    num_heads: int = 6
    context_length: int = 256
    vocab_size: int = 2000
    dropout: float = 0.1
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 40


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 20
    warmup_steps: int = 500
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    gradient_clip: float = 1.0
    checkpoint_every_steps: int = 500
    keep_last_n_checkpoints: int = 3
    early_stopping_patience: int = 3
    device: str = "cpu"
    mixed_precision: bool = False
    seed: int = 42


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_sequence_length: int = 256
    min_sequence_length: int = 10
    remove_duplicates: bool = True
    domain: str = "quantum_computing"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    compute_perplexity: bool = True
    compute_bleu: bool = False
    compute_semantic_similarity: bool = True
    compute_custom_metrics: bool = True
    qualitative_sample_size: int = 20
    rubric: tuple[str, ...] = ("correctness", "clarity", "coherence")
    compare_to_baseline: bool = True
    baseline_model: str = "distilgpt2"


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    backend: str = "mlflow"
    tracking_uri: str = "file://./mlruns"
    experiment_name: str = "tiny_llm_v1"
    log_metrics_every: int = 100
    log_params: bool = True
    log_artifacts: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 40
    sampling_strategy: str = "top_k"
    timeout_seconds: float = 5.0
    cache_model: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"


@dataclass
class Config:
    """Complete configuration object."""
    model: ModelConfig
    training: TrainConfig
    data: DataConfig
    evaluation: EvalConfig
    experiment_tracking: ExperimentConfig
    inference: InferenceConfig
    api: APIConfig
    logging: LoggingConfig
    paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "evaluation": asdict(self.evaluation),
            "experiment_tracking": asdict(self.experiment_tracking),
            "inference": asdict(self.inference),
            "api": asdict(self.api),
            "logging": asdict(self.logging),
            "paths": self.paths,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
    
    Returns:
        Config object with all settings.
    
    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config is invalid.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError("Config file is empty")
    
    try:
        data_config = DataConfig(**config_dict.get("data", {}))
        if abs(
            data_config.train_ratio + data_config.val_ratio + data_config.test_ratio - 1.0
        ) > 1e-6:
            raise ValueError("data split ratios must sum to 1.0")

        config = Config(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainConfig(**config_dict.get("training", {})),
            data=data_config,
            evaluation=EvalConfig(**config_dict.get("evaluation", {})),
            experiment_tracking=ExperimentConfig(**config_dict.get("experiment_tracking", {})),
            inference=InferenceConfig(**config_dict.get("inference", {})),
            api=APIConfig(**config_dict.get("api", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            paths=config_dict.get("paths", {})
        )
        return config
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid config format: {e}")


def get_config() -> Config:
    """
    Get global config instance.
    Convenience function; loads from default location.
    """
    return load_config()
