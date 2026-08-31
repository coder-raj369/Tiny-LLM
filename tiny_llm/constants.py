"""
Constants and utilities for Tiny LLM.
"""

from enum import Enum
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Key paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOGS_DIR, RUNS_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Domain(str, Enum):
    """Supported domains."""
    QUANTUM_COMPUTING = "quantum_computing"


class DataSplit(str, Enum):
    """Data splits."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SamplingStrategy(str, Enum):
    """Text generation sampling strategies."""
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


# Special tokens
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,  # Beginning of sequence
    "<eos>": 3,  # End of sequence
}

# Tokenizer info
BPE_VOCAB_SIZE = 2000
BPE_MERGE_OPERATIONS = 2000

# Model constraints (for validation)
MIN_EMBEDDING_DIM = 64
MAX_EMBEDDING_DIM = 512
MIN_NUM_LAYERS = 1
MAX_NUM_LAYERS = 12
MIN_NUM_HEADS = 1
MAX_NUM_HEADS = 16
MIN_CONTEXT_LENGTH = 128
MAX_CONTEXT_LENGTH = 4096

# Training constraints
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-1
MIN_EPOCHS = 1
MAX_EPOCHS = 100

# Timing/performance
INFERENCE_TIMEOUT_SECONDS = 5.0
MAX_TOKENS_PER_REQUEST = 512

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
