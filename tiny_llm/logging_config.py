"""
Logging configuration for Tiny LLM.
Sets up structured logging for all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from tiny_llm.constants import LOGS_DIR, LOG_FORMAT


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger for a module.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to log to (in logs/ directory)
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = LOGS_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global loggers
TRAIN_LOGGER = setup_logging("tiny_llm.training", level="INFO", log_file="train.log")
EVAL_LOGGER = setup_logging("tiny_llm.evaluation", level="INFO", log_file="eval.log")
DATA_LOGGER = setup_logging("tiny_llm.data", level="INFO", log_file="data.log")
INFERENCE_LOGGER = setup_logging("tiny_llm.inference", level="INFO", log_file="inference.log")
API_LOGGER = setup_logging("tiny_llm.api", level="INFO", log_file="api.log")
