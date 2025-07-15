# parser_service_updated_prompts/__init__.py
import logging

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    if not logging.getLogger().hasHandlers():
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    else:
        logger.setLevel(logging.INFO)

from .pipeline import ProductParsingPipeline
from .models import ModelManager
from .config import (
    GEMMA_MODEL_ID,
    EXAONE_MODEL_ID,
    CLIP_MODEL_ID,
    YOLO_MODEL_PATH,
    DEVICE,
    DEFAULT_RESULTS_DIR
)

__all__ = [
    "ProductParsingPipeline",
    "ModelManager",
    "GEMMA_MODEL_ID",
    "EXAONE_MODEL_ID",
    "CLIP_MODEL_ID",
    "YOLO_MODEL_PATH",
    "DEVICE",
    "DEFAULT_RESULTS_DIR"
]

__version__ = "0.2.0" # Updated version 