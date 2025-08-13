"""
Utility functions model selection.
"""

import os

from functools import lru_cache
from tensorflow.keras.models import load_model

from app.config import MODEL_FILE_FORMAT


@lru_cache(maxsize=3)
def load_model_variant(variant: str = 'light', path = None):
    if path is None:
        here = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(here, MODEL_FILE_FORMAT.format(variant))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model variant '{variant}' not found at {path}")

    return load_model(path, compile=False)