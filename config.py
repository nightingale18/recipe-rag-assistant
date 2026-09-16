import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RECIPE_DIR: str = "data/recipe_uploads"
    DATA_DIR: str = "data"
    TOP_K: int = 5
    SIM_THRESHOLD: float = 0.5

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.RECIPE_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
