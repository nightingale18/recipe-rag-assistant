import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # File paths
    RECIPE_DIR: str = "recipe_uploads"
    VECTOR_STORE_PATH: str = "data/faiss_index.bin"
    RECIPES_DATA_PATH: str = "data/recipes.pkl"

    # Real-time settings
    CHECK_INTERVAL: int = 2  # seconds
    AUTO_SAVE: bool = True

    # Search settings
    TOP_K: int = 5
    SIM_THRESHOLD: float = 0.6

    # Cuisines list (your provided list)
    DEFAULT_CUISINES: List[str] = field(
        default_factory=lambda: [
            "Italian",
            "European",
            "Fusion",
            "American",
            "Asian",
            "Mexican",
            "Mediterranean",
            "Middle Eastern",
            "Nordic",
            "Indian",
        ]
    )

    def ensure_dirs(self):
        """Create necessary directories"""
        os.makedirs(self.RECIPE_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)
