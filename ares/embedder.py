"""
Embedding layer using sentence-transformers.
"""
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed(text: str) -> list[float]:
    return get_model().encode(text).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts).tolist()