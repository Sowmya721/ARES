"""
Endee client wrapper - uses REST API directly with correct payload format.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8080/api/v1"
AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
CHECKSUM = -1

def _headers():
    return {"Authorization": AUTH_TOKEN, "Content-Type": "application/json"}

def create_index(index_name: str, dim: int):
    data = {
        "index_name": index_name,
        "dim": dim,
        "space_type": "cosine",
        "M": 16,
        "ef_con": 128,
        "checksum": CHECKSUM,
        "precision": "int8d",
        "version": None
    }
    r = requests.post(f"{BASE_URL}/index/create", json=data, headers=_headers())
    if r.status_code == 200:
        return {"status": "created"}
    else:
        return {"status": "exists_or_error", "detail": r.text}

def upsert_vectors(index_name: str, vectors: list):
    """
    Uses the Endee SDK for upsert since it handles the format correctly.
    vectors: list of {id, vector, meta, filter}
    """
    from endee import Endee
    client = Endee()
    index = client.get_index(name=index_name)
    index.upsert(vectors)
    return {"status": "ok"}

def query_vectors(index_name: str, vector: list, top_k: int) -> list:
    from endee import Endee
    client = Endee()
    index = client.get_index(name=index_name)
    results = index.query(vector=vector, top_k=top_k)
    normalized = []
    for r in results:
        normalized.append({
            "id": r.get("id"),
            "score": r.get("similarity", 0.0),
            "metadata": r.get("meta", {}),
        })
    return normalized

def list_indexes():
    r = requests.get(f"{BASE_URL}/index/list", headers=_headers())
    return r.json()