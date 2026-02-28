"""
Benchmarks static k=5 vs ARES Adaptive retrieval.
Prints a comparison table.
"""
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ares.embedder import embed
from ares.endee_client import query_vectors
from ares.retriever import retrieve
from ares.diagnostics import compute_confidence_metrics

INDEX = os.getenv("DEFAULT_INDEX", "ares_index")

QUERIES = [
    "interest rates monetary policy",
    "machine learning transformers",
    "container orchestration kubernetes",
    "inflation supply chain",
    "vector database semantic search",
]

def static_retrieve(query: str, k: int = 5) -> dict:
    start = time.time()
    vec = embed(query)
    results = query_vectors(INDEX, vec, top_k=k)
    scores = [r["score"] for r in results]
    metrics = compute_confidence_metrics(scores)
    latency = round((time.time() - start) * 1000, 2)
    return {"results": results, "metrics": metrics, "latency_ms": latency}

def run_benchmark():
    print(f"\n{'='*70}")
    print(f"{'Query':<35} {'Mode':<12} {'Confidence':<12} {'Latency(ms)':<12} {'k'}")
    print(f"{'='*70}")

    for query in QUERIES:
        # Static k=5
        s = static_retrieve(query, k=5)
        print(f"{query[:34]:<35} {'Static k=5':<12} "
              f"{s['metrics'].get('confidence','—'):<12} "
              f"{s['latency_ms']:<12} 5")

        # ARES Adaptive
        a = retrieve(query, base_k=5, allow_expansion=True)
        print(f"{'':<35} {'ARES Adapt':<12} "
              f"{a['confidence_metrics'].get('confidence','—'):<12} "
              f"{a['latency_ms']:<12} {a['final_k']}")
        print(f"{'-'*70}")

if __name__ == "__main__":
    run_benchmark()