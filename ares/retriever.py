"""
ARES Core Retrieval Engine.
Implements: adaptive k, confidence modeling, query expansion, metadata filtering.
"""
import time
import os
from dotenv import load_dotenv
from ares.embedder import embed
from ares.endee_client import query_vectors
from ares.diagnostics import compute_confidence_metrics, should_expand_query, adaptive_k

load_dotenv()

INDEX = os.getenv("DEFAULT_INDEX", "ares_index")

EXPANSION_TEMPLATES = [
    "{query} detailed explanation",
    "{query} overview and summary",
    "what is {query}",
    "{query} key concepts",
]

def _apply_metadata_filter(results: list[dict], filters: dict) -> list[dict]:
    if not filters:
        return results
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if all(meta.get(k) == v for k, v in filters.items()):
            filtered.append(r)
    return filtered

def retrieve(
    query: str,
    base_k: int = 5,
    metadata_filters: dict = None,
    allow_expansion: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Main ARES retrieval function.
    Returns results + full diagnostics trace.
    """
    start = time.time()
    trace = []
    
    # --- Pass 1: Initial retrieval ---
    vec = embed(query)
    initial_results = query_vectors(INDEX, vec, top_k=base_k)
    scores = [r["score"] for r in initial_results]
    metrics = compute_confidence_metrics(scores)
    trace.append({
        "stage": "initial",
        "query": query,
        "k": base_k,
        "scores": scores,
        "metrics": metrics,
    })

    # --- Adaptive k: re-query with larger k if confidence is low ---
    new_k = adaptive_k(metrics, base_k=base_k)
    if new_k > base_k:
        initial_results = query_vectors(INDEX, vec, top_k=new_k)
        scores = [r["score"] for r in initial_results]
        metrics = compute_confidence_metrics(scores)
        trace.append({
            "stage": "adaptive_k",
            "query": query,
            "k": new_k,
            "scores": scores,
            "metrics": metrics,
        })

    # --- Query Expansion: triggered when confidence < threshold ---
    expanded_results = []
    expansion_used = None
    if allow_expansion and should_expand_query(metrics):
        for template in EXPANSION_TEMPLATES:
            exp_query = template.format(query=query)
            exp_vec = embed(exp_query)
            exp_results = query_vectors(INDEX, exp_vec, top_k=new_k)
            if exp_results:
                exp_scores = [r["score"] for r in exp_results]
                exp_metrics = compute_confidence_metrics(exp_scores)
                trace.append({
                    "stage": "expansion",
                    "query": exp_query,
                    "k": new_k,
                    "scores": exp_scores,
                    "metrics": exp_metrics,
                })
                # Use expansion if it yields better confidence
                if exp_metrics["confidence"] > metrics["confidence"]:
                    expanded_results = exp_results
                    expansion_used = exp_query
                    metrics = exp_metrics
                    break

    # Merge & deduplicate
    combined = initial_results + expanded_results
    seen_ids = set()
    unique_results = []
    for r in combined:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_results.append(r)

    # Sort by score descending
    unique_results.sort(key=lambda x: x["score"], reverse=True)

    # --- Metadata filtering ---
    final_results = _apply_metadata_filter(unique_results, metadata_filters or {})

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "query": query,
        "expansion_used": expansion_used,
        "final_k": len(final_results),
        "results": final_results,
        "confidence_metrics": metrics,
        "trace": trace,
        "latency_ms": latency_ms,
    }