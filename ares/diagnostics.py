"""
Confidence modeling and similarity analysis for ARES.
"""
import numpy as np

def compute_confidence_metrics(scores: list[float]) -> dict:
    """
    Given a list of cosine similarity scores (higher = better),
    compute retrieval confidence metrics.
    """
    if not scores:
        return {}
    
    arr = np.array(scores)
    mean_sim   = float(np.mean(arr))
    std_sim    = float(np.std(arr))
    top1       = float(arr[0]) if len(arr) > 0 else 0.0
    topk_mean  = float(np.mean(arr[1:])) if len(arr) > 1 else top1
    gap        = top1 - topk_mean          # top-1 vs rest gap

    # Confidence heuristic: high gap + high top-1 = confident retrieval
    # Normalize gap to [0,1] and blend with mean similarity
    confidence = min(1.0, (top1 * 0.5) + (gap * 0.5))

    return {
        "mean_similarity": round(mean_sim, 4),
        "std_similarity":  round(std_sim, 4),
        "top1_score":      round(top1, 4),
        "topk_mean":       round(topk_mean, 4),
        "top1_topk_gap":   round(gap, 4),
        "confidence":      round(confidence, 4),
    }

def should_expand_query(metrics: dict, threshold: float = 0.3) -> bool:
    """
    Return True if retrieval confidence is too low → trigger query expansion.
    """
    return metrics.get("confidence", 1.0) < threshold

def adaptive_k(metrics: dict, base_k: int = 5, max_k: int = 20) -> int:
    """
    Increase k when confidence is low, so we cast a wider net.
    """
    conf = metrics.get("confidence", 1.0)
    if conf >= 0.7:
        return base_k
    elif conf >= 0.4:
        return min(base_k * 2, max_k)
    else:
        return max_k