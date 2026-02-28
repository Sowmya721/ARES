"""Basic smoke tests for ARES."""
import pytest
from ares.diagnostics import compute_confidence_metrics, adaptive_k, should_expand_query

def test_confidence_metrics_high():
    scores = [0.92, 0.45, 0.41, 0.38, 0.35]
    m = compute_confidence_metrics(scores)
    assert m["top1_score"] == 0.92
    assert m["confidence"] > 0.5
    assert m["top1_topk_gap"] > 0

def test_confidence_metrics_low():
    scores = [0.41, 0.40, 0.39, 0.38]
    m = compute_confidence_metrics(scores)
    assert m["confidence"] < 0.5
    assert should_expand_query(m) == True

def test_adaptive_k_high_confidence():
    m = {"confidence": 0.85}
    assert adaptive_k(m, base_k=5) == 5

def test_adaptive_k_low_confidence():
    m = {"confidence": 0.25}
    assert adaptive_k(m, base_k=5, max_k=20) == 20