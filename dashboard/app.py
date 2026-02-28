"""
ARES Diagnostics Dashboard — Streamlit
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ares.retriever import retrieve

st.set_page_config(page_title="ARES Dashboard", layout="wide")
st.title("🔍 ARES — Adaptive Retrieval Engine for Semantic Systems")
st.caption("Vector retrieval quality layer built on Endee")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Query Settings")
query = st.sidebar.text_input("Query", value="interest rates and inflation")
base_k = st.sidebar.slider("Base k", 3, 20, 5)
allow_expansion = st.sidebar.checkbox("Allow Query Expansion", value=True)

st.sidebar.subheader("Metadata Filters (optional)")
topic_filter = st.sidebar.selectbox("Topic", ["(none)", "finance", "ai", "infrastructure"])
year_filter  = st.sidebar.selectbox("Year",  ["(none)", "2023", "2024"])

filters = {}
if topic_filter != "(none)":
    filters["topic"] = topic_filter
if year_filter != "(none)":
    filters["year"] = int(year_filter)

run = st.sidebar.button("🚀 Run Retrieval")

# ── Main area ─────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Retrieving..."):
        output = retrieve(
            query=query,
            base_k=base_k,
            metadata_filters=filters,
            allow_expansion=allow_expansion,
        )

    # ── Metrics row ───────────────────────────────────────────────────────────
    m = output["confidence_metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Confidence",    m.get("confidence", "—"))
    col2.metric("Mean Similarity", m.get("mean_similarity", "—"))
    col3.metric("Std Dev",       m.get("std_similarity", "—"))
    col4.metric("Top-1 Score",   m.get("top1_score", "—"))
    col5.metric("Top-1 vs Rest Gap", m.get("top1_topk_gap", "—"))

    st.markdown(f"**Latency:** `{output['latency_ms']} ms` | "
                f"**Final results returned:** `{output['final_k']}` | "
                f"**Expansion used:** `{output['expansion_used'] or 'None'}`")

    # ── Similarity histogram ──────────────────────────────────────────────────
    all_scores = []
    stage_labels = []
    for t in output["trace"]:
        all_scores.append(t["scores"])
        stage_labels.append(t["stage"])

    st.subheader("📊 Similarity Score Distribution")
    fig, ax = plt.subplots(figsize=(9, 3))
    colors = ["#4C9BE8", "#F4845F", "#56C596", "#A78BFA"]
    for i, (scores, label) in enumerate(zip(all_scores, stage_labels)):
        if scores:
            ax.hist(scores, bins=10, alpha=0.6, label=label, color=colors[i % len(colors)])
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # ── Retrieval trace ───────────────────────────────────────────────────────
    st.subheader("🔎 Retrieval Trace")
    for step in output["trace"]:
        with st.expander(f"Stage: `{step['stage']}` — query: _{step['query']}_"):
            st.json({
                "k": step["k"],
                "scores": step["scores"],
                "metrics": step["metrics"],
            })

    # ── Final results ──────────────────────────────────────────────────────────
    st.subheader("📄 Retrieved Documents")
    for i, r in enumerate(output["results"], 1):
        score_bar = "█" * int(r["score"] * 20)
        st.markdown(f"**{i}. [{r['id']}]** — Score: `{r['score']:.4f}` `{score_bar}`")
        meta = r.get("metadata", {})
        st.caption(f"Topic: {meta.get('topic','—')} | Year: {meta.get('year','—')} | Source: {meta.get('source','—')}")
        if "text" in r:
            st.write(r["text"])
else:
    st.info("Set your query in the sidebar and click **Run Retrieval**.")