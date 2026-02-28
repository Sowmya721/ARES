# ARES — Adaptive Retrieval Engine for Semantic Systems

> A self-aware vector retrieval quality layer built on top of Endee, a high-performance local vector database.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Vector DB](https://img.shields.io/badge/VectorDB-Endee-orange)
![Embeddings](https://img.shields.io/badge/Embeddings-sentence--transformers-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚨 Problem

Static vector retrieval is **blind by default**.

When you run similarity search with a fixed `k=5`, you always get 5 results — even if none are relevant.  
There is **no signal** telling you whether retrieval actually succeeded.

This leads to:

- ❌ Irrelevant context in RAG pipelines  
- ❌ Poor recommendations  
- ❌ No observability into retrieval quality  

---

## ✅ Solution — ARES

ARES adds an **adaptive quality layer** on top of Endee that makes retrieval:

- 📊 **Confidence-aware** (mean, std, top-1 gap → score ∈ [0,1])  
- 🔁 **Adaptive-k** (widens search when confidence is low)  
- 🧠 **Query-expanding** (rephrases weak queries automatically)  
- 🏷️ **Metadata-filtered** (topic, year, source)  
- 📈 **Observable** via a live diagnostics dashboard  

Result → **self-aware retrieval** that is more accurate than static search.

---

# 🏗️ System Architecture
User Query
│
▼
Embedder (all-MiniLM-L6-v2 → 384d vector)
│
▼
Endee Vector DB (top-k + similarity)
│
▼
Confidence Model
    • mean similarity
    • std deviation
    • top-1 vs rest gap
    │
    ├─ confidence ≥ 0.7 → return as-is
    ├─ 0.4–0.7 → double k
    └─ < 0.4 → max k + query expansion
                        │
                        ▼
                Metadata Filter
                        │
                        ▼
            Final Results + Diagnostics

---

## 📐 Confidence Modeling

| Metric           | Formula                         | Purpose                     |
|------------------|---------------------------------|-----------------------------|
| Mean similarity  | `mean(scores)`                  | Overall relevance signal    |
| Std. deviation   | `std(scores)`                   | Uncertainty / spread        |
| Top-1 gap        | `scores[0] - mean(scores[1:])`  | Strength of best match      |
| Confidence       | `0.5 * top1 + 0.5 * gap`        | Final decision score        |

---

## 🔍 Adaptive Behaviour

| Confidence Range | Action                          |
|------------------|---------------------------------|
| `>= 0.7`         | Keep base `k`                   |
| `0.4 - 0.7`      | Double `k`                      |
| `< 0.4`          | Max `k` + query expansion       |

---

# 🔁 Query Expansion

Triggered when confidence `< 0.3`.

ARES tries:

- `"{query} detailed explanation"`  
- `"{query} overview and summary"`  
- `"what is {query}"`  
- `"{query} key concepts"`  

The expansion with **highest confidence** is selected.

---

# 🗄️ How Endee Is Used

⭐ Forked from: https://github.com/EndeeLabs/endee

| Operation | Usage |
|---|---|
`index/create` | Create `ares_index` (dim=384, cosine) |
`index.upsert` | Store vectors + `meta` + `filter` |
`index.query` | Retrieve top-k candidates |
`index/list` | Startup index check |

### Stored Document Format

```json
{
  "id": "doc_001",
  "vector": [0.032, -0.018, ...],
  "meta": {
    "topic": "finance",
    "year": 2024,
    "source": "internal_docs"
  },
  "filter": {
    "topic": "finance",
    "year": 2024
  }
}

---
📊 Benchmarks
Mode	        Avg Confidence	    Avg Latency	        Behavior
Static k=5	        0.52	            ~38 ms	        Fixed retrieval
ARES Adaptive	    0.58	            ~59 ms	        Self-adjusting + expansion

Key insight: High-confidence queries skip expansion → no wasted compute.

---
📂 Project Structure
ARES/
├── docker-compose.yml
├── .env
├── requirements.txt
├── README.md
├── ares/
│   ├── endee_client.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── ingest.py
│   └── diagnostics.py
├── dashboard/
│   └── app.py
├── scripts/
│   └── benchmark.py
└── tests/
    └── test_retriever.py


---
⚙️ Setup
1️⃣ Clone repo
git clone https://github.com/Sowmya721/ARES.git
cd ARES
2️⃣ Virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
🐳 Start Endee
docker compose up -d

Verify:

python -c "import requests; print(requests.get('http://127.0.0.1:8080/api/v1/index/list').text)"

Expected:

{"indexes":[...]}
📥 Ingest Documents
python -m ares.ingest

Expected output:

Creating index 'ares_index' (dim=384)...
Index status: created
Embedding documents...
Upserting 12 documents into Endee...
→ {'status': 'ok'}
✅ Ingestion complete.
📊 Dashboard
streamlit run dashboard/app.py

Open → http://localhost:8501

Features:

Confidence score

Similarity histogram

Retrieval trace

Metadata filtering


---
🧪 Benchmark
python scripts/benchmark.py

Compares static vs adaptive retrieval.

✅ Tests
python -m pytest tests/

All tests should pass.

---

🧰 Tech Stack
Component	    Technology
Vector DB	    Endee
Embeddings	    sentence-transformers
Dashboard	    Streamlit
Math	        NumPy
HTTP Client	    requests
Testing	        pytest

---
📜 License
MIT

---
🌟 Key Takeaway
ARES turns vector search from a black box into a measurable, adaptive system.

It doesn’t just retrieve results —
it knows whether retrieval worked.
