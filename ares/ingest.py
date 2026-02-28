"""
Ingest sample documents into Endee with metadata.
Run this once to populate the index.
"""
import os
from dotenv import load_dotenv
from ares.embedder import embed_batch
from ares.endee_client import create_index, upsert_vectors

load_dotenv()

INDEX = os.getenv("DEFAULT_INDEX", "ares_index")
DIM   = int(os.getenv("VECTOR_DIM", 384))

DOCUMENTS = [
    {"id": "doc_001", "text": "The Federal Reserve raised interest rates to combat inflation in 2024.", "meta": {"topic": "finance", "year": 2024, "source": "internal_docs"}, "filter": {"topic": "finance", "year": 2024}},
    {"id": "doc_002", "text": "Inflation reached a 40-year high driven by supply chain disruptions and energy costs.", "meta": {"topic": "finance", "year": 2023, "source": "internal_docs"}, "filter": {"topic": "finance", "year": 2023}},
    {"id": "doc_003", "text": "Machine learning models require large labeled datasets for supervised training.", "meta": {"topic": "ai", "year": 2024, "source": "research"}, "filter": {"topic": "ai", "year": 2024}},
    {"id": "doc_004", "text": "Transformer architectures revolutionized natural language processing tasks.", "meta": {"topic": "ai", "year": 2023, "source": "research"}, "filter": {"topic": "ai", "year": 2023}},
    {"id": "doc_005", "text": "HNSW indexing enables approximate nearest-neighbor search at scale.", "meta": {"topic": "infrastructure", "year": 2024, "source": "internal_docs"}, "filter": {"topic": "infrastructure", "year": 2024}},
    {"id": "doc_006", "text": "Vector databases store high-dimensional embeddings for semantic retrieval.", "meta": {"topic": "infrastructure", "year": 2024, "source": "internal_docs"}, "filter": {"topic": "infrastructure", "year": 2024}},
    {"id": "doc_007", "text": "Stock market volatility increased following geopolitical tensions in 2024.", "meta": {"topic": "finance", "year": 2024, "source": "news"}, "filter": {"topic": "finance", "year": 2024}},
    {"id": "doc_008", "text": "Retrieval Augmented Generation (RAG) combines LLMs with external knowledge bases.", "meta": {"topic": "ai", "year": 2024, "source": "research"}, "filter": {"topic": "ai", "year": 2024}},
    {"id": "doc_009", "text": "Docker containerization simplifies deployment of distributed systems.", "meta": {"topic": "infrastructure", "year": 2023, "source": "internal_docs"}, "filter": {"topic": "infrastructure", "year": 2023}},
    {"id": "doc_010", "text": "Central banks worldwide coordinated policy responses to manage global inflation.", "meta": {"topic": "finance", "year": 2023, "source": "news"}, "filter": {"topic": "finance", "year": 2023}},
    {"id": "doc_011", "text": "Fine-tuning large language models on domain-specific data improves task performance.", "meta": {"topic": "ai", "year": 2024, "source": "research"}, "filter": {"topic": "ai", "year": 2024}},
    {"id": "doc_012", "text": "Kubernetes orchestrates containerized workloads across cloud infrastructure.", "meta": {"topic": "infrastructure", "year": 2024, "source": "internal_docs"}, "filter": {"topic": "infrastructure", "year": 2024}},
]

def main():
    print(f"Creating index '{INDEX}' (dim={DIM})...")
    result = create_index(INDEX, DIM)
    print(f"  → {result}")

    texts = [d["text"] for d in DOCUMENTS]
    print("Embedding documents...")
    vectors = embed_batch(texts)

    payload = [
        {
            "id": d["id"],
            "vector": v,
            "meta": d["meta"],
            "filter": d["filter"]
        }
        for d, v in zip(DOCUMENTS, vectors)
    ]

    print(f"Upserting {len(payload)} documents into Endee...")
    result = upsert_vectors(INDEX, payload)
    print(f"  → {result}")
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()