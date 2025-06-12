import os
import json
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# ─── Init OpenAI Client ────────────────────────────────────────────────────────

openai = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjEwMDIyODVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.QL8DL1JSynSX8fAicz_79Fy2aUEZnBQvyyk-Hf9jqSM",
    base_url="https://aipipe.org/openai/v1"
)

# ─── Init Pinecone Client ──────────────────────────────────────────────────────

pc = Pinecone(
    api_key="pcsk_7V5KEq_6WMzyiLZLPfNZPDJrbgBveFACV3A7yrjDKb3mG9ugJ8Ak7vmrehFNTK6xeURmkz"
)

INDEX_NAME = "tds-qa-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",  # or "euclidean" or "dotproduct"
        spec=ServerlessSpec(
            cloud="aws",         # must be AWS for free tier
            region="us-east-1"   # only us-east-1 works on free/Starter
        )
    )

index = pc.Index(INDEX_NAME)

# ─── Load your saved chunks ────────────────────────────────────────────────────

chunks = json.loads(Path("all_chunks.json").read_text(encoding="utf-8"))["chunks"]

# ─── Helper function to embed text ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # Update model name format
        input=text
    )
    return response.data[0].embedding  # use `.data[0].embedding` in OpenAI SDK v1

# ─── Batch upload embeddings to Pinecone ───────────────────────────────────────

BATCH_SIZE = 100
vectors = []

if Path("vector_store.json").exists():
    vectors_store = json.loads(Path("vector_store.json").read_text())
    existing_ids = {v[0] for v in vectors_store}
else:
    vectors_store = []
    existing_ids = set()


for i, chunk in enumerate(chunks):
    if chunk["id"] in existing_ids:
        continue  # Skip if already embedded and stored

    vec = embed_text(chunk["text"])
    
    raw_meta = {
        "source": chunk["source"],
        **chunk["metadata"]
    }
    metadata = {k: v for k, v in raw_meta.items() if v is not None}

    vectors.append((chunk["id"], vec, metadata))
    vectors_store.append((chunk["id"], vec, metadata))  # Save to local store too

    if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(chunks):
        index.upsert(vectors=vectors)
        print(f"✅ Upserted {i+1}/{len(chunks)} vectors")
        vectors = []

        # 💾 Save current vector store to disk
        with open("vector_store.json", "w") as f:
            json.dump(vectors_store, f)
