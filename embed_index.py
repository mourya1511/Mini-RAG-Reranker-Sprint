import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/chunks_meta.pkl"
BATCH_SIZE = 64

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, url, chunk FROM chunks")
    rows = cur.fetchall()

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = []
    embeddings = []

    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[i:i+BATCH_SIZE]
        texts = [r[3] for r in batch]
        # explicit conversion to float32 and normalization
        emb_batch = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb_batch)
        for r in batch:
            chunk_id, title, url, text = r
            chunks.append({"id": chunk_id, "title": title, "url": url, "chunk": text})

    embeddings = np.vstack(embeddings).astype("float32")

    # Build FAISS index using inner product (cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save metadata as a dict (for baseline_search)
    meta = {
        "ids": [c["id"] for c in chunks],
        "texts": [c["chunk"] for c in chunks],
        "metas": chunks
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"FAISS index and metadata saved. Total chunks: {len(chunks)}")

if __name__ == "__main__":
    main()
