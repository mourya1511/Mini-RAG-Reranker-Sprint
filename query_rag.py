import sqlite3, pickle, numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

DB = Path("data/chunks.db")
EMB_DIR = Path("data")
INDEX_FILE = EMB_DIR / "faiss.index"
META_FILE = EMB_DIR / "chunks_meta.pkl"

def load_index_and_meta():
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    index, meta = load_index_and_meta()
    D, I = index.search(query_emb.astype("float32"), top_k)

    print("\nTop results:")
    for rank, idx in enumerate(I[0]):
        print(f"[{rank+1}] {meta['texts'][idx]}")
    print()

def main():
    if not DB.exists() or not INDEX_FILE.exists():
        print("No index found. Please run embed_index.py first.")
        return

    while True:
        q = input("Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        search(q)

if __name__ == "__main__":
    main()
