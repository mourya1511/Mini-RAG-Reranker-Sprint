# baseline_search.py
import numpy as np, pickle, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB_DIR = Path("data/embeddings")
INDEX_FILE = Path("data/faiss.index")       # was data/embeddings/faiss.index
META_FILE = Path("data/chunks_meta.pkl")   
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query, k=5):
    q_emb = MODEL.encode([query], normalize_embeddings=True)
    index = faiss.read_index(str(INDEX_FILE))
    D, I = index.search(q_emb.astype("float32"), k)
    with open(META_FILE,"rb") as f:
        meta = pickle.load(f)
    ids = meta["ids"]
    texts = meta["texts"]
    metas = meta["metas"]
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        results.append({
            "score": float(score),
            "idx": int(idx),
            "chunk": texts[idx],
            "meta": metas[idx]
        })
    return results

if __name__ == "__main__":
    q = "machine guarding standards for conveyors"
    res = vector_search(q, k=5)
    for r in res:
        print(r["score"], r["meta"]["title"], r["chunk"][:200].replace("\n"," "), "\n---")
