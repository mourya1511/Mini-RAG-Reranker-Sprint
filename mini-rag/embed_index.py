# embed_index.py
import sqlite3, os, numpy as np, pickle
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import torch, random

DB = Path("data/chunks.db")
EMB_DIR = Path("data/embeddings")
INDEX_FILE = EMB_DIR / "faiss.index"
META_FILE = EMB_DIR / "meta.pkl"
EMB_ARR_FILE = EMB_DIR / "embeddings.npy"
SEED = int(os.getenv("RAG_SEED","42"))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def main():
    set_seed(SEED)
    os.makedirs(EMB_DIR, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, source_id, title, url, chunk FROM chunks ORDER BY id")
    rows = cur.fetchall()
    ids, metas, texts = [], [], []
    for r in rows:
        id_, source_id, title, url, chunk = r
        ids.append(id_)
        metas.append({"id": id_, "source_id": source_id, "title": title, "url": url})
        texts.append(chunk)

    print("Computing embeddings for", len(texts), "chunks")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    np.save(EMB_ARR_FILE, embeddings)
    with open(META_FILE, "wb") as f:
        pickle.dump({"ids": ids, "metas": metas, "texts": texts}, f)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_FILE))
    print("Saved FAISS index and metadata.")

if __name__ == "__main__":
    main()
