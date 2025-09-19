# reranker.py
import re, numpy as np
from baseline_search import MODEL, vector_search, META_FILE
import pickle

ALPHA = 0.7

def tokenize(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    tokens = [t for t in text.split() if t]
    STOP = {"the","and","a","an","of","in","for","to","on","with","by","is","are","that"}
    return [t for t in tokens if t not in STOP]

def keyword_score(query_tokens, chunk_tokens):
    if not query_tokens:
        return 0.0
    matches = sum(1 for t in query_tokens if t in chunk_tokens)
    return matches / len(query_tokens)

def rerank(query, baseline_results, alpha=ALPHA):
    q_tokens = tokenize(query)
    reranked = []
    k_scores, v_scores = [], []
    for r in baseline_results:
        chunk_tokens = tokenize(r["chunk"])
        k = keyword_score(q_tokens, chunk_tokens)
        v = r["score"]
        k_scores.append(k); v_scores.append(v)
        reranked.append({**r, "keyword": k, "vector": v})
    def norm(arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    vn = norm(v_scores)
    kn = norm(k_scores)
    for i, item in enumerate(reranked):
        item["vector_norm"] = float(vn[i])
        item["keyword_norm"] = float(kn[i])
        item["combined_score"] = float(alpha * vn[i] + (1-alpha) * kn[i])
    reranked_sorted = sorted(reranked, key=lambda x: x["combined_score"], reverse=True)
    return reranked_sorted
