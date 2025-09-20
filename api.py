# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from baseline_search import vector_search
from reranker import rerank
import uvicorn, re

app = FastAPI()

class AskReq(BaseModel):
    q: str
    k: Optional[int] = 5
    mode: Optional[str] = "hybrid"

THRESHOLD = 0.25

def pick_answer(query, top_chunk_text, max_len=300):
    sentences = re.split(r'(?<=[.!?])\s+', top_chunk_text.strip())
    q_tokens = set(re.sub(r'[^a-z0-9\s]',' ', query.lower()).split())
    for s in sentences:
        toks = set(re.sub(r'[^a-z0-9\s]',' ', s.lower()).split())
        if q_tokens & toks:
            return s.strip()[:max_len]
    return sentences[0].strip()[:max_len]

@app.post("/ask")
def ask(req: AskReq):
    q = req.q
    k = req.k or 5
    mode = req.mode or "hybrid"
    baseline = vector_search(q, k=k*3)
    if mode == "baseline":
        used = "baseline"
        contexts = baseline[:k]
        contexts_resp = [{"chunk": c["chunk"], "meta": c["meta"], "vector_score": c["score"]} for c in contexts]
        answer = None
        if contexts and contexts[0]["score"] > THRESHOLD:
            answer = pick_answer(q, contexts[0]["chunk"])
        return {"answer": answer, "contexts": contexts_resp, "reranker_used": used}
    else:
        used = "hybrid"
        reranked = rerank(q, baseline)
        contexts = reranked[:k]
        contexts_resp = []
        for c in contexts:
            contexts_resp.append({
                "chunk": c["chunk"],
                "meta": c["meta"],
                "vector_score": c["vector"],
                "keyword_score": c["keyword"],
                "combined_score": c["combined_score"]
            })
        top_score = contexts[0]["combined_score"] if contexts else 0.0
        answer = pick_answer(q, contexts[0]["chunk"]) if contexts and top_score >= THRESHOLD else None
        return {"answer": answer, "contexts": contexts_resp, "reranker_used": used}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
