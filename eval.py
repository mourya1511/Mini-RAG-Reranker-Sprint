# eval.py
import requests, json, time, csv
API = "http://localhost:8000/ask"
QUESTIONS_FILE = "8_questions.json"
OUT_CSV = "results.csv"

def call(q, mode):
    resp = requests.post(API, json={"q": q, "k": 5, "mode": mode})
    return resp.json()

def run_all():
    data = json.load(open(QUESTIONS_FILE))
    rows = []
    for item in data:
        q = item["q"]
        print("Query:", q)
        baseline = call(q, "baseline")
        hybrid = call(q, "hybrid")
        rows.append({
            "q": q,
            "baseline_answer": baseline.get("answer"),
            "hybrid_answer": hybrid.get("answer"),
            "hybrid_top_combined": hybrid["contexts"][0]["combined_score"] if hybrid.get("contexts") else None
        })
        time.sleep(0.1)
    keys = rows[0].keys() if rows else []
    with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", OUT_CSV)

if __name__ == "__main__":
    run_all()
