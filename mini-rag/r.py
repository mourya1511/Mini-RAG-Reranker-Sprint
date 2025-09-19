from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAG API running. Use POST /ask to query."}
