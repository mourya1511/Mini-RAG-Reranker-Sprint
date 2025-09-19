# ingest.py
import os, json, sqlite3, re, datetime
from pdfminer.high_level import extract_text
from pathlib import Path

DATA_DIR = Path("pdfs")
DB = Path("data/chunks.db")
SOURCES_JSON = Path("sources.json")

def normalize_whitespace(s):
    return re.sub(r'\s+', ' ', s).strip()

def split_into_paragraph_chunks(text, max_chars=1200):
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    final = []
    for c in chunks:
        if len(c) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', c)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = (cur + " " + s).strip()
                else:
                    final.append(cur)
                    cur = s
            if cur:
                final.append(cur)
        else:
            final.append(c)
    return final

def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS sources (
      source_id TEXT PRIMARY KEY,
      title TEXT,
      url TEXT
    );
    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY,
      source_id TEXT,
      title TEXT,
      url TEXT,
      chunk TEXT,
      created_at TEXT
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk, content='chunks', content_rowid='id');
    """)
    conn.commit()

    sources = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        print("Processing", pdf_path.name)
        text = extract_text(str(pdf_path))
        text = normalize_whitespace(text)
        title = pdf_path.stem
        url = f"file://{pdf_path.resolve()}"
        source_id = pdf_path.stem
        sources.append({"title": title, "url": url})
        chunks = split_into_paragraph_chunks(text, max_chars=1200)
        for chunk in chunks:
            now = datetime.datetime.utcnow().isoformat()
            cur.execute("INSERT INTO chunks (source_id,title,url,chunk,created_at) VALUES (?,?,?,?,?)",
                        (source_id, title, url, chunk, now))
            rowid = cur.lastrowid
            cur.execute("INSERT INTO chunks_fts(rowid, chunk) VALUES (?,?)", (rowid, chunk))
        cur.execute("INSERT OR IGNORE INTO sources (source_id,title,url) VALUES (?,?,?)",
                    (source_id, title, url))
        conn.commit()
    with open(SOURCES_JSON, "w") as f:
        json.dump(sources, f, indent=2)
    conn.close()
    print("Done ingesting.")

if __name__ == "__main__":
    main()
