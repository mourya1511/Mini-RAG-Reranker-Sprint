import sqlite3
conn = sqlite3.connect("data/chunks.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chunks")
print(cur.fetchone())
