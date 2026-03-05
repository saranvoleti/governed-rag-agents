import os
import json
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = os.environ.get("DB_PATH", "memory.db")
MODEL_NAME = "all-MiniLM-L6-v2"

def init_retrieval_store():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id TEXT,
            session_id TEXT,
            document_name TEXT,
            chunk_index INTEGER,
            text TEXT,
            embedding TEXT
        )
    """)
    con.close()

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def extract_text(file_path, file_type):
    if file_type == "txt":
        return open(file_path, encoding="utf-8", errors="ignore").read()
    if file_type == "pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(file_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    if file_type == "docx":
        import docx
        doc = docx.Document(file_path)
        return " ".join(p.text for p in doc.paragraphs)
    return ""

def ingest_document(file_path, file_type, session_id, document_name):
    init_retrieval_store()
    text = extract_text(file_path, file_type)
    chunks = chunk_text(text)
    model = SentenceTransformer(MODEL_NAME)
    con = duckdb.connect(DB_PATH)
    con.execute(
        "DELETE FROM document_chunks WHERE session_id = ? AND document_name = ?",
        [session_id, document_name]
    )
    for i, chunk in enumerate(chunks):
        chunk_id = f"{session_id}_{document_name}_{i}"
        embedding = model.encode(chunk)
        con.execute(
            "INSERT INTO document_chunks VALUES (?, ?, ?, ?, ?, ?)",
            [chunk_id, session_id, document_name, i, chunk, json.dumps(embedding.tolist())]
        )
    con.close()
    return len(chunks)

def retrieve(query, session_id, top_k=5):
    init_retrieval_store()
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query)
    con = duckdb.connect(DB_PATH)
    rows = con.execute(
        "SELECT text, embedding FROM document_chunks WHERE session_id = ? OR session_id = 'library'",
        [session_id]
    ).fetchall()
    con.close()
    if not rows:
        return []
    results = []
    for text, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-9))
        results.append({"text": text, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
