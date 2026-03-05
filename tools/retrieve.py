
import re
import json
import duckdb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DB_PATH = "memory.db"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def init_retrieval_store():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id      VARCHAR PRIMARY KEY,
            session_id    VARCHAR,
            document_name VARCHAR,
            chunk_index   INTEGER,
            text          VARCHAR,
            embedding     VARCHAR
        )
    """)
    con.close()


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text(file_path: str, file_type: str) -> str:
    if file_type == "pdf":
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip()

    if file_type == "txt":
        return Path(file_path).read_text()

    if file_type == "docx":
        import docx
        doc = docx.Document(file_path)
        return " ".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {file_type}")


def ingest_document(
    file_path: str,
    file_type: str,
    session_id: str,
    document_name: str
) -> int:
    text = extract_text(file_path, file_type)
    chunks = chunk_text(text)
    model = get_model()
    embeddings = model.encode(chunks)

    con = duckdb.connect(DB_PATH)
    con.execute(
        "DELETE FROM document_chunks WHERE session_id = ?",
        [session_id]
    )

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{session_id}_{document_name}_{i}"
        con.execute(
            "INSERT OR REPLACE INTO document_chunks VALUES (?, ?, ?, ?, ?, ?)",
            [chunk_id, session_id, document_name, i, chunk,
             json.dumps(embedding.tolist())]
        )

    con.close()
    return len(chunks)


def retrieve(
    query: str,
    session_id: str,
    top_k: int = 5
) -> list[dict]:
    model = get_model()
    query_embedding = model.encode([query])[0]

    con = duckdb.connect(DB_PATH)
    rows = con.execute(
        "SELECT chunk_id, text, embedding FROM document_chunks WHERE session_id = ?",
        [session_id]
    ).fetchall()
    con.close()

    if not rows:
        return []

    scored = []
    for chunk_id, text, emb_json in rows:
        chunk_emb = np.array(json.loads(emb_json))
        score = float(
            np.dot(query_embedding, chunk_emb) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-9)
        )
        scored.append({"chunk_id": chunk_id, "text": text, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import tempfile, os
    init_retrieval_store()

    sample = """
    Medicare Part B covers medically necessary services and preventive services.
    Prior authorization is required for certain covered outpatient services.
    Policy number CMS-2024-PA-001 applies to all participating providers.
    Coverage determination must be obtained before the scheduled date of service.
    Providers must submit Form CMS-1450 with supporting clinical documentation.
    Failure to obtain prior authorization may result in claim denial.
    Appeals must be submitted within 120 days of the initial determination.
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample)
        tmp_path = f.name

    chunks_stored = ingest_document(tmp_path, "txt", "test_session", "cms_policy.txt")
    print(f"Ingested: {chunks_stored} chunks")

    results = retrieve("prior authorization requirements", "test_session", top_k=2)
    print(f"Retrieved: {len(results)} chunks")
    for r in results:
        print(f"  score={r['score']:.3f} | {r['text'][:80]}...")

    os.unlink(tmp_path)
    print("Retrieval test passed")
