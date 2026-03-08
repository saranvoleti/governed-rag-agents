"""
tools/retrieve.py
Day 8: recursive_chunk_text() - splits by paragraph/sentence/word. No new deps.
Day 9: compute_grounding_score() - float 0.0-1.0 grounding metric.
"""
import os, re, json, time
import duckdb, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_PATH    = os.environ.get("DB_PATH", "memory.db")
MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL_CACHE = None

def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer(MODEL_NAME)
    return _MODEL_CACHE

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

def _split_by_separator(text, separator):
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p + separator for p in parts[:-1]] + [parts[-1]] if len(parts) > 1 else parts

def _build_bm25(texts):
    """Build BM25 index from list of chunk texts."""
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)

def _rrf_fuse(dense_results, bm25_results, k=60):
    """Reciprocal Rank Fusion — combine dense and BM25 rankings."""
    scores = {}
    texts  = {}
    for rank, r in enumerate(dense_results):
        key = r["chunk_index"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        texts[key]  = r
    for rank, r in enumerate(bm25_results):
        key = r["chunk_index"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        texts[key]  = r
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [texts[key] for key, _ in fused]

def recursive_chunk_text(text, chunk_size=512, chunk_overlap=64, separators=None):
    """Recursively split text respecting paragraph/sentence/word boundaries."""
    if separators is None:
        separators = _SEPARATORS
    separator = separators[0]
    remaining = separators[1:]
    splits = _split_by_separator(text, separator)
    chunks = []; current = []; current_len = 0
    for split in splits:
        split_len = len(split)
        if split_len > chunk_size and remaining:
            if current:
                chunks.append("".join(current).strip())
                ov = "".join(current)[-chunk_overlap:]
                current = [ov]; current_len = len(ov)
            chunks.extend(recursive_chunk_text(split, chunk_size, chunk_overlap, remaining))
            current = []; current_len = 0; continue
        if current_len + split_len > chunk_size and current:
            chunks.append("".join(current).strip())
            ov = "".join(current)[-chunk_overlap:]
            current = [ov, split]; current_len = len(ov) + split_len
        else:
            current.append(split); current_len += split_len
    if current:
        last = "".join(current).strip()
        if last: chunks.append(last)
    return [c for c in chunks if c.strip()]

def compute_grounding_score(response, chunks):
    """Float 0.0-1.0: supported_terms / total_meaningful_response_terms."""
    if not chunks or not response:
        return 0.0
    skip = {"the","a","an","is","are","was","were","has","have","this","that",
            "these","those","it","in","on","at","to","for","of","with","by",
            "from","and","or","but","not","as","be","been","will","would",
            "could","should","may","document","states","indicates","notes",
            "according","based","which","also","their","they","them","its",
            "more","than","then","when","what","where","who","how"}
    corpus = set(" ".join(chunks).lower().split())
    terms  = [w.lower().strip(".,;:!?\"'()*#_[]") for w in response.split()
              if len(w) > 4 and w.lower().strip(".,;:!?\"'()*#_[]") not in skip]
    if not terms: return 1.0
    return round(sum(1 for t in terms if t in corpus) / len(terms), 4)

def init_retrieval_store():
    con = duckdb.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS document_chunks_v2 (
        chunk_id TEXT, session_id TEXT, document_name TEXT,
        chunk_index INTEGER, text TEXT, embedding TEXT)""")
    con.close()

def clean_pdf_text(text):
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"[.]\d+ ", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_text(file_path, file_type):
    if file_type == "txt":
        return open(file_path, encoding="utf-8", errors="ignore").read()
    if file_type == "pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(file_path)
        return clean_pdf_text(" ".join(p.extract_text() or "" for p in reader.pages))
    if file_type == "docx":
        import docx
        return " ".join(p.text for p in docx.Document(file_path).paragraphs)
    return ""

def ingest_document(file_path, file_type, session_id, document_name,
                    chunk_size=512, chunk_overlap=64):
    init_retrieval_store()
    text   = extract_text(file_path, file_type)
    chunks = recursive_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    model  = _get_model()
    con    = duckdb.connect(DB_PATH)
    con.execute("DELETE FROM document_chunks_v2 WHERE session_id=? AND document_name=?",
                [session_id, document_name])
    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk)
        con.execute("INSERT INTO document_chunks_v2 VALUES (?,?,?,?,?,?)",
                    [f"{session_id}_{i}", session_id, document_name, i,
                     chunk, json.dumps(emb.tolist())])
    con.close()
    return len(chunks)

def retrieve(query, session_id, top_k=5):
    """Hybrid retrieval: dense (semantic) + BM25 (keyword) fused via RRF."""
    init_retrieval_store()
    t0    = time.perf_counter()
    model = _get_model()
    qemb  = model.encode(query)
    con   = duckdb.connect(DB_PATH)
    rows  = con.execute(
        "SELECT text, embedding, chunk_index FROM document_chunks_v2 "
        "WHERE session_id=? OR session_id='library'", [session_id]).fetchall()
    con.close()
    if not rows: return []

    texts   = [r[0] for r in rows]
    indexes = [r[2] for r in rows]

    # Dense ranking
    dense = []
    for i, (text, emb_json, chunk_index) in enumerate(rows):
        emb   = np.array(json.loads(emb_json))
        score = float(np.dot(qemb, emb) / (np.linalg.norm(qemb) * np.linalg.norm(emb) + 1e-9))
        dense.append({"text": text, "score": score, "chunk_index": chunk_index})
    dense.sort(key=lambda x: x["score"], reverse=True)

    # BM25 ranking
    bm25       = _build_bm25(texts)
    qtokens    = query.lower().split()
    bm25_scores = bm25.get_scores(qtokens)
    bm25_ranked = sorted(
        [{"text": texts[i], "score": float(bm25_scores[i]), "chunk_index": indexes[i]}
         for i in range(len(texts))],
        key=lambda x: x["score"], reverse=True
    )

    # RRF fusion
    fused = _rrf_fuse(dense[:20], bm25_ranked[:20])
    top   = fused[:top_k]
    if top: top[0]["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return top

if __name__ == "__main__":
    sample = """
    The Pay-As-You-Go (PAYGO) rule requires that new legislation increasing
    mandatory spending or reducing revenue must be offset by other provisions
    that decrease spending or increase revenue by equal or greater amounts.

    This fiscal rule was established to enforce budget discipline. It applies
    to both the House and Senate under their respective rules.
    """
    chunks = recursive_chunk_text(sample, chunk_size=200, chunk_overlap=40)
    print(f"Chunks produced: {len(chunks)}")
    score = compute_grounding_score(
        "PAYGO requires new legislation increasing mandatory spending to be offset.", chunks)
    print(f"Grounding score: {score}")
    assert score > 0.5
    print("Self-test passed")
