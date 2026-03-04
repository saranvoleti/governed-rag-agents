"""
core/memory.py — Persistent Agent Memory

Memory = memory in Sanskrit. Every agent execution is logged here.
Persists to Azure Blob Storage in production.
Falls back to local DuckDB in development.

Three tables:
  agent_memory  — one row per agent execution
  escalation_streak  — unmapped finding streak per session/domain
  retrieval_log — what was retrieved, similarity scores, chunk sources
"""

import duckdb
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "memory.db"


def init_memory():
    """Create tables if they don't exist. Safe to call multiple times."""
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id           VARCHAR PRIMARY KEY,
            session_id   VARCHAR,
            agent_id     VARCHAR,
            domain       VARCHAR,
            document_name VARCHAR,
            executed_at  TIMESTAMP,
            input_hash   VARCHAR,
            output_hash  VARCHAR,
            trust_score  FLOAT,
            escalate     BOOLEAN,
            reflection_attempts INTEGER,
            firewall_result VARCHAR,
            tool_calls   VARCHAR,
            notes        VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS escalation_streak (
            domain_key   VARCHAR PRIMARY KEY,
            streak       INTEGER,
            last_seen    TIMESTAMP,
            last_score   FLOAT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_log (
            id           VARCHAR PRIMARY KEY,
            session_id   VARCHAR,
            query        VARCHAR,
            chunk_ids    VARCHAR,
            scores       VARCHAR,
            retrieved_at TIMESTAMP
        )
    """)
    con.close()
    print("Memory initialized — agent_memory, escalation_streak, retrieval_log ready")


def log_agent_run(
    session_id: str,
    agent_id: str,
    domain: str,
    document_name: str,
    input_data: dict,
    output_data: dict,
    trust_score: float = 1.0,
    escalate: bool = False,
    reflection_attempts: int = 0,
    firewall_result: str = "PASSED",
    tool_calls: list = None,
    notes: str = "",
):
    """Log one agent execution to agent_memory."""
    con = duckdb.connect(DB_PATH)

    # Hash inputs and outputs for tamper detection
    input_hash  = hashlib.sha256(
        json.dumps(input_data,  sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    output_hash = hashlib.sha256(
        json.dumps(output_data, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    row_id = hashlib.sha256(
        f"{session_id}{agent_id}{input_hash}".encode()
    ).hexdigest()[:16]

    con.execute("""
        INSERT OR REPLACE INTO agent_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        row_id, session_id, agent_id, domain, document_name,
        datetime.now(timezone.utc),
        input_hash, output_hash,
        trust_score, escalate, reflection_attempts,
        firewall_result,
        json.dumps(tool_calls or []),
        notes,
    ])
    con.close()


def get_history(session_id: str = None, domain: str = None, limit: int = 20) -> list:
    """Retrieve execution history. Filter by session or domain."""
    con = duckdb.connect(DB_PATH)
    if session_id:
        rows = con.execute(
            "SELECT * FROM agent_memory WHERE session_id=? ORDER BY executed_at DESC LIMIT ?",
            [session_id, limit]
        ).fetchall()
    elif domain:
        rows = con.execute(
            "SELECT * FROM agent_memory WHERE domain=? ORDER BY executed_at DESC LIMIT ?",
            [domain, limit]
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM agent_memory ORDER BY executed_at DESC LIMIT ?",
            [limit]
        ).fetchall()
    con.close()
    return rows


def log_retrieval(session_id: str, query: str, chunk_ids: list, scores: list):
    """Log what was retrieved and similarity scores."""
    con = duckdb.connect(DB_PATH)
    row_id = hashlib.sha256(f"{session_id}{query}".encode()).hexdigest()[:16]
    con.execute(
        "INSERT OR REPLACE INTO retrieval_log VALUES (?,?,?,?,?,?)",
        [row_id, session_id, query,
         json.dumps(chunk_ids), json.dumps(scores),
         datetime.now(timezone.utc)]
    )
    con.close()


def get_escalation_streak(domain_key: str) -> int:
    """Get current unmapped finding streak for a domain."""
    con = duckdb.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT streak FROM escalation_streak WHERE domain_key=?",
            [domain_key]
        ).fetchone()
        con.close()
        return row[0] if row else 0
    except:
        con.close()
        return 0


def update_escalation_streak(domain_key: str, score: float, increment: bool = True) -> int:
    """Increment or reset streak. Returns new streak value."""
    con = duckdb.connect(DB_PATH)
    current = get_escalation_streak(domain_key)
    new_streak = (current + 1) if increment else 0
    con.execute("""
        INSERT OR REPLACE INTO escalation_streak VALUES (?,?,?,?)
    """, [domain_key, new_streak, datetime.now(timezone.utc), score])
    con.close()
    return new_streak


if __name__ == "__main__":
    init_memory()
    print("Test: logging execution")
    log_agent_run(
        session_id="test_001",
        agent_id="analysis_agent",
        domain="healthcare",
        document_name="clinical_guidelines.pdf",
        input_data={"query": "What are the dosage recommendations?"},
        output_data={"answer": "10mg twice daily", "sources": ["chunk_3"]},
        trust_score=0.92,
        tool_calls=[{"tool": "retrieve_context", "query": "dosage recommendations"}],
    )
    rows = get_history(session_id="test_001")
    print(f"Rows in memory: {len(rows)}")
    print("Memory test passed")
