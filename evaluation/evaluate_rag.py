# BM25 run 2026-03-07: Recall@5=1.00  Grounding=1.00  Hallucination=0.0%  Latency=737ms
# Baseline 2026-03-07: Recall@5=0.95  Grounding=1.00  Hallucination=0.0%  Latency=326ms  Coverage=100%
import os, sys, json, time, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.retrieve import retrieve, init_retrieval_store
from core.firewall import run_firewall
from core.memory import init_memory

def compute_recall_at_k(query, expected_keywords, session_id, k=5):
    results = retrieve(query, session_id, top_k=k)
    if not results:
        return {"recall":0.0,"hit":False,"chunks_retrieved":0,"matched_keyword":"","top_score":0.0}
    combined = " ".join(r["text"].lower() for r in results)
    matched  = next((kw for kw in expected_keywords if kw.lower() in combined), None)
    return {"recall":1.0 if matched else 0.0,"hit":bool(matched),
            "chunks_retrieved":len(results),"matched_keyword":matched or "",
            "top_score":round(results[0]["score"],4)}

def evaluate_query(query_obj, session_id, k=5):
    query    = query_obj["query"]
    keywords = query_obj["expected_keywords"]
    t_start  = time.perf_counter()
    recall_r = compute_recall_at_k(query, keywords, session_id, k=k)
    results  = retrieve(query, session_id, top_k=k)
    chunks   = [r["text"] for r in results]
    proxy    = chunks[0][:300] if chunks else ""
    fw       = run_firewall(proxy, chunks, is_fallback=True) if proxy else run_firewall("", [])
    return {"id":query_obj["id"],"query":query,"domain":query_obj.get("domain","general"),
            "recall":recall_r["recall"],"hit":recall_r["hit"],
            "matched_keyword":recall_r["matched_keyword"],"top_score":recall_r.get("top_score",0.0),
            "grounding_score":fw.grounding_score,"firewall_passed":fw.passed,
            "fallback_used":fw.fallback_used,"chunks_retrieved":recall_r["chunks_retrieved"],
            "latency_ms":round((time.perf_counter()-t_start)*1000,1),
            "notes":query_obj.get("notes","")}

def run_evaluation(queries_path, session_id, k=5, save_results=True):
    with open(queries_path) as f:
        queries = json.load(f)
    print(f"\nRunning evaluation on {len(queries)} queries (Recall@{k})...")
    print(f"Session: {session_id}\n")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i:02d}/{len(queries)}] {q['id']}: {q['query'][:55]:<55}", end=" ", flush=True)
        r = evaluate_query(q, session_id, k=k)
        results.append(r)
        print(f"{'✅ HIT' if r['hit'] else '❌ MISS'}  grounding={r['grounding_score']:.2f}  {r['latency_ms']}ms")
    total         = len(results)
    hits          = sum(1 for r in results if r["hit"])
    recall_at_k   = round(hits/total, 4)
    avg_grounding = round(statistics.mean(r["grounding_score"] for r in results), 4)
    avg_latency   = round(statistics.mean(r["latency_ms"] for r in results), 1)
    fw_failed     = sum(1 for r in results if not r["firewall_passed"])
    halluc_rate   = round(fw_failed/total, 4)
    fallback_rate = round(sum(1 for r in results if r["fallback_used"])/total, 4)
    coverage      = round(sum(1 for r in results if r["chunks_retrieved"]>0)/total, 4)
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains: domains[d]={"total":0,"hits":0}
        domains[d]["total"]+=1; domains[d]["hits"]+=int(r["hit"])
    domain_recall = {d:round(v["hits"]/v["total"],2) for d,v in domains.items()}
    summary = {"total_queries":total,"recall_at_k":recall_at_k,"hits":hits,
               "avg_grounding_score":avg_grounding,"hallucination_rate":halluc_rate,
               "avg_latency_ms":avg_latency,"fallback_rate":fallback_rate,
               "coverage":coverage,"domain_recall":domain_recall,"k":k}
    sep = "="*54
    print(f"\n{sep}")
    print(f"  RAG Evaluation Report")
    print(f"{sep}")
    print(f"  Queries evaluated  : {total}")
    print(f"  Recall@{k}           : {recall_at_k:.2f}  ({hits}/{total} hits)")
    print(f"  Avg Grounding Score: {avg_grounding:.2f}")
    print(f"  Hallucination Rate : {halluc_rate:.1%}")
    print(f"  Avg Latency        : {avg_latency}ms")
    print(f"  Fallback Rate      : {fallback_rate:.1%}")
    print(f"  Coverage           : {coverage:.1%}")
    print(f"{sep}")
    print(f"  Domain Recall:")
    for d, rec in sorted(domain_recall.items()):
        print(f"    {d:<12} {rec:.2f}  [{'#'*int(rec*20):<20}]")
    print(f"{sep}")
    issues = []
    if recall_at_k  < 0.70: issues.append(f"Recall@{k} {recall_at_k:.2f} below 0.70 target")
    if avg_grounding< 0.70: issues.append(f"Grounding {avg_grounding:.2f} below 0.70 target")
    if halluc_rate  > 0.10: issues.append(f"Hallucination {halluc_rate:.1%} above 10% target")
    if avg_latency  > 3000: issues.append(f"Latency {avg_latency}ms above 3000ms target")
    print(f"  Status: {'PRODUCTION READY ✅' if not issues else 'NEEDS IMPROVEMENT ⚠️'}")
    for issue in issues: print(f"    • {issue}")
    print(f"{sep}\n")
    if save_results:
        with open("evaluation/results.json","w") as f:
            json.dump({"summary":summary,"results":results},f,indent=2)
        print(f"  Results saved to evaluation/results.json")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="evaluation/test_queries.json")
    parser.add_argument("--session", default="eval_session")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    init_memory(); init_retrieval_store()
    import duckdb
    con   = duckdb.connect(os.environ.get("DB_PATH","memory.db"))
    count = con.execute("SELECT COUNT(*) FROM document_chunks_v2 WHERE session_id=? OR session_id='library'",[args.session]).fetchone()[0]
    con.close()
    if count == 0:
        print(f"\n⚠️  No chunks for session '{args.session}'. Running dry-run...\n")
    run_evaluation(args.queries, args.session, k=args.k, save_results=not args.no_save)
