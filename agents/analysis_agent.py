"""
agents/analysis_agent.py
Day 9: per-stage latency (retrieval_ms, generation_ms, firewall_ms, total_ms).
       grounding_score float logged to memory.db via notes field.
"""
import os, json, time
from anthropic import Anthropic
from tools.retrieve import retrieve, init_retrieval_store
from core.memory import init_memory, log_agent_run, get_history
from core.escalation import evaluate
from core.firewall import run_firewall, build_fallback

TOOLS = [
    {"name":"retrieve_from_document",
     "description":"Search the uploaded document for content relevant to the query. Always call this before answering.",
     "input_schema":{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}},
    {"name":"check_memory",
     "description":"Check if this question was asked in a previous session.",
     "input_schema":{"type":"object","properties":{"domain":{"type":"string","description":"Domain to check"}},"required":["domain"]}},
    {"name":"flag_for_review",
     "description":"Flag a finding for human review when confidence is low.",
     "input_schema":{"type":"object","properties":{"reason":{"type":"string","description":"Why this needs review"},"confidence":{"type":"number","description":"Confidence score 0 to 1"}},"required":["reason","confidence"]}}
]

SYSTEM_PROMPT = """You are a document analysis agent. Answer questions strictly based on the uploaded document.
Rules:
- Always call retrieve_from_document before answering
- Only state what the document explicitly says
- If uncertain, call flag_for_review
- Never use outside knowledge
- Keep answers concise and factual"""

def run_tool(tool_name, tool_input, session_id, domain):
    if tool_name == "retrieve_from_document":
        top_k   = 20 if "summar" in tool_input["query"].lower() else 5
        results = retrieve(tool_input["query"], session_id, top_k=top_k)
        if not results: return "No relevant content found in the document."
        return json.dumps([{"text":r["text"],"score":round(r["score"],3)} for r in results])
    if tool_name == "check_memory":
        rows = get_history(domain=tool_input["domain"], limit=5)
        return f"Found {len(rows)} previous sessions." if rows else "No previous sessions found."
    if tool_name == "flag_for_review":
        d = evaluate(domain, tool_input["confidence"], "analysis")
        return json.dumps({"flagged":True,"state":d.state,"message":d.message})
    return f"Unknown tool: {tool_name}"

def analyze(query, session_id, domain="healthcare", trace_callback=None, api_key=None):
    """
    Returns: answer, firewall_passed, firewall_reason, fallback_used,
             tool_calls, escalate, escalation_message, confidence,
             retrieved_chunks, grounding_score,
             latency: {retrieval_ms, generation_ms, firewall_ms, total_ms}
    """
    init_memory()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    if not key: raise ValueError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=key)

    t_total   = time.perf_counter()
    messages  = [{"role":"user","content":query}]
    tool_log  = []; chunks = []; final_text = ""
    ret_ms    = 0.0; gen_ms = 0.0

    if trace_callback: trace_callback("agent_start", {"query":query})

    t_gen = time.perf_counter()
    for _ in range(10):
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=1000,
                                      system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        if resp.stop_reason == "end_turn":
            final_text = next((b.text for b in resp.content if hasattr(b,"text")), "")
            gen_ms     = round((time.perf_counter() - t_gen) * 1000, 1)
            break
        if resp.stop_reason == "tool_use":
            messages.append({"role":"assistant","content":resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use": continue
                if trace_callback: trace_callback("tool_call",{"tool":block.name,"input":block.input})
                if block.name == "retrieve_from_document":
                    t_r    = time.perf_counter()
                    result = run_tool(block.name, block.input, session_id, domain)
                    ret_ms += round((time.perf_counter() - t_r) * 1000, 1)
                    try: chunks = [c["text"] for c in json.loads(result)]
                    except: pass
                else:
                    result = run_tool(block.name, block.input, session_id, domain)
                tool_log.append({"tool":block.name,"input":block.input})
                results.append({"type":"tool_result","tool_use_id":block.id,"content":result})
                if trace_callback: trace_callback("tool_result",{"tool":block.name,"preview":result[:150]})
            messages.append({"role":"user","content":results})

    fallback_used = False
    if trace_callback: trace_callback("firewall_check", {})
    t_fw = time.perf_counter()
    fw   = run_firewall(final_text, chunks)
    fw_ms = round((time.perf_counter() - t_fw) * 1000, 1)

    if not fw.passed:
        if trace_callback: trace_callback("firewall_rejected",{"reason":fw.reason})
        final_text    = build_fallback(chunks, query)
        fw            = run_firewall(final_text, chunks, attempt=2, is_fallback=True)
        fallback_used = True

    if trace_callback:
        trace_callback("firewall_result",{"passed":fw.passed,"reason":fw.reason,"grounding_score":fw.grounding_score})

    confidence = 0.9 if fw.passed and not fallback_used else 0.5
    escalation = evaluate(domain, confidence, "analysis")
    total_ms   = round((time.perf_counter() - t_total) * 1000, 1)
    latency    = {"retrieval_ms":ret_ms,"generation_ms":gen_ms,"firewall_ms":fw_ms,"total_ms":total_ms}

    if trace_callback:
        trace_callback("complete",{"escalate":escalation.escalate,"confidence":confidence,
                                   "grounding_score":fw.grounding_score,"latency":latency})

    log_agent_run(session_id=session_id, agent_id="analysis_agent", domain=domain,
                  document_name="", input_data={"query":query},
                  output_data={"answer":final_text[:200]}, trust_score=confidence,
                  escalate=escalation.escalate, reflection_attempts=1 if not fallback_used else 2,
                  firewall_result=fw.reason, tool_calls=tool_log,
                  notes=json.dumps({"grounding_score":fw.grounding_score,"latency":latency}))

    return {"answer":final_text,"firewall_passed":fw.passed,"firewall_reason":fw.reason,
            "fallback_used":fallback_used,"tool_calls":tool_log,"escalate":escalation.escalate,
            "escalation_message":escalation.message,"confidence":confidence,
            "retrieved_chunks":chunks,"grounding_score":fw.grounding_score,"latency":latency}
