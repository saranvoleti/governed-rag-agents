import os
import json
from anthropic import Anthropic
from tools.retrieve import retrieve, init_retrieval_store
from core.memory import init_memory, log_agent_run, get_history
from core.escalation import evaluate
from core.firewall import run_firewall, build_fallback

TOOLS = [
    {
        "name": "retrieve_from_document",
        "description": "Search the uploaded document for content relevant to the query. Always call this before answering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "check_memory",
        "description": "Check if this question was asked in a previous session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain to check"}
            },
            "required": ["domain"]
        }
    },
    {
        "name": "flag_for_review",
        "description": "Flag a finding for human review when confidence is low.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why this needs review"},
                "confidence": {"type": "number", "description": "Confidence score 0 to 1"}
            },
            "required": ["reason", "confidence"]
        }
    }
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
        top_k = 20 if "summar" in tool_input["query"].lower() else 5
        results = retrieve(tool_input["query"], session_id, top_k=top_k)
        if not results:
            return "No relevant content found in the document."
        return json.dumps([{"text": r["text"], "score": round(r["score"], 3)} for r in results])
    if tool_name == "check_memory":
        rows = get_history(domain=tool_input["domain"], limit=5)
        return f"Found {len(rows)} previous sessions." if rows else "No previous sessions found."
    if tool_name == "flag_for_review":
        decision = evaluate(domain, tool_input["confidence"], "analysis")
        return json.dumps({"flagged": True, "state": decision.state, "message": decision.message})
    return f"Unknown tool: {tool_name}"


def analyze(query, session_id, domain="healthcare", trace_callback=None, api_key=None):
    init_memory()

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=key)

    messages = [{"role": "user", "content": query}]
    tool_calls_log = []
    retrieved_chunks = []
    final_text = ""

    if trace_callback:
        trace_callback("agent_start", {"query": query})

    for _ in range(10):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_name = block.name
                tool_input = block.input

                if trace_callback:
                    trace_callback("tool_call", {"tool": tool_name, "input": tool_input})

                result = run_tool(tool_name, tool_input, session_id, domain)
                tool_calls_log.append({"tool": tool_name, "input": tool_input})

                if tool_name == "retrieve_from_document":
                    try:
                        retrieved_chunks = [c["text"] for c in json.loads(result)]
                    except Exception:
                        pass

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

                if trace_callback:
                    trace_callback("tool_result", {"tool": tool_name, "preview": result[:150]})

            messages.append({"role": "user", "content": tool_results})

    fallback_used = False

    fallback_used = False

    if trace_callback:
        trace_callback("firewall_check", {})

    firewall_result = run_firewall(final_text, retrieved_chunks)

    if not firewall_result.passed:
        if trace_callback:
            trace_callback("firewall_rejected", {"reason": firewall_result.reason})
        final_text = build_fallback(retrieved_chunks, query)
        firewall_result = run_firewall(final_text, retrieved_chunks, attempt=2, is_fallback=True)
        fallback_used = True

    if trace_callback:
        trace_callback("firewall_result", {"passed": firewall_result.passed, "reason": firewall_result.reason})

    confidence = 0.9 if firewall_result.passed else 0.5
    escalation = evaluate(domain, confidence, "analysis")

    log_agent_run(
        session_id=session_id,
        agent_id="analysis_agent",
        domain=domain,
        document_name="",
        input_data={"query": query},
        output_data={"answer": final_text[:200]},
        trust_score=confidence,
        escalate=escalation.escalate,
        reflection_attempts=1 if firewall_result.passed else 2,
        firewall_result=firewall_result.reason,
        tool_calls=tool_calls_log,
    )

    if trace_callback:
        trace_callback("complete", {"escalate": escalation.escalate, "confidence": confidence})

    return {
        "answer": final_text,
        "firewall_passed": firewall_result.passed,
        "firewall_reason": firewall_result.reason,
        "fallback_used": fallback_used,
        "tool_calls": tool_calls_log,
        "escalate": escalation.escalate,
        "escalation_message": escalation.message,
        "confidence": confidence,
        "retrieved_chunks": retrieved_chunks,
    }
