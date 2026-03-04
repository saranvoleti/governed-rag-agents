import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from tools.retrieve import retrieve, init_retrieval_store
from core.memory import init_memory, log_agent_run, get_history
from core.escalation import evaluate
from core.firewall import run_firewall, build_fallback

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
- If the document does not contain the answer, say so clearly
- If you are uncertain, call flag_for_review with your confidence score
- Never use outside knowledge
- Keep answers concise and factual"""


def run_tool(tool_name, tool_input, session_id, domain):
    if tool_name == "retrieve_from_document":
        results = retrieve(tool_input["query"], session_id, top_k=5)
        if not results:
            return "No relevant content found in the document."
        return json.dumps([{"text": r["text"], "score": round(r["score"], 3)} for r in results])

    if tool_name == "check_memory":
        rows = get_history(domain=tool_input["domain"], limit=5)
        if not rows:
            return "No previous sessions found."
        return f"Found {len(rows)} previous sessions for domain: {tool_input['domain']}"

    if tool_name == "flag_for_review":
        decision = evaluate(domain, tool_input["confidence"], "analysis")
        return json.dumps({
            "flagged": True,
            "state": decision.state,
            "message": decision.message,
            "reason": tool_input["reason"]
        })

    return f"Unknown tool: {tool_name}"


def analyze(query, session_id, domain="healthcare", trace_callback=None):
    init_memory()
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
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input

                if trace_callback:
                    trace_callback("tool_call", {"tool": tool_name, "input": tool_input})

                result = run_tool(tool_name, tool_input, session_id, domain)
                tool_calls_log.append({"tool": tool_name, "input": tool_input})

                if tool_name == "retrieve_from_document":
                    try:
                        retrieved_chunks = [c["text"] for c in json.loads(result)]
                    except:
                        pass

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

                if trace_callback:
                    trace_callback("tool_result", {"tool": tool_name, "preview": result[:150]})

            messages.append({"role": "user", "content": tool_results})

    if trace_callback:
        trace_callback("firewall_check", {})

    firewall_result = run_firewall(final_text, retrieved_chunks)

    if not firewall_result.passed:
        if trace_callback:
            trace_callback("firewall_rejected", {"reason": firewall_result.reason})
        final_text = build_fallback(retrieved_chunks, query)
        firewall_result = run_firewall(final_text, retrieved_chunks, attempt=2)

    if trace_callback:
        trace_callback("firewall_result", {
            "passed": firewall_result.passed,
            "reason": firewall_result.reason
        })

    confidence = 0.9 if firewall_result.passed and not firewall_result.fallback_used else 0.5
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
        trace_callback("complete", {
            "escalate": escalation.escalate,
            "confidence": confidence
        })

    return {
        "answer": final_text,
        "firewall_passed": firewall_result.passed,
        "firewall_reason": firewall_result.reason,
        "fallback_used": firewall_result.fallback_used,
        "tool_calls": tool_calls_log,
        "escalate": escalation.escalate,
        "escalation_message": escalation.message,
        "confidence": confidence,
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    import tempfile, os
    init_retrieval_store()

    sample = """
    Medicare Part B covers medically necessary outpatient services.
    Prior authorization is required for advanced imaging including MRI, CT, and PET scans.
    Policy CMS-2024-PA-001 requires Form CMS-1450 with clinical documentation.
    Requests must be submitted 5 business days before the service date.
    Urgent requests may be submitted 24 hours in advance with physician attestation.
    Appeals must be filed within 120 days of initial determination.
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample)
        tmp = f.name

    from tools.retrieve import ingest_document
    ingest_document(tmp, "txt", "test_001", "cms_policy.txt")
    os.unlink(tmp)

    def trace(event, data):
        print(f"  [{event}] {str(data)[:120]}")

    print("Running analysis agent...\n")
    result = analyze(
        query="What is required to get prior authorization?",
        session_id="test_001",
        domain="healthcare",
        trace_callback=trace
    )

    print(f"\nAnswer:   {result['answer']}")
    print(f"Firewall: {result['firewall_reason']}")
    print(f"Escalate: {result['escalate']}")
    print(f"Tools:    {len(result['tool_calls'])} calls")
