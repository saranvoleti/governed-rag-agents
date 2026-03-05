import streamlit as st
import uuid
import os
import sys
import tempfile

# Add project root to path so imports work on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key — Streamlit secrets in cloud, environment variable locally
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

from tools.retrieve import init_retrieval_store, ingest_document
from agents.analysis_agent import analyze
from core.memory import init_memory, get_history

st.set_page_config(
    page_title="Governed RAG Agents",
    page_icon="🔍",
    layout="wide"
)

st.title("Governed RAG Agents")
st.caption("Multi-agent document analysis with validation and human-in-the-loop escalation")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "trace" not in st.session_state:
    st.session_state.trace = []
if "result" not in st.session_state:
    st.session_state.result = None

init_retrieval_store()
init_memory()

with st.sidebar:
    st.header("Session")
    st.code(f"ID: {st.session_state.session_id}")

    st.header("Upload Document")
    uploaded = st.file_uploader("PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])

    if uploaded:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded.name.split('.')[-1]}"
        ) as f:
            f.write(uploaded.read())
            tmp_path = f.name

        file_type = uploaded.name.split(".")[-1].lower()

        with st.spinner("Processing document..."):
            chunks = ingest_document(
                tmp_path, file_type,
                st.session_state.session_id,
                uploaded.name
            )
        os.unlink(tmp_path)
        st.session_state.document_loaded = True
        st.session_state.document_name = uploaded.name
        st.success(f"Loaded: {uploaded.name} ({chunks} chunks)")

    domain = st.selectbox("Domain", ["healthcare", "finance", "legal", "ops", "general"])

    st.header("Memory")
    if st.button("View session history"):
        rows = get_history(session_id=st.session_state.session_id)
        st.write(f"{len(rows)} runs this session")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Ask a Question")

    if not st.session_state.document_loaded:
        st.info("Upload a document in the sidebar to get started.")
        st.markdown("**Try these public documents:**")
        st.markdown("- [CMS Prior Auth Policy](https://www.cms.gov)")
        st.markdown("- [CDC Clinical Guidelines](https://www.cdc.gov)")
        st.markdown("- [FDA Drug Labels](https://www.fda.gov)")
    else:
        st.success(f"Document ready: {st.session_state.document_name}")

    query = st.text_area(
        "Your question",
        placeholder="What are the prior authorization requirements?",
        height=100
    )

    run = st.button(
        "Analyze",
        disabled=not st.session_state.document_loaded,
        type="primary"
    )

    if run and query:
        st.session_state.trace = []
        st.session_state.result = None
        trace_container = st.empty()

        def on_trace(event, data):
            icons = {
                "agent_start":       "🚀",
                "tool_call":         "🔧",
                "tool_result":       "📄",
                "firewall_check":    "🛡️",
                "firewall_rejected": "❌",
                "firewall_result":   "✅",
                "complete":          "🏁",
            }
            icon = icons.get(event, "•")
            st.session_state.trace.append(f"{icon} **{event}** — {str(data)[:100]}")
            trace_container.markdown("\n\n".join(st.session_state.trace))

        with st.spinner("Agents working..."):
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            result = analyze(
                query=query,
                session_id=st.session_state.session_id,
                domain=domain,
                trace_callback=on_trace,
                api_key=api_key
            )
            st.session_state.result = result

with col2:
    st.header("Agent Trace")

    if st.session_state.trace:
        for line in st.session_state.trace:
            st.markdown(line)
    else:
        st.caption("Agent steps appear here in real time")

    if st.session_state.result:
        r = st.session_state.result
        st.divider()
        st.header("Answer")

        if r["escalate"]:
            st.warning(f"⚠️ Human Review Required\n\n{r['escalation_message']}")

        st.markdown(r["answer"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{r['confidence']:.0%}")
        c2.metric("Firewall", "Passed" if r["firewall_passed"] else "Fallback")
        c3.metric("Tool Calls", len(r["tool_calls"]))

        if r["fallback_used"]:
            st.info("Verbatim extract returned — LLM answer did not pass validation")

        with st.expander("Retrieved chunks"):
            for i, chunk in enumerate(r["retrieved_chunks"]):
                st.markdown(f"**Chunk {i+1}**")
                st.caption(chunk[:300])
