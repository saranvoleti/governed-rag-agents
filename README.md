# governed-rag-agents

Governed multi-agent RAG platform with agentic tool use, persistent memory,
reflection loops, and human-in-the-loop escalation. Deployed on Azure.

## Agentic Patterns
- Tool use — Claude calls tools autonomously
- Multi-agent orchestration — governed handoff contracts
- Memory — Citta persists across sessions (Azure Blob)
- Reflection — self-healing validation loop
- Human-in-the-loop — escalation pauses for human confirmation
- Agentic RAG — agent decides when to retrieve

## Azure Stack
- Azure Container Apps + Container Registry
- Azure Blob Storage (persistent memory)
- Azure Key Vault (secrets)
- Azure OpenAI (LLM backend)
- GitHub Actions (CI/CD)
