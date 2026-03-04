"""
core/firewall.py

Validates LLM output against retrieved document chunks before
the response reaches the user. Three checks, two attempts,
deterministic fallback. Nothing published without passing this.
"""

import re
from dataclasses import dataclass


FORBIDDEN_PATTERNS = [
    "research shows", "studies indicate", "evidence suggests",
    "according to medical literature", "clinically proven",
    "it is well known", "experts recommend", "it is likely that",
    "this implies", "this suggests", "i recommend", "i suggest",
    "you should consider", "patients typically", "in most cases",
    "generally speaking", "as a rule of thumb", "based on common practice",
]

MAX_WORDS = 300

SKIP_WORDS = {
    "the","a","an","is","are","was","were","has","have","this","that",
    "these","those","it","in","on","at","to","for","of","with","by",
    "from","and","or","but","not","as","be","been","will","would",
    "could","should","may","document","states","indicates","notes",
    "according","based"
}


@dataclass
class FirewallResult:
    passed: bool
    reason: str
    check_failed: str = ""
    attempt: int = 1
    fallback_used: bool = False


def check_forbidden_patterns(text: str) -> tuple[bool, str]:
    text_lower = text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in text_lower:
            return False, f"'{pattern}' — LLM citing external knowledge, not the document"
    return True, "OK"


def check_length(text: str) -> tuple[bool, str]:
    count = len(text.split())
    if count > MAX_WORDS:
        return False, f"{count} words exceeds limit of {MAX_WORDS}"
    return True, "OK"


def check_grounded(response: str, chunks: list[str]) -> tuple[bool, str]:
    if not chunks:
        return False, "No retrieved context — cannot validate response"
    corpus = set(" ".join(chunks).lower().split())
    response_terms = set(
        w.lower().strip(".,;:!?\"'()")
        for w in response.split()
        if len(w) > 4 and w.lower() not in SKIP_WORDS
    )
    unsupported = response_terms - corpus
    if len(unsupported) > 6:
        sample = list(unsupported)[:4]
        return False, f"{len(unsupported)} terms not found in source document: {sample}"
    return True, "OK"


def run_firewall(response: str, chunks: list[str], attempt: int = 1) -> FirewallResult:
    checks = [
        ("forbidden_patterns", check_forbidden_patterns(response)),
        ("length",             check_length(response)),
        ("grounded",           check_grounded(response, chunks)),
    ]
    for name, (passed, reason) in checks:
        if not passed:
            return FirewallResult(passed=False, reason=reason,
                                  check_failed=name, attempt=attempt)
    return FirewallResult(passed=True, reason="Passed", attempt=attempt)


def build_fallback(chunks: list[str], query: str) -> str:
    if not chunks:
        return "No relevant content found in the uploaded document."
    sentences = re.split(r'(?<=[.!?])\s+', chunks[0])
    extract = " ".join(sentences[:3])
    return f"[Verbatim extract from document]\n{extract}"


def validate_with_budget(
    generate_fn,
    chunks: list[str],
    query: str,
    max_attempts: int = 2
) -> tuple[str, FirewallResult]:
    for attempt in range(1, max_attempts + 1):
        response = generate_fn(stricter=(attempt > 1))
        if response:
            result = run_firewall(response, chunks, attempt=attempt)
            if result.passed:
                return response, result
    fallback = build_fallback(chunks, query)
    return fallback, FirewallResult(
        passed=True, reason="Fallback used",
        attempt=max_attempts, fallback_used=True
    )


if __name__ == "__main__":
    chunks = [
        "Patient requires prior authorization for the requested procedure. "
        "Policy number PA-2024-001 applies. Authorization must be obtained "
        "before the scheduled date of service."
    ]
    good = (
        "The document states that prior authorization is required. "
        "Policy PA-2024-001 applies and authorization must be obtained "
        "before the scheduled date."
    )
    bad = "Research shows prior authorization delays are common. Studies indicate patients typically wait 3-5 days."

    r1 = run_firewall(good, chunks)
    print(f"Good response:     {'PASSED' if r1.passed else 'REJECTED'} — {r1.reason}")

    r2 = run_firewall(bad, chunks)
    print(f"Forbidden pattern: {'PASSED' if r2.passed else 'REJECTED'} — {r2.reason}")

    fb = build_fallback(chunks, "authorization policy")
    print(f"Fallback:          {fb[:80]}...")
