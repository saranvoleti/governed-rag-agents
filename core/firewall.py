"""
core/firewall.py
Day 9: grounding_score (float 0.0-1.0) added to FirewallResult.
       Dual gate: GROUNDING_THRESHOLD (0.60) + MAX_UNSUPPORTED (15).
"""
import re
from dataclasses import dataclass

FORBIDDEN_PATTERNS = [
    "research shows","studies indicate","evidence suggests",
    "according to medical literature","clinically proven","it is well known",
    "experts recommend","it is likely that","this implies","this suggests",
    "i recommend","i suggest","you should consider","patients typically",
    "in most cases","generally speaking","as a rule of thumb",
    "based on common practice",
]
MAX_WORDS=300; GROUNDING_THRESHOLD=0.60; MAX_UNSUPPORTED=15
SKIP_WORDS={"the","a","an","is","are","was","were","has","have","this","that",
    "these","those","it","in","on","at","to","for","of","with","by","from",
    "and","or","but","not","as","be","been","will","would","could","should",
    "may","document","states","indicates","notes","according","based","which",
    "also","their","they","them","its","more","than","then","when","what",
    "where","who","how"}

@dataclass
class FirewallResult:
    passed:bool; reason:str; check_failed:str=""; attempt:int=1
    fallback_used:bool=False; grounding_score:float=0.0

def check_forbidden_patterns(text):
    tl = text.lower()
    for p in FORBIDDEN_PATTERNS:
        if p in tl: return False, f"'{p}' -- LLM citing external knowledge"
    return True, "OK"

def check_length(text):
    c = len(text.split())
    return (False, f"{c} words exceeds {MAX_WORDS}") if c > MAX_WORDS else (True, "OK")

def check_grounded(response, chunks):
    if not chunks: return False, "No retrieved context", 0.0
    corpus = set(" ".join(chunks).lower().split())
    terms  = [w.lower().strip(".,;:!?\"'()*#_[]") for w in response.split()
              if len(w) > 4 and w.lower().strip(".,;:!?\"'()*#_[]") not in SKIP_WORDS]
    if not terms: return True, "OK", 1.0
    unsup = [t for t in terms if t not in corpus]
    score = round((len(terms) - len(unsup)) / len(terms), 4)
    if score < GROUNDING_THRESHOLD:
        return False, f"Grounding score {score:.2f} below threshold {GROUNDING_THRESHOLD}", score
    if len(unsup) > MAX_UNSUPPORTED:
        return False, f"{len(unsup)} unsupported terms exceeds {MAX_UNSUPPORTED}: {unsup[:4]}", score
    return True, "OK", score

def run_firewall(response, chunks, attempt=1, is_fallback=False):
    if is_fallback:
        return FirewallResult(passed=True, reason="Passed", grounding_score=1.0)
    passed, reason = check_forbidden_patterns(response)
    if not passed:
        return FirewallResult(passed=False, reason=reason,
                              check_failed="forbidden_patterns", attempt=attempt)
    passed, reason = check_length(response)
    if not passed:
        return FirewallResult(passed=False, reason=reason,
                              check_failed="length", attempt=attempt)
    passed, reason, score = check_grounded(response, chunks)
    if not passed:
        return FirewallResult(passed=False, reason=reason,
                              check_failed="grounded", attempt=attempt, grounding_score=score)
    return FirewallResult(passed=True, reason="Passed", attempt=attempt, grounding_score=score)

def build_fallback(chunks, query):
    if not chunks: return "No relevant content found in the uploaded document."
    return "[Verbatim extract from document]\n" + " ".join(re.split(r'(?<=[.!?])\s+', chunks[0])[:3])

def validate_with_budget(generate_fn, chunks, query, max_attempts=2):
    for attempt in range(1, max_attempts + 1):
        response = generate_fn(stricter=(attempt > 1))
        if response:
            result = run_firewall(response, chunks, attempt=attempt)
            if result.passed: return response, result
    fb = build_fallback(chunks, query)
    return fb, FirewallResult(passed=True, reason="Fallback used",
                              attempt=max_attempts, fallback_used=True, grounding_score=1.0)

if __name__ == "__main__":
    chunks = ["Patient requires prior authorization for the requested procedure. "
              "Policy number PA-2024-001 applies. Authorization must be obtained "
              "before the scheduled date of service."]
    good = ("The document states that prior authorization is required. "
            "Policy PA-2024-001 applies and authorization must be obtained before the scheduled date.")
    bad  = "Research shows prior authorization delays are common. Studies indicate patients typically wait."
    r1 = run_firewall(good, chunks)
    print(f"Good:    {'PASSED' if r1.passed else 'REJECTED'} grounding={r1.grounding_score}")
    r2 = run_firewall(bad, chunks)
    print(f"Bad:     {'PASSED' if r2.passed else 'REJECTED'} reason={r2.reason[:50]}")
    r3 = run_firewall(build_fallback(chunks,"test"), chunks, is_fallback=True)
    print(f"Fallbk:  {'PASSED' if r3.passed else 'REJECTED'} grounding={r3.grounding_score}")
    assert r1.passed and not r2.passed and r3.grounding_score == 1.0
    print("All firewall tests passed")
