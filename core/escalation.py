"""
core/escalation.py

Decides when to stop and ask a human.

Three states:
    CLEAR    - confident finding, publish
    WATCH    - low confidence, log and continue
    ESCALATE - persistent low confidence, pause for human
"""

from dataclasses import dataclass
from core.memory import init_memory, get_escalation_streak, update_escalation_streak

ESCALATE_AFTER   = 2
CONFIDENCE_FLOOR = 0.6


@dataclass
class EscalationDecision:
    state: str
    escalate: bool
    streak: int
    confidence: float
    reason: str
    message: str = ""


def evaluate(domain, confidence, finding_key="default"):
    domain_key = f"{domain}:{finding_key}"
    if confidence >= CONFIDENCE_FLOOR:
        update_escalation_streak(domain_key, confidence, increment=False)
        return EscalationDecision("CLEAR", False, 0, confidence, "Confidence above threshold")
    streak = update_escalation_streak(domain_key, confidence, increment=True)
    if streak < ESCALATE_AFTER:
        return EscalationDecision("WATCH", False, streak, confidence, f"Low confidence ({confidence:.2f}), monitoring")
    return EscalationDecision(
        state="ESCALATE", escalate=True, streak=streak, confidence=confidence,
        reason=f"Low confidence for {streak} consecutive findings",
        message=f"Agent confidence was {confidence:.0%}. Please review the source document and confirm."
    )


def reset(domain, finding_key="default"):
    update_escalation_streak(f"{domain}:{finding_key}", 1.0, increment=False)


if __name__ == "__main__":
    init_memory()
    print("Testing escalation...\n")
    for i, score in enumerate([0.85, 0.45, 0.40, 0.38]):
        d = evaluate("healthcare", score, "coverage_criteria")
        print(f"  Finding {i+1}: confidence={score:.2f}  state={d.state}")
        if d.escalate:
            print(f"  ESCALATE: {d.message}")
    print("\nEscalation test passed")
