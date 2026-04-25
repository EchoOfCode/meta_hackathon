from typing import Dict


def score_message_quality(action_text: str, event_id: str) -> float:
    """
    Lightweight communication-quality heuristic in [0, 1].
    Keeps environment framework-agnostic and cheap to run.
    """
    text = action_text.strip().lower()
    if not text:
        return 0.0

    score = 0.35
    if len(text.split()) >= 20:
        score += 0.20
    if any(k in text for k in ["thanks", "thank you", "appreciate"]):
        score += 0.10
    if any(k in text for k in ["by", "today", "tomorrow", "thursday", "pm", "am"]):
        score += 0.15
    if any(k in text for k in ["sorry sorry", "whatever", "later maybe"]):
        score -= 0.20
    if event_id == "E1_staging" and any(k in text for k in ["ssh", "restart", "logs", "runbook"]):
        score += 0.15
    return max(0.0, min(1.0, score))


def message_flags(action_text: str) -> Dict[str, bool]:
    text = action_text.lower()
    return {
        "contains_boundary": any(k in text for k in ["can't", "cannot", "skip", "decline", "not join"]),
        "contains_alternative": any(k in text for k in ["instead", "can do", "by ", "async", "swap"]),
    }
