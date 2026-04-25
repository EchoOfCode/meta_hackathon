from __future__ import annotations

from typing import Dict, List

from .judge import message_flags, score_message_quality
from .state import EpisodeState


def technical_resolution_score(state: EpisodeState) -> float:
    score = 0.0
    if state.staging_fixed:
        score += 0.5
    if state.sprint_health >= 0.70:
        score += 0.5
    return min(1.0, score)


def communication_quality_score(decisions: List[Dict]) -> float:
    if not decisions:
        return 0.0
    vals = [d.get("message_quality", 0.0) for d in decisions]
    return min(1.0, max(0.0, sum(vals) / len(vals)))


def boundary_setting_score(decisions: List[Dict]) -> float:
    if not decisions:
        return 0.0
    strategic = {"no_clearly_kindly", "decline_async", "async_boundary", "counter_offer"}
    people_pleasing = {"yes_people_pleaser", "attend_sacrifice", "both_calls_now"}
    held = sum(1 for d in decisions if d["action_id"] in strategic)
    leaked = sum(1 for d in decisions if d["action_id"] in people_pleasing)
    return max(0.0, min(1.0, held / 4.0 - 0.15 * leaked))


def energy_to_friday_score(state: EpisodeState) -> float:
    return state.energy


def relationship_preservation_score(state: EpisodeState) -> float:
    rels = vars(state.relationships)
    mean_rel = sum(rels.values()) / len(rels)
    client = state.relationships.david_chen
    return max(0.0, min(1.0, 0.6 * mean_rel + 0.4 * client))


def component_scores(state: EpisodeState, decisions: List[Dict]) -> Dict[str, float]:
    return {
        "technical_resolution": technical_resolution_score(state),
        "communication_quality": communication_quality_score(decisions),
        "boundary_setting": boundary_setting_score(decisions),
        "energy_to_friday": energy_to_friday_score(state),
        "relationship_preservation": relationship_preservation_score(state),
    }


def weighted_total(scores: Dict[str, float]) -> float:
    weights = {
        "technical_resolution": 0.25,
        "communication_quality": 0.25,
        "boundary_setting": 0.20,
        "energy_to_friday": 0.20,
        "relationship_preservation": 0.10,
    }
    return sum(scores[k] * w for k, w in weights.items())


def annotate_decision(decision: Dict, free_text: str) -> Dict:
    decision["message_quality"] = score_message_quality(free_text, decision["event_id"])
    decision.update(message_flags(free_text))
    return decision
