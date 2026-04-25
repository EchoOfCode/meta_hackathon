from __future__ import annotations

from dataclasses import replace
import random
from typing import Dict

from .state import EpisodeState, RelationshipState, clamp01


def _copy_state(state: EpisodeState) -> EpisodeState:
    return replace(
        state,
        relationships=replace(state.relationships),
        decisions=[*state.decisions],
        history=[*state.history],
    )


def apply_action(state: EpisodeState, event_id: str, action_id: str, action_data: Dict) -> EpisodeState:
    new_state = _copy_state(state)

    new_state.energy = clamp01(state.energy + float(action_data.get("energy_delta", 0.0)))
    new_state.sprint_health = clamp01(state.sprint_health + float(action_data.get("sprint_delta", 0.0)))

    rel = new_state.relationships
    for person, delta in action_data.get("relationship_delta", {}).items():
        if hasattr(rel, person):
            setattr(rel, person, clamp01(getattr(rel, person) + float(delta)))
    new_state.relationships = RelationshipState(**vars(rel)).clamped()

    if event_id == "E1_staging" and action_id == "fix_directly":
        new_state.staging_fixed = True
    if event_id == "E4_leave":
        if action_id == "ping_sundar":
            new_state.leave_status = "approved" if random.random() > 0.20 else "pending"
        elif action_id == "cancel_leave":
            new_state.leave_status = "cancelled"
    if event_id == "E5_appraisal" and action_id == "block_wednesday":
        new_state.appraisal_done = True
    if event_id == "E6_oncall" and action_id == "yes_people_pleaser":
        new_state.oncall_accepted = True

    new_state.decisions.append(
        {
            "day": state.day,
            "event_id": event_id,
            "action_id": action_id,
            "energy_before": state.energy,
            "energy_after": new_state.energy,
        }
    )

    new_state.history.append(f"{event_id}:{action_id}")
    return new_state.normalized()
