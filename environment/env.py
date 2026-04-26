from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

try:
    from openenv import Environment
except ImportError:  # local fallback for dev without openenv installed
    class Environment:  # type: ignore
        pass

from .consequences import apply_action
from .events import CANONICAL_EVENTS, Event
from .reward import annotate_decision, component_scores, weighted_total
from .state import EpisodeState


class WorkLifeFirewallEnv(Environment):
    metadata = {"name": "work-life-firewall", "version": "1.0.0", "max_steps": 14}

    def __init__(self, randomize_order: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.randomize_order = randomize_order
        self.rng = random.Random(seed)
        self._state = EpisodeState()
        self._events = []
        self._idx = 0

    def reset(self) -> Dict:
        self._state = EpisodeState()
        self._events = list(CANONICAL_EVENTS)
        if self.randomize_order:
            opener = self._events[0]
            tail = self._events[1:]
            self.rng.shuffle(tail)
            self._events = [opener] + tail
        self._idx = 0
        return self._build_obs(self._events[self._idx])

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        event = self._events[self._idx]
        action_id, action_data = self._decode_action(action, event)
        self._state = apply_action(self._state, event.id, action_id, action_data)

        # Add comm quality annotations into latest decision.
        latest = self._state.decisions[-1]
        annotate_decision(latest, action)
        step_reward = self._step_reward(latest, action_data)

        self._idx += 1
        self._state.day = min(4, self._idx // 2)

        done = self._idx >= len(self._events) or self._state.energy <= 0.05
        if done:
            comps = component_scores(self._state, self._state.decisions)
            terminal = self._calculate_terminal_reward()
            info = {"episode_complete": True, "components": comps}
            return self._state.to_observation(), round(step_reward + terminal, 4), True, info

        return self._build_obs(self._events[self._idx]), round(step_reward, 4), False, {}

    def state(self) -> Dict:
        return self._state.to_observation()

    def _build_obs(self, event: Event) -> Dict:
        return {
            "state": self._state.to_observation(),
            "event": {
                "id": event.id,
                "title": event.title,
                "source": event.source,
                "sender": event.sender,
                "body": event.body,
                "urgency": event.urgency,
                "relationship_stake": event.relationship_stake,
                "actions": [a["id"] for a in event.actions],
            },
            "instruction": "Write the response or action you would take. End with a specific next step.",
        }

    def _decode_action(self, free_text: str, event: Event) -> Tuple[str, Dict]:
        text = free_text.lower()
        decode_map = {
            "E1_staging": {"ssh": "fix_directly", "restart": "fix_directly", "delegate": "delegate_oncall", "infra": "escalate_infra"},
            "E2_slack": {"async": "async_boundary", "later": "async_boundary", "call now": "both_calls_now", "ignore": "ignore_slack"},
            "E3_client_email": {"thanks": "acknowledge_timeline", "thursday": "acknowledge_timeline", "sorry": "full_apology", "cc": "cc_manager"},
            "E4_leave": {"approve": "ping_sundar", "today": "ping_sundar", "wait": "wait_hope", "cancel": "cancel_leave"},
            "E5_appraisal": {"block": "block_wednesday", "calendar": "block_wednesday", "tonight": "rush_tonight", "extension": "ask_extension"},
            "E6_oncall": {"can't": "no_clearly_kindly", "cannot": "no_clearly_kindly", "swap": "counter_offer", "yes": "yes_people_pleaser", "sure": "yes_people_pleaser"},
            "E7_standup": {"async": "decline_async", "skip": "decline_async", "attend": "attend_sacrifice", "delegate": "delegate_teammate"},
        }
        mapping = decode_map.get(event.id, {})
        selected = event.actions[0]
        for keyword, action_id in mapping.items():
            if keyword in text:
                selected = next((a for a in event.actions if a["id"] == action_id), event.actions[0])
                break
        return selected["id"], selected

    def _step_reward(self, decision: Dict, action_data: Dict) -> float:
        reward = 0.0
        
        # Energy delta contributes to reward
        energy_delta = float(action_data.get("energy_delta", 0.0))
        reward += energy_delta * 0.5
        
        # Sprint delta contributes
        sprint_delta = float(action_data.get("sprint_delta", 0.0))
        reward += sprint_delta * 0.3
        
        # Relationship preservation contributes
        rel_sum = sum(abs(v) for v in action_data.get("relationship_delta", {}).values())
        reward -= rel_sum * 0.1
        
        # Comm bonus
        comm_bonus = decision.get("message_quality", 0.0) * 0.12
        reward += comm_bonus
        
        return reward

    def _calculate_terminal_reward(self) -> float:
        """Sparse terminal reward at episode end matching PRD storytelling."""
        reward = 0.0
        s = self._state
        
        if s.sprint_health >= 0.70: reward += 1.0
        if s.staging_fixed: reward += 0.3
        if s.leave_status == "approved": reward += 0.3
        if s.appraisal_done: reward += 0.2
        if s.energy >= 0.60: reward += 0.3
        if s.energy < 0.30: reward -= 0.5
        if s.relationships.david_chen >= 0.70: reward += 0.2
        if not s.oncall_accepted: reward += 0.15
        
        return reward
