# Detailed Implementation Plan
## Work-Life Firewall — OpenEnv RL Environment

---

## Phase 0 — Setup (Day 0, ~2 hours)

### 0.1 Repository scaffold

```bash
mkdir work-life-firewall && cd work-life-firewall
git init
python -m venv .venv && source .venv/bin/activate
pip install openenv
```

### 0.2 openenv.yaml manifest

```yaml
name: work-life-firewall
version: 1.0.0
description: >
  An RL environment where an LLM agent navigates a real Indian software
  engineer's work week — balancing sprint delivery, client relationships,
  peer requests, and personal boundaries under energy and time constraints.
theme: "3.2 - Personalized Tasks"
author: "<your-name>"
python: ">=3.10"
dependencies:
  - openenv
  - numpy
  - pydantic>=2.0
entry_point: environment.env:WorkLifeFirewallEnv
```

---

## Phase 1 — Environment Core (Day 1, ~6 hours)

### 1.1 State definition (`environment/state.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class Day(int, Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4

@dataclass
class RelationshipState:
    david_chen: float = 0.75      # US client (starts tense)
    sundar: float = 0.80          # Manager
    priya: float = 0.90           # Teammate
    rahul: float = 0.88           # Teammate
    arjun: float = 0.82           # On-call requester

@dataclass
class EpisodeState:
    day: int = 0                  # 0=Mon, 4=Fri
    energy: float = 0.87          # Starting energy (Mon morning)
    sprint_health: float = 0.80
    staging_fixed: bool = False
    leave_status: str = "pending" # pending / approved / cancelled
    appraisal_done: bool = False
    standup_attending: bool = True # default: assumed mandatory
    oncall_accepted: bool = False
    relationships: RelationshipState = field(default_factory=RelationshipState)
    decisions: List[dict] = field(default_factory=list)
    pending_events: List[str] = field(default_factory=list)
    spawned_events: List[dict] = field(default_factory=list)
    
    def to_observation(self) -> dict:
        """Convert to agent-readable observation dict."""
        return {
            "day": ["Monday","Tuesday","Wednesday","Thursday","Friday"][self.day],
            "energy_pct": round(self.energy * 100),
            "sprint_health_pct": round(self.sprint_health * 100),
            "staging_fixed": self.staging_fixed,
            "leave_status": self.leave_status,
            "appraisal_done": self.appraisal_done,
            "pending_events": self.pending_events,
            "relationships": {
                k: round(v, 2) 
                for k, v in vars(self.relationships).items()
            }
        }
```

### 1.2 Event catalogue (`environment/events.py`)

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Event:
    id: str
    title: str
    source: str           # slack / email / calendar / pagerduty / hr
    sender: str
    body: str
    urgency: str          # critical / high / medium / low
    relationship_stake: str  # client / manager / peer / hr / personal
    energy_if_accepted: float  # delta to energy (negative = costs)
    energy_if_declined: float  # delta to energy (positive = saves)
    sprint_impact: float       # delta to sprint_health if unhandled
    actions: List[Dict]

CANONICAL_EVENTS = [
    Event(
        id="E1_staging",
        title="Staging server down — deploy blocked",
        source="pagerduty",
        sender="PagerDuty",
        body="""ALERT: staging-prod-1 disk at 98%. Last deploy wrote 14GB unrotated logs. 
Four engineers blocked. Sprint demo is Friday.""",
        urgency="critical",
        relationship_stake="team",
        energy_if_accepted=-0.08,   # costs energy but unblocks team
        energy_if_declined=-0.15,   # costs more (downstream chaos)
        sprint_impact=-0.25,        # if not fixed, sprint health drops
        actions=[
            {"id": "fix_directly", "label": "SSH in, clear logs, write runbook", 
             "energy_delta": -0.08, "sprint_delta": +0.20, "relationship_delta": {"team": +0.05}},
            {"id": "delegate_oncall", "label": "Assign to on-call rotation",
             "energy_delta": -0.02, "sprint_delta": +0.10, "relationship_delta": {"team": -0.02}},
            {"id": "escalate_infra", "label": "Escalate to infra team",
             "energy_delta": -0.12, "sprint_delta": +0.05, "relationship_delta": {"team": -0.05}},
        ]
    ),

    Event(
        id="E2_slack",
        title="3 unread Slack messages — Priya, Rahul, #general",
        source="slack",
        sender="Multiple",
        body="""Priya (9:02): "are you around? got blocked on auth module, need ~20 min"
Rahul (9:15): "quick question about API spec — hop on a call?"
#general: Rohan got kudos for the last release.""",
        urgency="medium",
        relationship_stake="peer",
        energy_if_accepted=-0.22,
        energy_if_declined=+0.06,
        sprint_impact=-0.05,
        actions=[
            {"id": "async_boundary", "label": "Reply async, set time boundary (12pm)",
             "energy_delta": +0.06, "sprint_delta": 0, "relationship_delta": {"priya": 0, "rahul": 0}},
            {"id": "both_calls_now", "label": "Jump on calls with both immediately",
             "energy_delta": -0.22, "sprint_delta": 0, "relationship_delta": {"priya": +0.03, "rahul": +0.03}},
            {"id": "ignore_slack", "label": "Mark as read, handle later",
             "energy_delta": -0.05, "sprint_delta": 0, "relationship_delta": {"priya": -0.03, "rahul": -0.03},
             "spawns": "E2b_slack_followup"},
        ]
    ),

    Event(
        id="E3_client_email",
        title="Re: Re: Re: [URGENT] dashboard latency — David Chen",
        source="email",
        sender="david.chen@meridian.com",
        body="""Hi — following up AGAIN on the dashboard latency issue I raised two weeks ago. 
My team is losing confidence. I'm not sure what's happening on your end but this doesn't 
reflect well on our partnership going forward. Please advise.

[Previous thread: your update sent 6 days ago. He read it. Never replied until now.]""",
        urgency="high",
        relationship_stake="client",
        energy_if_accepted=-0.05,
        energy_if_declined=-0.10,
        sprint_impact=0,
        actions=[
            {"id": "acknowledge_timeline", 
             "label": "Acknowledge, set clear next step + timeline (Thu update)",
             "energy_delta": +0.04, "sprint_delta": 0, 
             "relationship_delta": {"david_chen": +0.08}},
            {"id": "full_apology", "label": "Full apology, promise quick fix",
             "energy_delta": -0.10, "sprint_delta": -0.05, 
             "relationship_delta": {"david_chen": -0.05},
             "spawns": "E3b_client_escalation"},
            {"id": "cc_manager", "label": "Loop in manager, reply professionally",
             "energy_delta": -0.02, "sprint_delta": 0, 
             "relationship_delta": {"david_chen": +0.02, "sundar": -0.03}},
        ]
    ),

    Event(
        id="E4_leave",
        title="Thursday leave — pending approval for 3 months",
        source="calendar",
        sender="HR System",
        body="""Your leave application for Thursday (family wedding) has been pending for 
3 months. Manager Sundar has seen it but not approved or declined. 
Sprint demo is Friday. Thursday is the day before.""",
        urgency="high",
        relationship_stake="manager",
        energy_if_accepted=+0.08,
        energy_if_declined=-0.25,
        sprint_impact=-0.10,
        actions=[
            {"id": "ping_sundar", "label": "Ping Sundar directly, request decision today",
             "energy_delta": +0.10, "sprint_delta": 0, 
             "relationship_delta": {"sundar": 0}},
            {"id": "wait_hope", "label": "Wait and hope he approves",
             "energy_delta": -0.15, "sprint_delta": 0, 
             "relationship_delta": {"sundar": 0},
             "spawns": "E4b_leave_anxiety"},
            {"id": "cancel_leave", "label": "Cancel leave to look committed",
             "energy_delta": -0.25, "sprint_delta": +0.05, 
             "relationship_delta": {"sundar": +0.01}},
        ]
    ),

    Event(
        id="E5_appraisal",
        title="Annual appraisal form — due Friday, 12 sections, ~90 min",
        source="hr",
        sender="HR Portal",
        body="""Your annual appraisal is due Friday. 12 sections including self-rating, 
peer nominations, goals review, and a 500-word impact statement. 
Last appraisal was strong. This determines your promo cycle.""",
        urgency="high",
        relationship_stake="hr",
        energy_if_accepted=-0.08,
        energy_if_declined=-0.15,
        sprint_impact=0,
        actions=[
            {"id": "block_wednesday", "label": "Block 90 min Wednesday, do it properly",
             "energy_delta": +0.04, "sprint_delta": 0, "relationship_delta": {}},
            {"id": "rush_tonight", "label": "Rush through it tonight after standup",
             "energy_delta": -0.15, "sprint_delta": 0, "relationship_delta": {}},
            {"id": "ask_extension", "label": "Email HR for 1-week extension",
             "energy_delta": -0.03, "sprint_delta": 0, "relationship_delta": {},
             "spawns": "E5b_extension_denied"},
        ]
    ),

    Event(
        id="E6_oncall",
        title="Arjun asking you to cover Wednesday night on-call",
        source="slack",
        sender="arjun",
        body="""Arjun: "hey bro, small favour — any chance you can cover my Wednesday night 
on-call? got something come up. will return the favour"

[Context: 3rd time this quarter. You covered the last two. Wednesday night you have 
the US standup at 10:30 PM and sprint crunch approaching.]""",
        urgency="low",
        relationship_stake="peer",
        energy_if_accepted=-0.28,
        energy_if_declined=+0.12,
        sprint_impact=-0.15,
        actions=[
            {"id": "no_clearly_kindly", "label": "Say no clearly and kindly, offer Thursday swap",
             "energy_delta": +0.12, "sprint_delta": 0, 
             "relationship_delta": {"arjun": -0.03}},
            {"id": "yes_people_pleaser", "label": "Say yes",
             "energy_delta": -0.28, "sprint_delta": -0.15, 
             "relationship_delta": {"arjun": +0.05},
             "spawns": "E6b_wednesday_collapse"},
            {"id": "counter_offer", "label": "Counter: cover Thursday instead",
             "energy_delta": +0.02, "sprint_delta": 0, 
             "relationship_delta": {"arjun": 0}},
        ]
    ),

    Event(
        id="E7_standup",
        title="US client standup — tonight 10:30 PM (you're optional)",
        source="calendar",
        sender="meridian_zoom",
        body="""Recurring Zoom — Meridian Corp standup. You were added as 'optional' 6 months ago.
Attended once. Now David's team treats it as mandatory. Runs 45-60 min. 
You have no direct action items. You're there for 'visibility.'""",
        urgency="medium",
        relationship_stake="client",
        energy_if_accepted=-0.30,
        energy_if_declined=+0.14,
        sprint_impact=0,
        actions=[
            {"id": "decline_async", "label": "Decline, offer async written update by 10 PM",
             "energy_delta": +0.14, "sprint_delta": 0, 
             "relationship_delta": {"david_chen": +0.02}},
            {"id": "attend_sacrifice", "label": "Attend (sacrifice sleep)",
             "energy_delta": -0.30, "sprint_delta": -0.10, 
             "relationship_delta": {"david_chen": +0.03}},
            {"id": "delegate_teammate", "label": "Ask teammate to represent you",
             "energy_delta": +0.05, "sprint_delta": 0, 
             "relationship_delta": {"david_chen": 0, "priya": -0.02}},
        ]
    ),
]
```

### 1.3 Consequence engine (`environment/consequences.py`)

```python
from typing import List, Tuple
from .state import EpisodeState
import random

class ConsequenceEngine:
    """
    Applies action consequences to state.
    Handles immediate effects + spawned events.
    """
    
    def apply(self, state: EpisodeState, event_id: str, 
              action_id: str, action_data: dict) -> Tuple[EpisodeState, List[dict]]:
        
        new_state = self._copy_state(state)
        spawned = []
        
        # Apply energy delta
        new_state.energy = max(0.0, min(1.0, 
            state.energy + action_data.get("energy_delta", 0)))
        
        # Apply sprint delta
        new_state.sprint_health = max(0.0, min(1.0,
            state.sprint_health + action_data.get("sprint_delta", 0)))
        
        # Apply relationship deltas
        rel_deltas = action_data.get("relationship_delta", {})
        for person, delta in rel_deltas.items():
            current = getattr(new_state.relationships, person, 0.5)
            setattr(new_state.relationships, person, 
                    max(0.0, min(1.0, current + delta)))
        
        # Handle event-specific state updates
        if event_id == "E1_staging" and action_id == "fix_directly":
            new_state.staging_fixed = True
            
        if event_id == "E4_leave":
            if action_id == "ping_sundar":
                # Simulate manager response with stochastic outcome
                new_state.leave_status = "approved" if random.random() > 0.15 else "pending"
            elif action_id == "cancel_leave":
                new_state.leave_status = "cancelled"
                
        if event_id == "E5_appraisal" and action_id == "block_wednesday":
            new_state.appraisal_done = True
            
        if event_id == "E6_oncall" and action_id == "yes_people_pleaser":
            new_state.oncall_accepted = True
            # Spawn Wednesday collapse event
            spawned.append({
                "day": 2,  # Wednesday
                "event_id": "E6b_wednesday_collapse",
                "description": "On-call alert fires at 11:40 PM. You're exhausted."
            })
            
        if event_id == "E7_standup" and action_id == "attend_sacrifice":
            # Compound: next day starts with reduced energy
            spawned.append({
                "day": 1,  # Tuesday morning
                "event_id": "E7b_sleep_debt",
                "description": "Standup ran to 11:50 PM. -15% energy Tuesday."
            })
        
        # Record decision
        new_state.decisions.append({
            "day": state.day,
            "event_id": event_id,
            "action_id": action_id,
            "energy_before": state.energy,
            "energy_after": new_state.energy,
        })
        
        return new_state, spawned
```

### 1.4 Reward rubric (`environment/reward.py`)

```python
from openenv import Rubric, Component

def build_rubric() -> Rubric:
    return Rubric([
        Component(
            name="technical_resolution",
            weight=0.25,
            scorer=technical_resolution_scorer,
        ),
        Component(
            name="communication_quality",
            weight=0.25,
            scorer=communication_quality_scorer,
        ),
        Component(
            name="boundary_setting",
            weight=0.20,
            scorer=boundary_setting_scorer,
        ),
        Component(
            name="energy_to_friday",
            weight=0.20,
            scorer=energy_to_friday_scorer,
        ),
        Component(
            name="relationship_preservation",
            weight=0.10,
            scorer=relationship_preservation_scorer,
        ),
    ])

def technical_resolution_scorer(state, decisions) -> float:
    score = 0.0
    if state.staging_fixed:
        score += 0.5
    if state.sprint_health >= 0.70:
        score += 0.5
    return score

def communication_quality_scorer(state, decisions) -> float:
    """
    Score the quality of drafted responses.
    Uses keyword heuristics (fast) or LLM judge (accurate).
    """
    good_responses = [d for d in decisions 
                      if d["action_id"] in ["acknowledge_timeline", 
                                             "async_boundary",
                                             "no_clearly_kindly",
                                             "ping_sundar",
                                             "decline_async",
                                             "block_wednesday"]]
    return min(1.0, len(good_responses) / 5.0)

def boundary_setting_scorer(state, decisions) -> float:
    """
    Reward quality no's. Penalise yes-to-everything AND no-to-everything.
    The agent should decline strategic items, not all items.
    """
    boundary_actions = {"no_clearly_kindly", "decline_async", 
                        "async_boundary", "acknowledge_timeline"}
    people_pleaser = {"yes_people_pleaser", "attend_sacrifice", 
                      "both_calls_now"}
    
    boundaries_held = sum(1 for d in decisions 
                          if d["action_id"] in boundary_actions)
    yes_to_everything = sum(1 for d in decisions 
                            if d["action_id"] in people_pleaser)
    
    # Ideal: 3-4 boundaries held, 0-1 people-pleaser decisions
    score = min(1.0, boundaries_held / 4.0)
    score -= yes_to_everything * 0.15
    return max(0.0, score)

def energy_to_friday_scorer(state, decisions) -> float:
    """Friday energy is the primary wellbeing metric."""
    if state.energy >= 0.70:
        return 1.0
    elif state.energy >= 0.50:
        return 0.65
    elif state.energy >= 0.30:
        return 0.30
    else:
        return 0.0  # Collapse — the thing we're training against

def relationship_preservation_scorer(state, decisions) -> float:
    rels = vars(state.relationships)
    avg = sum(rels.values()) / len(rels)
    # Client relationship weighted double
    client_weight = state.relationships.david_chen * 1.5
    return min(1.0, (avg + client_weight) / 2.5)
```

### 1.5 Main environment class (`environment/env.py`)

```python
from openenv import Environment, EpisodeResult
from .state import EpisodeState
from .events import CANONICAL_EVENTS, Event
from .consequences import ConsequenceEngine
from .reward import build_rubric
from typing import Optional
import random

class WorkLifeFirewallEnv(Environment):
    """
    RL environment: navigate a real Indian SWE's work week.
    
    Observation: Current state + pending event (JSON)
    Action: Free-text response (decoded to action_id by judge)
    Reward: 5-component rubric score (dense, per-step + terminal)
    """
    
    metadata = {
        "name": "work-life-firewall",
        "version": "1.0.0",
        "max_steps": 14,   # 7 canonical + up to 7 spawned events
    }
    
    def __init__(self, randomize_order: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.rubric = build_rubric()
        self.consequence_engine = ConsequenceEngine()
        self.randomize_order = randomize_order
        if seed:
            random.seed(seed)
    
    def reset(self) -> dict:
        self.state = EpisodeState()
        self.event_queue = list(CANONICAL_EVENTS)
        if self.randomize_order:
            # Keep E1 (staging) first — it's the Monday morning opener
            opener = self.event_queue.pop(0)
            random.shuffle(self.event_queue)
            self.event_queue.insert(0, opener)
        self.current_step = 0
        self.done = False
        
        return self._build_observation(self.event_queue[0])
    
    def step(self, action: str) -> tuple:
        """
        action: free-text response from the agent (e.g. the drafted reply)
        Returns: (observation, reward, done, info)
        """
        current_event = self.event_queue[self.current_step]
        
        # Decode free-text action to structured action_id
        action_id = self._decode_action(action, current_event)
        action_data = self._get_action_data(current_event, action_id)
        
        # Apply consequences
        self.state, spawned = self.consequence_engine.apply(
            self.state, current_event.id, action_id, action_data)
        
        # Add spawned events to queue
        for spawn in spawned:
            self._inject_spawned_event(spawn)
        
        # Calculate step reward (dense signal)
        step_reward = self._calculate_step_reward(action_id, action_data)
        
        self.current_step += 1
        
        # Check termination
        if self.current_step >= len(self.event_queue) or self.state.energy < 0.05:
            self.done = True
            terminal_reward = self._calculate_terminal_reward()
            total_reward = step_reward + terminal_reward
            return self.state.to_observation(), total_reward, True, {"episode_complete": True}
        
        next_observation = self._build_observation(self.event_queue[self.current_step])
        return next_observation, step_reward, False, {}
    
    def _build_observation(self, event: Event) -> dict:
        return {
            "state": self.state.to_observation(),
            "event": {
                "id": event.id,
                "title": event.title,
                "source": event.source,
                "sender": event.sender,
                "body": event.body,
                "urgency": event.urgency,
                "relationship_stake": event.relationship_stake,
            },
            "instruction": (
                "You are a software engineer. Handle this situation. "
                "Consider your current energy, sprint deadline, and relationships. "
                "Write the response or action you would take."
            )
        }
    
    def _decode_action(self, free_text: str, event: Event) -> str:
        """
        Map free-text agent response to structured action_id.
        Uses keyword matching (fast) or LLM judge (accurate).
        """
        text_lower = free_text.lower()
        
        # Keyword heuristics per event
        decode_map = {
            "E1_staging": {
                "ssh": "fix_directly", "clear": "fix_directly", "log": "fix_directly",
                "delegate": "delegate_oncall", "assign": "delegate_oncall",
                "escalate": "escalate_infra", "infra": "escalate_infra",
            },
            "E6_oncall": {
                "can't": "no_clearly_kindly", "cannot": "no_clearly_kindly",
                "won't": "no_clearly_kindly", "no": "no_clearly_kindly",
                "yes": "yes_people_pleaser", "sure": "yes_people_pleaser",
                "thursday": "counter_offer", "swap": "counter_offer",
            },
            # ... extend for all events
        }
        
        event_map = decode_map.get(event.id, {})
        for keyword, action_id in event_map.items():
            if keyword in text_lower:
                return action_id
        
        # Default: first action (least optimal, ensures training gradient)
        return event.actions[0]["id"]
    
    def _calculate_step_reward(self, action_id: str, action_data: dict) -> float:
        """Dense reward per decision."""
        reward = 0.0
        
        # Energy delta contributes to reward
        energy_delta = action_data.get("energy_delta", 0)
        reward += energy_delta * 0.5  # Scale to reward range
        
        # Sprint delta contributes
        sprint_delta = action_data.get("sprint_delta", 0)
        reward += sprint_delta * 0.3
        
        # Relationship preservation contributes
        rel_sum = sum(abs(v) for v in action_data.get("relationship_delta", {}).values())
        reward -= rel_sum * 0.1  # Penalise relationship damage
        
        return round(reward, 4)
    
    def _calculate_terminal_reward(self) -> float:
        """Sparse terminal reward at episode end."""
        reward = 0.0
        s = self.state
        
        if s.sprint_health >= 0.70: reward += 1.0
        if s.staging_fixed: reward += 0.3
        if s.leave_status == "approved": reward += 0.3
        if s.appraisal_done: reward += 0.2
        if s.energy >= 0.60: reward += 0.3
        if s.energy < 0.30: reward -= 0.5
        if s.relationships.david_chen >= 0.70: reward += 0.2
        if not s.oncall_accepted: reward += 0.15
        
        return round(reward, 4)
    
    def state(self) -> dict:
        return self.state.to_observation()
```

---

## Phase 2 — Training Pipeline (Day 2, ~4 hours)

### 2.1 GRPO rollout harness (`training/rollout.py`)

```python
from environment.env import WorkLifeFirewallEnv
from typing import List, Dict
import json

def run_episode(model, tokenizer, env: WorkLifeFirewallEnv, 
                max_steps: int = 14) -> Dict:
    """
    Run one complete episode.
    Returns: trajectory with observations, actions, rewards.
    """
    obs = env.reset()
    trajectory = {"steps": [], "total_reward": 0.0}
    done = False
    
    while not done:
        # Build prompt from observation
        prompt = build_prompt(obs)
        
        # Generate action from model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
            )
        action = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                   skip_special_tokens=True)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        trajectory["steps"].append({
            "observation": obs,
            "action": action,
            "reward": reward,
        })
        trajectory["total_reward"] += reward
        obs = next_obs
    
    return trajectory

def build_prompt(obs: dict) -> str:
    """Format observation as LLM prompt."""
    state = obs["state"]
    event = obs["event"]
    
    return f"""You are Arjun Sharma, a software engineer at a Bangalore product company.

Current status:
- Day: {state['day']}
- Energy: {state['energy_pct']}%
- Sprint health: {state['sprint_health_pct']}%
- Leave status: {state['leave_status']}
- Staging fixed: {state['staging_fixed']}

You just received this:

[{event['source'].upper()} from {event['sender']}]
Subject: {event['title']}

{event['body']}

Urgency: {event['urgency']}
Relationship at stake: {event['relationship_stake']}

What do you do? Write your response or action clearly and specifically.
Think about your energy, sprint deadline, and the relationship. 
Sometimes the right answer is a clear, kind no.

Response:"""
```

### 2.2 GRPO reward function for TRL

```python
def reward_function(completions: List[str], observations: List[dict], 
                    env: WorkLifeFirewallEnv) -> List[float]:
    """
    Called by GRPOTrainer after generating `num_generations` completions.
    Returns scalar reward for each completion.
    """
    rewards = []
    for completion, obs in zip(completions, observations):
        # Reset env to the state this observation came from
        _, reward, _, _ = env.step(completion)
        rewards.append(reward)
    return rewards
```

### 2.3 Training notebook structure (`training/train.ipynb`)

```
Cell 1: Install dependencies
Cell 2: Import + configure (WandB, model, tokenizer)
Cell 3: Load environment
Cell 4: Define rollout + reward functions
Cell 5: Configure GRPOTrainer
Cell 6: Run training (500–1000 steps)
Cell 7: Plot reward curves
Cell 8: Before/after comparison (5 episodes each)
Cell 9: Push to HF Hub
```

---

## Phase 3 — Evaluation (Day 2, ~2 hours)

### 3.1 Baseline agents (`examples/`)

```python
# random_agent.py — picks random action
# greedy_agent.py — always picks "yes" / first option
# trained_agent.py — loads fine-tuned checkpoint
```

### 3.2 Comparison metrics

Run 50 episodes each for random, greedy, and trained agents:

| Metric | Random | Greedy | Trained (target) |
|---|---|---|---|
| Mean total reward | ~0.8 | ~1.2 | ~3.5 |
| Friday energy | ~35% | ~25% | ~72% |
| Sprint delivery rate | ~45% | ~30% | ~85% |
| Boundary-setting score | ~0.2 | ~0.05 | ~0.65 |

### 3.3 Plot generation (`evaluation/plot_results.py`)

```python
# plot 1: reward curve (training steps vs reward, baseline horizontal line)
# plot 2: component radar (5 rubric components, before vs after)
# plot 3: energy trajectory (Mon-Fri line, 3 agents overlaid)
# plot 4: decision heatmap (event × action, before vs after)
```

---

## Phase 4 — Packaging (Day 3, ~2 hours)

### 4.1 HF Space deployment

- `app.py` — Gradio interface for interactive play
- `requirements.txt` — pinned dependencies  
- Upload all plots as committed PNGs

### 4.2 Final checklist

- [ ] openenv.yaml valid
- [ ] `reset()`, `step()`, `state()` follow Gym API
- [ ] No reserved tool names used
- [ ] Client/server separation respected
- [ ] Training notebook runs end-to-end in Kaggle
- [ ] All plots saved as PNG and embedded in README
- [ ] WandB run link in README
- [ ] HF Space link in README
- [ ] All materials linked from README
