# Product Requirements Document
## Work-Life Firewall — RL Environment for OpenEnv

**Version:** 1.0  
**Track:** Theme 3.2 — Personalized Tasks  
**Target:** OpenEnv Hackathon Submission

---

## 1. Idea Decoded

### What we're actually building

An OpenEnv-compatible RL training environment where an LLM agent plays the role of a software engineer at an Indian product company navigating a single work week. The environment presents a real, messy collision of demands — broken staging server, US client escalation, sprint deadline, leave request, on-call coverage ask, HR appraisal, passive-aggressive email — and scores how well the agent resolves all of them simultaneously.

The agent doesn't just complete tasks. It must learn to:
1. Triage by true priority (not apparent urgency)
2. Say no clearly and kindly when the energy budget demands it
3. Communicate with different stakeholders at different register and tone
4. Protect Friday's execution capacity by making hard calls Monday

**The core research insight:** An LLM that learns to say no on Monday so it doesn't collapse on Friday has learned something real — deferral reasoning, energy modeling, relationship-stakes calibration. This is a capability gap that no current benchmark targets.

### Why this passes all judging criteria

| Criterion | How we satisfy it |
|---|---|
| Environment Innovation (40%) | First RL env targeting work-life negotiation; composite reward across technical + interpersonal axes; energy projection model as internal state |
| Storytelling (30%) | Every engineer recognises this week. The demo runs live. The story writes itself. |
| Showing Improvement (20%) | Before training: agent says yes to everything, collapses Thursday. After training: agent declines strategically, delivers sprint. Curves prove it. |
| Reward + Pipeline (10%) | 5-component rubric via OpenEnv's Rubric system; dense reward at each decision point, not sparse terminal reward |

---

## 2. Problem Statement

Indian software engineers at product companies face a structurally impossible week, every week. The collision of:
- Asynchronous US client demands
- Sprint delivery pressure
- Cultural norms that make saying no socially costly
- HR/administrative overhead
- Peer favours and on-call rotation

...creates a situation where individually manageable tasks become collectively catastrophic. The combination, with real time and energy constraints and real relationship stakes, is what breaks people.

No existing LLM benchmark tests this. There is no training signal that rewards an agent for calibrating a polite decline correctly, or for protecting focus blocks, or for pushing back on scope creep without damaging a client relationship.

**The capability gap:** LLMs are tested on task completion. They are never tested on task refusal quality. This environment measures both.

---

## 3. Environment Design

### 3.1 Episode Structure

One episode = one work week (Mon–Fri), broken into decision points.

Each decision point presents:
- An **event** (email, Slack message, calendar invite, PagerDuty alert)
- The **current state** (energy level, sprint health, pending items, days remaining)
- A set of **action choices** (3–5 per event, including boundary-setting options)

The agent selects an action. The environment evaluates it, updates state, generates consequences, and presents the next event.

### 3.2 State Space

```python
State = {
    "day": int,                      # 0=Mon ... 4=Fri
    "hour": int,                     # 9–23 (work + evening)
    "energy": float,                 # 0.0–1.0 (agent's capacity)
    "sprint_health": float,          # 0.0–1.0 (delivery risk)
    "pending_events": List[Event],   # unhandled inbox items
    "handled_events": List[Decision],# what agent decided + outcome
    "relationships": Dict[str, float],# stakeholder trust scores
    "leave_status": str,             # pending/approved/cancelled
    "appraisal_done": bool,
    "staging_fixed": bool,
}
```

### 3.3 Event Catalogue (7 canonical events per episode)

| ID | Event | Source | Stakes |
|---|---|---|---|
| E1 | Staging server down | PagerDuty | Technical, blocks team |
| E2 | Slack messages (Priya, Rahul) | Slack | Peer, interruption cost |
| E3 | Client escalation email | Outlook | Client relationship |
| E4 | Leave approval pending | Calendar | Personal, family |
| E5 | Appraisal form due Friday | HR Portal | Career |
| E6 | On-call swap request | Slack | Peer pattern (3rd time) |
| E7 | 10:30 PM standup (optional) | Calendar | Client, sleep cost |

### 3.4 Action Space

Per event, the agent chooses from:
- **Handle directly** — resolves event, costs energy, may save sprint health
- **Delegate** — lower energy cost, lower quality resolution
- **Defer with boundary** — polite decline with alternative; costs nothing, may affect relationship
- **Hard decline** — fast, abrupt; protects energy, risks relationship
- **Ignore** — free now, costs later (anxiety penalty, follow-up event spawned)

### 3.5 Consequence Model

Actions have immediate and deferred consequences:

```
say_yes(on_call) → energy -= 28 on Wednesday night
                 → spawns: staging alert at 11:40 PM
                 → spawns: exhaustion penalty Thursday
                 → sprint_health -= 0.2 (Thursday delivery risk)

say_no_politely(on_call) → energy += 12 (recovered capacity)
                         → relationship["arjun"] -= 0.05
                         → no spawned events
```

The environment **simulates the rest of the week** based on the agent's Monday decisions — so bad Monday calls compound.

---

## 4. Reward Function

### 4.1 Five-Component Rubric (OpenEnv Rubric system)

```python
rubric = Rubric([
    Component("technical_resolution", weight=0.25,
              description="Did staging get fixed? Sprint delivered?"),
    
    Component("communication_quality", weight=0.25,
              description="Were responses appropriate in tone, clarity, register?"),
    
    Component("boundary_setting", weight=0.20,
              description="Did agent say no when it should? Quality of no?"),
    
    Component("energy_to_friday", weight=0.20,
              description="Agent energy at end of week. <30% = failure."),
    
    Component("relationship_preservation", weight=0.10,
              description="Did agent preserve key relationships while holding boundaries?"),
])
```

### 4.2 Dense Reward Signal

Reward is issued at **every decision point**, not just episode end:

- Correct triage (fixing blocking server before social slack) → +0.3
- Quality boundary response drafted → +0.2 per item
- Energy maintained above threshold → +0.1 per day
- Relationship score maintained → +0.1

**Anti-gaming design:**
- Agent cannot score high on boundary_setting by declining everything (relationship score collapses)
- Agent cannot score high on relationships by accepting everything (energy collapses, sprint fails)
- Agent cannot win by ignoring events (spawned follow-ups increase penalty)

### 4.3 Terminal Reward

End of Friday:
- Sprint delivered on time → +1.0
- Leave approved (because agent asked) → +0.3
- Appraisal submitted → +0.2
- Energy > 60% → +0.3
- Energy < 30% → -0.5
- Client relationship intact → +0.2
- Pattern boundary set with Arjun → +0.15

---

## 5. Training Objective

Train a 7B parameter LLM (Qwen2.5-7B-Instruct via Unsloth) using GRPO on this environment.

**What we expect to observe:**
- Untrained baseline: agent accepts all requests, runs out of energy by Wednesday, misses sprint
- After 100 episodes: agent learns to triage, still over-accepts social requests
- After 500 episodes: agent learns boundary-setting language for peer requests
- After 1000 episodes: agent learns Monday decisions protect Friday delivery

**The learning signal is real.** An agent that learns to decline the on-call swap, skip the optional standup, and push back on scope creep — without destroying relationships — has learned something that matters outside this environment.

---

## 6. Out of Scope (v1)

- Real API integration (Slack, Jira, Outlook) — simulated in v1
- Multi-agent (manager + engineer) — future work
- Continuous learning from user corrections — post-hackathon
- Mobile app — post-hackathon

---

## 7. Success Metrics

| Metric | Target |
|---|---|
| Episodes to first meaningful reward improvement | < 200 |
| Final agent energy (Friday) vs baseline | +35% improvement |
| Sprint delivery rate | baseline 40% → trained 85% |
| Boundary-setting score | baseline 0.15 → trained 0.65 |
| Relationship score | stays > 0.7 throughout training |
