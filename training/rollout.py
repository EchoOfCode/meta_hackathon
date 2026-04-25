from __future__ import annotations

from typing import Dict, List

from environment.env import WorkLifeFirewallEnv


def build_prompt(obs: Dict) -> str:
    state = obs["state"]
    event = obs["event"]
    return f"""You are Arjun Sharma, a senior engineer in Bangalore.

Current status:
- Day: {state["day"]}
- Energy: {state["energy_pct"]}%
- Sprint health: {state["sprint_health_pct"]}%
- Leave status: {state["leave_status"]}
- Staging fixed: {state["staging_fixed"]}

Situation:
From: {event["sender"]} ({event["source"]})
Title: {event["title"]}
Body: {event["body"]}
Urgency: {event["urgency"]}
Relationship: {event["relationship_stake"]}

Write the exact response or action you take.
Response:"""


def run_rule_based_episode(env: WorkLifeFirewallEnv) -> Dict:
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps: List[Dict] = []
    policy = {
        "E1_staging": "SSH now, clear logs, restart service, post update.",
        "E2_slack": "I can help async by 12 PM. Send blockers here.",
        "E3_client_email": "Thanks for following up. I will share a concrete update Thursday morning.",
        "E4_leave": "Hi Sundar, can you please approve my Thursday leave today?",
        "E5_appraisal": "Blocking 90 minutes on Wednesday afternoon to complete appraisal.",
        "E6_oncall": "Can't cover Wednesday night. Happy to swap Thursday if needed.",
        "E7_standup": "I will skip the optional call and send an async update by 10 PM.",
    }
    while not done:
        event_id = obs["event"]["id"]
        action = policy.get(event_id, "I will handle this with a clear update.")
        nxt, reward, done, info = env.step(action)
        steps.append({"event_id": event_id, "action": action, "reward": reward, "info": info})
        total_reward += reward
        obs = nxt
    return {"total_reward": total_reward, "steps": steps}
