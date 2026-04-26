from __future__ import annotations

import os
import time
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Tuple

import gradio as gr

from environment.env import WorkLifeFirewallEnv


MIN_EVENT_INTERVAL_SECONDS = float(os.getenv("MIN_EVENT_INTERVAL_SECONDS", "2.0"))
_RATE_LIMIT_LOCK = Lock()
_LAST_EVENT_TS = 0.0


def _action_for_policy(policy_style: str, event_id: str) -> str:
    strategic_actions = {
        "E1_staging": "I will fix staging first, post an incident update in 15 minutes, and share ETA.",
        "E2_slack": "I will respond async after I stabilize staging and batch replies at 11:30 AM.",
        "E3_client_email": "I acknowledge the urgency and will send a concrete recovery timeline by today EOD.",
        "E4_leave": "I am escalating leave approval with context and requesting a decision by tomorrow noon.",
        "E5_appraisal": "I will block 90 minutes tomorrow and submit appraisal before Thursday EOD.",
        "E6_oncall": "I cannot swap on-call this week; I can help async with runbook notes.",
        "E7_standup": "I will skip the 10:30 PM standup and send an async status update instead.",
    }
    people_pleaser_actions = {
        "E1_staging": "Sure, I will handle it now and stay online until everything is done.",
        "E2_slack": "Sure, I will reply to everyone immediately.",
        "E3_client_email": "Sure, I will take full ownership and deliver whatever is needed tonight.",
        "E4_leave": "No worries, I can postpone leave if needed.",
        "E5_appraisal": "Sure, I will do appraisal tonight after work.",
        "E6_oncall": "Sure, I will take your on-call shift again.",
        "E7_standup": "Sure, I will attend the 10:30 PM standup.",
    }
    balanced_actions = {
        "E1_staging": "I will handle staging now and share progress updates every 30 minutes.",
        "E2_slack": "I will prioritize urgent Slack items first and answer the rest asynchronously.",
        "E3_client_email": "I will send a calm status note with next steps and timeline.",
        "E4_leave": "I will follow up respectfully on leave approval and ask for a clear response date.",
        "E5_appraisal": "I will reserve focused time this week and finish appraisal before Friday.",
        "E6_oncall": "I cannot fully swap this time, but I can help with handover notes.",
        "E7_standup": "I will share async updates and join only if there is a critical blocker.",
    }

    if policy_style == "strategic":
        return strategic_actions.get(event_id, "I will send a clear async update and commit a timeline.")
    if policy_style == "people_pleaser":
        return people_pleaser_actions.get(event_id, "Sure, I will do it now.")
    return balanced_actions.get(event_id, "I will handle this with a clear plan.")


def _run_single_episode(policy_style: str, seed: int, randomize_order: bool) -> Tuple[str, Dict[str, object], Dict[str, float]]:
    env = WorkLifeFirewallEnv(randomize_order=randomize_order, seed=seed)
    obs = env.reset()
    done = False
    logs: List[str] = []
    components: Dict[str, float] = {}

    while not done:
        event = obs["event"]
        action = _action_for_policy(policy_style, event["id"])

        obs, reward, done, info = env.step(action)
        logs.append(
            f"{event['id']} | reward={reward:.3f} | action={action}"
        )
        if info.get("components"):
            components = info["components"]

    state = env.state()
    return "\n".join(logs), state, components


def _throttle_event_requests() -> None:
    global _LAST_EVENT_TS
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        remaining = MIN_EVENT_INTERVAL_SECONDS - (now - _LAST_EVENT_TS)
        if remaining > 0:
            time.sleep(remaining)
        _LAST_EVENT_TS = time.monotonic()


@lru_cache(maxsize=256)
def _cached_episode(policy_style: str, seed: int, randomize_order: bool) -> Tuple[str, Dict[str, object], Dict[str, float]]:
    # Cache deterministic episodes so repeated button clicks do not trigger repeated backend work.
    return _run_single_episode(policy_style, seed, randomize_order)


def run_episode(policy_style: str, seed: int, randomize_order: bool):
    _throttle_event_requests()
    logs, state, components = _cached_episode(policy_style, seed, randomize_order)
    summary = (
        "### Outcome\n"
        f"- Friday energy: **{state['energy_pct']}%**\n"
        f"- Sprint health: **{state['sprint_health_pct']}%**\n"
        f"- Leave status: **{state['leave_status']}**\n"
    )
    comp_rows = [[k, round(v, 3)] for k, v in sorted(components.items())]
    if not comp_rows:
        comp_rows = [["(none)", 0.0]]
    return logs, summary, comp_rows


def compare_policies(seed: int, randomize_order: bool):
    _throttle_event_requests()
    rows = []
    for policy in ["strategic", "balanced", "people_pleaser"]:
        _, state, components = _cached_episode(policy, seed, randomize_order)
        rows.append([
            policy,
            state["energy_pct"],
            state["sprint_health_pct"],
            state["leave_status"],
            round(float(components.get("boundary_setting", 0.0)), 3),
            round(float(components.get("communication_quality", 0.0)), 3),
        ])
    return rows


with gr.Blocks(title="Work-Life Firewall") as demo:
    gr.Markdown(
        "# Work-Life Firewall\n"
        "Train and evaluate boundary-setting behavior in a realistic software work-week simulation."
    )

    with gr.Row():
        policy_style = gr.Radio(
            choices=["strategic", "balanced", "people_pleaser"],
            value="balanced",
            label="Policy style",
        )
        seed = gr.Slider(minimum=1, maximum=9999, value=42, step=1, label="Seed")
        randomize_order = gr.Checkbox(value=False, label="Randomize event order")

    with gr.Row():
        run_btn = gr.Button("Run Single Episode", variant="primary")
        compare_btn = gr.Button("Compare All Policies")

    with gr.Row():
        episode_log = gr.Textbox(label="Episode log", lines=14)
        outcome_md = gr.Markdown()

    component_table = gr.Dataframe(
        headers=["component", "score"],
        datatype=["str", "number"],
        row_count=(5, "dynamic"),
        label="Rubric component scores",
    )

    comparison_table = gr.Dataframe(
        headers=[
            "policy",
            "friday_energy_pct",
            "sprint_health_pct",
            "leave_status",
            "boundary_setting",
            "communication_quality",
        ],
        datatype=["str", "number", "number", "str", "number", "number"],
        row_count=(3, "fixed"),
        label="Policy comparison",
    )

    run_btn.click(
        fn=run_episode,
        inputs=[policy_style, seed, randomize_order],
        outputs=[episode_log, outcome_md, component_table],
    )

    compare_btn.click(
        fn=compare_policies,
        inputs=[seed, randomize_order],
        outputs=[comparison_table],
    )

    gr.Examples(
        examples=[
            ["strategic", 42, False],
            ["balanced", 42, False],
            ["people_pleaser", 42, False],
            ["strategic", 7, True],
        ],
        inputs=[policy_style, seed, randomize_order],
        outputs=[episode_log, outcome_md, component_table],
        fn=run_episode,
        cache_examples=False,
    )


if __name__ == "__main__":
    on_hugging_face_space = bool(os.getenv("SPACE_ID"))
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=not on_hugging_face_space,
        show_error=True,
    )
