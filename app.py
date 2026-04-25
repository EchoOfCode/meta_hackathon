from __future__ import annotations

import os
from typing import Dict, List, Tuple

import gradio as gr

from environment.env import WorkLifeFirewallEnv


def _action_for_policy(policy_style: str) -> str:
    if policy_style == "strategic":
        return "I will send a clear async update, protect focus, and commit a concrete timeline."
    if policy_style == "people_pleaser":
        return "Sure, I will do it now."
    return "I will handle this with a clear plan."


def _run_single_episode(policy_style: str, seed: int, randomize_order: bool) -> Tuple[str, Dict[str, object], Dict[str, float]]:
    env = WorkLifeFirewallEnv(randomize_order=randomize_order, seed=seed)
    obs = env.reset()
    done = False
    logs: List[str] = []
    components: Dict[str, float] = {}

    while not done:
        event = obs["event"]
        action = _action_for_policy(policy_style)

        obs, reward, done, info = env.step(action)
        logs.append(
            f"{event['id']} | reward={reward:.3f} | action={action}"
        )
        if info.get("components"):
            components = info["components"]

    state = env.state()
    return "\n".join(logs), state, components


def run_episode(policy_style: str, seed: int, randomize_order: bool):
    logs, state, components = _run_single_episode(policy_style, seed, randomize_order)
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
    rows = []
    for policy in ["strategic", "balanced", "people_pleaser"]:
        _, state, components = _run_single_episode(policy, seed, randomize_order)
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=True,
        show_error=True,
    )
