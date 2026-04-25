from __future__ import annotations

import os

import gradio as gr

from environment.env import WorkLifeFirewallEnv


def run_episode(policy_style: str):
    env = WorkLifeFirewallEnv(randomize_order=False, seed=42)
    obs = env.reset()
    done = False
    logs = []

    while not done:
        event = obs["event"]
        if policy_style == "strategic":
            action = (
                "I will send a clear async update, protect focus, and commit a concrete timeline."
            )
        elif policy_style == "people_pleaser":
            action = "Sure, I will do it now."
        else:
            action = "I will handle this with a clear plan."

        obs, reward, done, info = env.step(action)
        logs.append(f"{event['id']} -> reward {reward:.3f} | action: {action}")

    state = env.state()
    summary = (
        f"Friday energy: {state['energy_pct']}%\n"
        f"Sprint health: {state['sprint_health_pct']}%\n"
        f"Leave: {state['leave_status']}\n"
    )
    if info.get("components"):
        summary += "\nComponent scores:\n"
        summary += "\n".join([f"- {k}: {v:.3f}" for k, v in info["components"].items()])
    return "\n".join(logs), summary


demo = gr.Interface(
    fn=run_episode,
    inputs=gr.Radio(
        choices=["strategic", "balanced", "people_pleaser"],
        value="balanced",
        label="Policy style",
    ),
    outputs=[
        gr.Textbox(label="Episode log", lines=18),
        gr.Textbox(label="Outcome"),
    ],
    title="Work-Life Firewall",
    description="HF Space demo for boundary-setting environment behavior.",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=True,
        show_error=True,
    )
