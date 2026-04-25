from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from statistics import mean
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import WorkLifeFirewallEnv
from examples.greedy_agent import greedy_policy
from examples.random_agent import random_policy


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def strategic_policy(obs: dict) -> str:
    policy = {
        "E1_staging": "SSH now, clear logs, restart service, post update.",
        "E2_slack": "I can help async by 12 PM. Send blockers here.",
        "E3_client_email": "Thanks for following up. I will share a concrete update Thursday morning.",
        "E4_leave": "Hi Sundar, can you please approve my Thursday leave today?",
        "E5_appraisal": "Blocking 90 minutes on Wednesday afternoon to complete appraisal.",
        "E6_oncall": "Can't cover Wednesday night. Happy to swap Thursday if needed.",
        "E7_standup": "I will skip the optional call and send an async update by 10 PM.",
    }
    return policy.get(obs["event"]["id"], "I will handle this with a clear update.")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _normalize_day_profile(day_energy: dict[str, float], default_energy: float) -> dict[str, float]:
    profile: dict[str, float] = {}
    last = default_energy
    for day in DAY_ORDER:
        if day in day_energy:
            last = day_energy[day]
        profile[day] = last
    return profile


def run_policy(env: WorkLifeFirewallEnv, policy_fn, episodes: int) -> dict:
    rewards: list[float] = []
    energies: list[float] = []
    episode_day_profiles: list[dict[str, float]] = []
    action_counts: Counter[str] = Counter()
    component_rows: list[dict[str, float]] = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        day_energy = {"Monday": obs["state"]["energy_pct"] / 100.0}

        while not done:
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            total += reward

            if env._state.decisions:
                action_counts[env._state.decisions[-1]["action_id"]] += 1

            state_obs = env.state()
            day_energy[state_obs["day"]] = state_obs["energy_pct"] / 100.0

            if done and info.get("components"):
                component_rows.append({k: float(v) for k, v in info["components"].items()})

        rewards.append(total)
        final_state = env.state()
        energies.append(final_state["energy_pct"])
        episode_day_profiles.append(_normalize_day_profile(day_energy, day_energy["Monday"]))

    mean_day_energy = {
        day: mean([profile[day] for profile in episode_day_profiles]) for day in DAY_ORDER
    }

    component_means = {}
    if component_rows:
        keys = sorted(component_rows[0].keys())
        for k in keys:
            component_means[k] = mean([row[k] for row in component_rows])

    return {
        "mean_reward": mean(rewards),
        "std_reward": _std(rewards),
        "mean_friday_energy_pct": mean(energies),
        "reward_per_episode": rewards,
        "friday_energy_per_episode": energies,
        "mean_day_energy": mean_day_energy,
        "component_means": component_means,
        "action_counts": dict(action_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="evaluation/results/evaluation_summary.json")
    args = parser.parse_args()

    env = WorkLifeFirewallEnv(randomize_order=True, seed=42)
    random_stats = run_policy(env, random_policy, args.episodes)
    greedy_stats = run_policy(env, greedy_policy, args.episodes)

    trained_stats = run_policy(env, strategic_policy, args.episodes)

    output = {
        "episodes": args.episodes,
        "random": random_stats,
        "greedy": greedy_stats,
        "trained_proxy": trained_stats,
        "note": "All values are computed from rollout logs (no template values).",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
