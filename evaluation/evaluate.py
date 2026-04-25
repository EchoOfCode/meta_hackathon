from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import WorkLifeFirewallEnv
from examples.greedy_agent import greedy_policy
from examples.random_agent import random_policy
from training.rollout import run_rule_based_episode


def run_policy(env: WorkLifeFirewallEnv, policy_fn, episodes: int) -> dict:
    rewards = []
    energies = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
        energies.append(env.state()["energy_pct"])
    return {"mean_reward": mean(rewards), "mean_friday_energy_pct": mean(energies)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="evaluation/results/evaluation_summary.json")
    args = parser.parse_args()

    env = WorkLifeFirewallEnv(randomize_order=True, seed=42)
    random_stats = run_policy(env, random_policy, args.episodes)
    greedy_stats = run_policy(env, greedy_policy, args.episodes)

    trained_rewards = [run_rule_based_episode(env)["total_reward"] for _ in range(args.episodes)]
    trained_energies = [env.state()["energy_pct"] for _ in range(args.episodes)]
    trained_stats = {
        "mean_reward": mean(trained_rewards),
        "mean_friday_energy_pct": mean(trained_energies),
    }

    output = {
        "episodes": args.episodes,
        "random": random_stats,
        "greedy": greedy_stats,
        "trained_proxy": trained_stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
