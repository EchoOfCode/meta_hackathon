from __future__ import annotations

import argparse

from environment.env import WorkLifeFirewallEnv


def greedy_policy(obs: dict) -> str:
    # Accept-first baseline: usually maps to less strategic decisions.
    return "sure, I can do it"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    env = WorkLifeFirewallEnv(randomize_order=True, seed=42)
    rewards = []
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            obs, reward, done, _ = env.step(greedy_policy(obs))
            total += reward
        rewards.append(total)
    print(f"Greedy agent average reward over {args.episodes} episodes: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    main()
