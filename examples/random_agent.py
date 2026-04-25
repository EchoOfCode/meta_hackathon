from __future__ import annotations

import argparse
import random

from environment.env import WorkLifeFirewallEnv


def random_policy(obs: dict) -> str:
    options = obs["event"]["actions"]
    return random.choice(options)


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
            obs, reward, done, _ = env.step(random_policy(obs))
            total += reward
        rewards.append(total)
    print(f"Random agent average reward over {args.episodes} episodes: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    main()
