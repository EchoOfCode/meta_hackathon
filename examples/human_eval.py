from __future__ import annotations

from environment.env import WorkLifeFirewallEnv


def main() -> None:
    env = WorkLifeFirewallEnv(randomize_order=False, seed=42)
    obs = env.reset()
    done = False
    print("Work-Life Firewall interactive run. Type your action/response each step.\n")
    while not done:
        event = obs["event"]
        state = obs["state"]
        print(f"[{event['id']}] {event['title']}")
        print(f"From: {event['sender']} | Urgency: {event['urgency']}")
        print(f"Energy {state['energy_pct']}% | Sprint {state['sprint_health_pct']}%")
        print(event["body"])
        action = input("\nResponse: ").strip()
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward:.3f}\n")
        if done:
            print("Episode complete.")
            if info.get("components"):
                print("Component scores:")
                for k, v in info["components"].items():
                    print(f"  - {k}: {v:.3f}")


if __name__ == "__main__":
    main()
