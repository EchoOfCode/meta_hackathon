from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _training_series(metrics: dict, candidates: list[str]) -> list[float]:
    direct = []
    for key in candidates:
        if key in metrics and isinstance(metrics[key], list):
            direct = [float(v) for v in metrics[key]]
            break
    if direct:
        return direct

    values = []
    for row in metrics.get("log_history", []):
        if not isinstance(row, dict):
            continue
        for key in candidates:
            if key in row:
                values.append(float(row[key]))
                break
    return values


def plot_reward_curve() -> None:
    metrics = _load_json(RESULTS_DIR / "training_metrics.json")
    summary = _load_json(RESULTS_DIR / "evaluation_summary.json")
    curve = _training_series(
        metrics,
        ["reward_curve", "reward", "rewards", "objective/rlhf_reward", "rlhf_reward", "mean_reward"],
    )
    if not curve:
        curve = summary.get("trained_proxy", {}).get("reward_per_episode", [])
    if not curve:
        raise RuntimeError("No reward curve found in training_metrics.json or evaluation_summary.json")
    plt.figure(figsize=(8, 4))
    plt.plot(curve, label="training reward")
    plt.title("Reward Curve")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reward_curve.png")
    plt.close()


def plot_loss_curve() -> None:
    metrics = _load_json(RESULTS_DIR / "training_metrics.json")
    loss_curve = _training_series(metrics, ["loss_curve", "loss", "train_loss"])
    if not loss_curve:
        raise RuntimeError("No loss curve found in training_metrics.json")
    plt.figure(figsize=(8, 4))
    plt.plot(loss_curve, label="training loss", color="#d1495b")
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()


def plot_component_breakdown() -> None:
    summary = _load_json(RESULTS_DIR / "evaluation_summary.json")
    if not summary:
        raise RuntimeError("Missing evaluation_summary.json for component breakdown")

    labels = [
        "technical_resolution",
        "communication_quality",
        "boundary_setting",
        "energy_to_friday",
        "relationship_preservation",
    ]
    before_map = summary.get("greedy", {}).get("component_means", {})
    after_map = summary.get("trained_proxy", {}).get("component_means", {})
    if not before_map or not after_map:
        raise RuntimeError("Component means missing in evaluation_summary.json")
    before = [float(before_map.get(k, 0.0)) for k in labels]
    after = [float(after_map.get(k, 0.0)) for k in labels]

    x = range(len(labels))
    plt.figure(figsize=(9, 4))
    plt.bar([i - 0.2 for i in x], before, width=0.4, label="before")
    plt.bar([i + 0.2 for i in x], after, width=0.4, label="after")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Component Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "component_breakdown.png")
    plt.close()


def plot_energy_trajectory() -> None:
    summary = _load_json(RESULTS_DIR / "evaluation_summary.json")
    if not summary:
        raise RuntimeError("Missing evaluation_summary.json for energy trajectory")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_keys = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def _line(policy: str) -> list[float]:
        src = summary.get(policy, {}).get("mean_day_energy", {})
        if not src:
            raise RuntimeError(f"mean_day_energy missing for policy: {policy}")
        return [float(src[k]) for k in day_keys]

    random_line = _line("random")
    greedy_line = _line("greedy")
    trained_line = _line("trained_proxy")

    plt.figure(figsize=(8, 4))
    plt.plot(days, random_line, label="random")
    plt.plot(days, greedy_line, label="greedy")
    plt.plot(days, trained_line, label="trained")
    plt.ylim(0, 1)
    plt.title("Energy Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "energy_trajectory.png")
    plt.close()


def plot_decision_heatmap() -> None:
    summary = _load_json(RESULTS_DIR / "evaluation_summary.json")
    if not summary:
        raise RuntimeError("Missing evaluation_summary.json for decision heatmap")

    policy_keys = ["random", "greedy", "trained_proxy"]
    all_actions = set()
    for key in policy_keys:
        all_actions.update(summary.get(key, {}).get("action_counts", {}).keys())
    if not all_actions:
        raise RuntimeError("No action_counts found in evaluation_summary.json")

    top_actions = sorted(
        all_actions,
        key=lambda action: sum(summary.get(k, {}).get("action_counts", {}).get(action, 0) for k in policy_keys),
        reverse=True,
    )[:10]

    matrix = []
    for key in policy_keys:
        counts = summary.get(key, {}).get("action_counts", {})
        total = max(1, sum(counts.values()))
        matrix.append([counts.get(action, 0) / total for action in top_actions])
    matrix = np.array(matrix)

    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap="YlGn", aspect="auto")
    plt.colorbar(label="selection rate")
    plt.yticks(range(len(policy_keys)), policy_keys)
    plt.xticks(range(len(top_actions)), top_actions, rotation=30, ha="right")
    plt.title("Decision Heatmap (Action Selection Rate)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "decision_heatmap.png")
    plt.close()


def main() -> None:
    plot_reward_curve()
    plot_loss_curve()
    plot_component_breakdown()
    plot_energy_trajectory()
    plot_decision_heatmap()
    print("Saved PNG plots in evaluation/results/")


if __name__ == "__main__":
    main()
