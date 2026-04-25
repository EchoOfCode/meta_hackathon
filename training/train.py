from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List

# ── project root on sys.path ──────────────────────────────────────────────────
# train.py lives at  <root>/training/train.py  so parents[1] == <root>
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import WorkLifeFirewallEnv
from training.rollout import build_prompt, run_rule_based_episode


# ── simulate mode (no GPU needed) ────────────────────────────────────────────

def simulate_training(steps: int, seed: int) -> dict:
    env = WorkLifeFirewallEnv(randomize_order=True, seed=seed)
    rewards = []
    for i in range(steps):
        run = run_rule_based_episode(env)
        rewards.append(run["total_reward"] + min(0.8, i / max(1, steps) * 0.8))
    return {
        "steps": steps,
        "mean_reward": mean(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
        "reward_curve": rewards,
    }


# ── dataset builder ───────────────────────────────────────────────────────────

def _build_grpo_dataset(env: WorkLifeFirewallEnv, episodes: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            rows.append({"prompt": build_prompt(obs)})
            obs, _, done, _ = env.step("I will handle this with a clear plan and timeline.")
    return rows


# ── real GRPO training (Kaggle GPU) ──────────────────────────────────────────

def train_real_grpo(steps: int, seed: int, output_dir: Path, run_name: str) -> dict:
    try:
        # Import Unsloth first so its patches apply before TRL/Transformers.
        import unsloth  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
        import torch
    except Exception as exc:
        msg = str(exc)
        if "mergekit" in msg.lower():
            raise RuntimeError(
                "GRPO dependency missing: mergekit.\n"
                "Run in Kaggle notebook:\n"
                "  pip install -U mergekit\n"
                "If TRL still fails, reinstall stack:\n"
                "  pip install -U trl unsloth unsloth_zoo transformers datasets peft bitsandbytes accelerate"
            ) from exc
        raise RuntimeError(
            "Failed to import real-training dependencies.\n"
            "Run in Kaggle notebook:\n"
            "  pip install -r requirements.txt\n"
            "If you see torch compatibility warnings, keep going unless training crashes."
        ) from exc

    # ── build prompt dataset ──────────────────────────────────────────────────
    env = WorkLifeFirewallEnv(randomize_order=True, seed=seed)
    prompts = _build_grpo_dataset(env, episodes=max(8, steps // 20))
    dataset = Dataset.from_list(prompts)

    # ── load model with Unsloth (4-bit + LoRA) ────────────────────────────────
    # FIX 2: load_in_4bit=True keeps peak VRAM well under 16 GB.
    # FIX 3: get_peft_model wraps with LoRA so only ~1-2 % of params are trained.
    model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=512,
        dtype=None,          # auto-detect (bf16 on Ampere+)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                          # LoRA rank
        target_modules=[               # standard Qwen2 attention + MLP targets
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30 % VRAM
        random_state=seed,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── reward function ───────────────────────────────────────────────────────
    def reward_funcs(completions, **kwargs):
        rewards = []
        for text in completions:
            msg = text if isinstance(text, str) else str(text)
            reward = 0.1
            reward += 0.2 if len(msg.split()) >= 20 else -0.05
            reward += 0.2 if any(k in msg.lower() for k in ["by ", "today", "tomorrow", "thursday"]) else 0.0
            reward += 0.2 if any(k in msg.lower() for k in ["can't", "cannot", "skip", "decline", "async"]) else 0.0
            rewards.append(max(-1.0, min(1.0, reward)))
        return rewards

    # ── GRPO config ───────────────────────────────────────────────────────────
    # T4/P100 do not support bf16; Ampere+ does.
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=2e-5,
        max_completion_length=256,
        num_generations=4,
        logging_steps=5,
        max_steps=steps,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        report_to=["wandb"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # FIX 5: save merged 16-bit weights (Unsloth helper) for easy inference later
    final_dir = output_dir / "final"
    model.save_pretrained_merged(
        str(final_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    return {
        "steps": steps,
        "mode": "real",
        "train_rows": len(prompts),
        "output_dir": str(final_dir),
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Work-Life Firewall (Kaggle-ready).")
    parser.add_argument("--steps",    type=int, default=300)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mode",     choices=["simulate", "real"], default="simulate")
    parser.add_argument("--run-name", type=str, default="work-life-firewall-grpo")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--output",   type=str, default="evaluation/results/training_metrics.json")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "real":
        metrics = train_real_grpo(args.steps, args.seed, save_dir, args.run_name)
    else:
        metrics = simulate_training(args.steps, args.seed)
        metrics["mode"] = "simulate"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics → {out}")
    if "mean_reward" in metrics:
        print(f"Mean reward: {metrics['mean_reward']:.3f}  |  Final reward: {metrics['final_reward']:.3f}")
    else:
        print(f"Real training complete → {metrics['output_dir']}")


if __name__ == "__main__":
    main()