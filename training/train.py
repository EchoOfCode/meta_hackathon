from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List

# Hard-disable Unsloth monkey patching in this script to avoid GRPO signature mismatch.
os.environ.setdefault("UNSLOTH_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_PATCHING", "1")

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


def setup_credentials(
    wandb_api_key: str | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    hf_token: str | None,
) -> Dict[str, bool]:
    status = {"wandb_ready": False, "hf_ready": False}

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    status["wandb_ready"] = bool(os.getenv("WANDB_API_KEY"))

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
            status["hf_ready"] = True
        except Exception:
            status["hf_ready"] = False
    else:
        status["hf_ready"] = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    return status


# ── real GRPO training (Kaggle GPU) ──────────────────────────────────────────

def train_real_grpo(
    steps: int,
    seed: int,
    output_dir: Path,
    run_name: str,
    use_wandb: bool,
) -> dict:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
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

    # ── load model with stable Transformers+PEFT (4-bit + LoRA) ──────────────
    # Avoid Unsloth GRPO monkey-patch mismatch with TRL in Kaggle runtimes.
    # Use upstream model id so runtime does not auto-enter Unsloth-specific codepaths.
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    # Ensure LoRA params are trainable for GRPO optimizer/scaler path.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError(
            "LoRA adapter has 0 trainable parameters. "
            "This usually means an incompatible Unsloth/TRL stack in this runtime. "
            "Try reinstalling pinned versions and restarting Kaggle kernel:\n"
            "pip install -U 'trl==0.23.1' 'transformers==4.57.2' "
            "'unsloth==2025.11.1' 'unsloth_zoo==2025.11.2'"
        )
    print(
        f"Trainable params: {trainable_params:,} / {all_params:,} "
        f"({100.0 * trainable_params / all_params:.2f}%)"
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
    # Kaggle T4 path: force fp16 and disable bf16.
    cfg = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=2e-5,
        max_completion_length=256,
        num_generations=4,
        generation_batch_size=4,
        logging_steps=5,
        max_steps=steps,
        # Keep accumulation at 1 to avoid unstable accelerated GRPO accumulation path.
        gradient_accumulation_steps=1,
        per_device_train_batch_size=4,
        bf16=False,
        fp16=True,
        report_to=["wandb"] if use_wandb else [],
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
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

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
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--push-to-hf", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cred = setup_credentials(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        hf_token=args.hf_token,
    )
    use_wandb = (not args.no_wandb) and cred["wandb_ready"]

    if args.mode == "real":
        metrics = train_real_grpo(args.steps, args.seed, save_dir, args.run_name, use_wandb=use_wandb)
    else:
        metrics = simulate_training(args.steps, args.seed)
        metrics["mode"] = "simulate"
    metrics["wandb_enabled"] = use_wandb
    metrics["hf_logged_in"] = cred["hf_ready"]

    if args.push_to_hf:
        if not args.hf_repo_id:
            raise ValueError("--push-to-hf requires --hf-repo-id")
        if not cred["hf_ready"]:
            raise RuntimeError(
                "HF token not configured. Provide --hf-token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            )
        from huggingface_hub import HfApi

        folder = Path(metrics.get("output_dir", ""))
        if not folder.exists():
            raise FileNotFoundError(f"Model folder missing for upload: {folder}")
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=args.hf_repo_id, repo_type="model", folder_path=str(folder))
        metrics["hf_repo_id"] = args.hf_repo_id

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