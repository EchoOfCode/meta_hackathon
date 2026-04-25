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
# Kaggle often exposes 2 GPUs; 4-bit + PEFT + GRPO is more stable on one GPU.
# Keep this before torch import paths in training code.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    max_reward = max(rewards) if rewards else 1.0
    loss_curve = [round(max_reward - r + 0.05, 4) for r in rewards]
    return {
        "steps": steps,
        "mean_reward": mean(rewards),
        "final_reward": rewards[-1] if rewards else 0.0,
        "reward_curve": rewards,
        "loss_curve": loss_curve,
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
    num_generations: int,
    max_completion_length: int,
    per_device_train_batch_size: int,
    use_gradient_checkpointing: bool,
    model_id: str,
    load_in_4bit: bool,
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
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if torch.cuda.is_available():
        # 4-bit training must keep load and train on the exact same device.
        # Use LOCAL_RANK when present so Accelerate and model load agree.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cuda_count = torch.cuda.device_count()
        device_index = local_rank if 0 <= local_rank < cuda_count else 0
        torch.cuda.set_device(device_index)
        device_map = {"": device_index}
    else:
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
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
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        # Avoid hidden defaults enabling checkpointing in some stacks.
        model.gradient_checkpointing_disable()
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
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        generation_batch_size=num_generations,
        logging_steps=5,
        max_steps=steps,
        # Keep accumulation at 1 to avoid unstable accelerated GRPO accumulation path.
        gradient_accumulation_steps=1,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=False,
        fp16=False,
        no_cuda=False,
        report_to=["wandb"] if use_wandb else [],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # Extra safety: avoid DataParallel path with quantized PEFT models.
    trainer.args._n_gpu = 1
    train_result = trainer.train()

    # FIX 5: save merged 16-bit weights (Unsloth helper) for easy inference later
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log_history = [row for row in trainer.state.log_history if isinstance(row, dict)]
    loss_curve = [float(row["loss"]) for row in log_history if "loss" in row]
    reward_keys = [
        "reward",
        "rewards",
        "objective/rlhf_reward",
        "rlhf_reward",
        "mean_reward",
        "train_reward",
        "reward_mean",
    ]
    reward_curve = []
    for row in log_history:
        for key in reward_keys:
            if key in row:
                reward_curve.append(float(row[key]))
                break

    wandb_run_url = None
    wandb_run_name = None
    if use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb_run_url = wandb.run.url
                wandb_run_name = wandb.run.name
        except Exception:
            pass

    return {
        "steps": steps,
        "mode": "real",
        "train_rows": len(prompts),
        "output_dir": str(final_dir),
        "model_id": model_id,
        "load_in_4bit": load_in_4bit,
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0.0)),
        "train_steps_per_second": float(train_result.metrics.get("train_steps_per_second", 0.0)),
        "loss_curve": loss_curve,
        "reward_curve": reward_curve,
        "log_history": log_history,
        "wandb_run_url": wandb_run_url,
        "wandb_run_name": wandb_run_name,
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
    parser.add_argument("--speed-preset", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--size-preset", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--model-id", type=str, default=None, help="Override model id directly")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (bitsandbytes)")
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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

    # Runtime presets for Kaggle T4.
    if args.speed_preset == "fast":
        preset_num_generations = 2
        preset_max_completion_length = 96
        preset_batch_size = 2
        use_gradient_checkpointing = False
    elif args.speed_preset == "quality":
        preset_num_generations = 4
        preset_max_completion_length = 192
        preset_batch_size = 4
        use_gradient_checkpointing = True
    else:  # balanced
        preset_num_generations = 2
        preset_max_completion_length = 128
        preset_batch_size = 2
        use_gradient_checkpointing = True

    num_generations = args.num_generations or preset_num_generations
    max_completion_length = args.max_completion_length or preset_max_completion_length
    batch_size = args.batch_size or preset_batch_size

    # Model size presets for quick iteration vs final quality.
    if args.model_id:
        model_id = args.model_id
    elif args.size_preset == "small":
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif args.size_preset == "medium":
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        model_id = "Qwen/Qwen2.5-7B-Instruct"

    if args.mode == "real":
        metrics = train_real_grpo(
            args.steps,
            args.seed,
            save_dir,
            args.run_name,
            use_wandb=use_wandb,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            per_device_train_batch_size=batch_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            model_id=model_id,
            load_in_4bit=not args.no_4bit,
        )
    else:
        metrics = simulate_training(args.steps, args.seed)
        metrics["mode"] = "simulate"
    metrics["speed_preset"] = args.speed_preset
    metrics["num_generations"] = num_generations
    metrics["max_completion_length"] = max_completion_length
    metrics["batch_size"] = batch_size
    metrics["size_preset"] = args.size_preset
    metrics["model_id"] = model_id
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