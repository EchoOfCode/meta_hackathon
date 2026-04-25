# Tech Stack
## Work-Life Firewall — OpenEnv RL Environment

---

## Core Framework

| Layer | Technology | Why |
|---|---|---|
| RL Environment | **OpenEnv** (latest) | Required by hackathon; MCP-native, LLM-optimised |
| Training | **Unsloth + GRPO** | Fast 7B fine-tuning on consumer GPU; GRPO works without critic model |
| Base Model | **Qwen2.5-7B-Instruct** | Strong instruction following; runs well in Kaggle GPU notebooks |
| Experiment Tracking | **Weights & Biases** | Reward curves, loss plots, per-component rubric scores |
| Hosting | **Hugging Face Spaces** | Required by hackathon for discoverability |

---

## Environment Stack

```
openenv/                        # OpenEnv base classes
├── Environment                 # Base class we extend
├── Rubric / Component          # Reward composition
└── MCPEnvironment              # Optional: MCP tool exposure

fastapi                         # MCP server (if MCPEnvironment)
pydantic v2                     # State schema validation
python-dotenv                   # Config management
```

### Environment internals

```python
# State management
dataclasses                     # Typed state objects
typing (TypedDict, Literal)     # Action/event type safety

# Consequence simulation
numpy                           # Energy/health curves
random                          # Stochastic event spawning

# Communication quality scoring
anthropic SDK                   # LLM-as-judge for tone/clarity rubric
# OR
sentence-transformers           # Embedding-based quality scoring (faster, cheaper)
```

---

## Training Stack

```
unsloth                         # LoRA fine-tuning, 2x faster than HF
trl >= 0.12                     # GRPO trainer (GRPOTrainer)
transformers >= 4.45            # Model loading
datasets                        # Episode rollout dataset format
torch >= 2.1                    # Base tensor ops
bitsandbytes                    # 4-bit quantisation for Kaggle GPU limits
accelerate                      # Multi-GPU (optional)
wandb                           # Logging
```

### Training configuration

```python
# Model
model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
max_seq_length = 4096           # Enough for full episode context
load_in_4bit = True             # Kaggle-compatible

# LoRA
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# GRPO
num_generations = 4             # Rollouts per prompt
max_new_tokens = 512            # Per decision response
temperature = 0.8               # Exploration
learning_rate = 2e-5
num_train_epochs = 3
gradient_accumulation_steps = 4
```

---

## Evaluation Stack

```
matplotlib / seaborn            # Reward curves, component breakdown plots
pandas                          # Episode result aggregation
jupyter / ipynb                 # Kaggle notebook format
```

### Key plots to generate

1. **Reward curve** — total episode reward vs training step (trained vs baseline on same axes)
2. **Component breakdown** — 5 rubric components over training (radar or stacked bar)
3. **Energy trajectory** — agent energy Mon→Fri, before vs after training
4. **Decision heatmap** — which actions agent chose per event, before vs after
5. **Sprint delivery rate** — binary metric across episodes

---

## Repository Structure

```
work-life-firewall/
├── README.md
├── CLAUDE.md
├── AGENT.md
├── openenv.yaml                # Manifest
│
├── environment/
│   ├── __init__.py
│   ├── env.py                  # WorkLifeFirewallEnv (extends Environment)
│   ├── events.py               # Event catalogue + spawning logic
│   ├── state.py                # State dataclass
│   ├── consequences.py         # Action → consequence model
│   ├── reward.py               # Rubric definitions + scoring
│   └── judge.py                # LLM-as-judge for communication quality
│
├── training/
│   ├── train.ipynb             # Kaggle notebook (primary submission artifact)
│   ├── train.py                # Script version
│   ├── grpo_config.py          # GRPO hyperparameters
│   └── rollout.py              # Episode rollout harness
│
├── evaluation/
│   ├── evaluate.py             # Run trained vs baseline
│   ├── plot_results.py         # Generate all plots
│   └── results/
│       ├── reward_curve.png
│       ├── component_breakdown.png
│       ├── energy_trajectory.png
│       └── decision_heatmap.png
│
├── examples/
│   ├── random_agent.py         # Baseline: random action selection
│   ├── greedy_agent.py         # Baseline: always accept
│   └── human_eval.py           # Interactive human play (demo)
│
└── requirements.txt
```

---

## Infrastructure Requirements

| Resource | Spec | Where |
|---|---|---|
| Training GPU | P100/T4/L4 (4-bit) | Kaggle Notebooks |
| Training time | ~2-3 hours for 1000 episodes | Kaggle |
| HF Space | CPU basic (env demo) | Hugging Face |
| WandB | Free tier | wandb.ai |

---

## Dependency Install Order

```bash
# 1. OpenEnv
pip install openenv

# 2. Unsloth
pip install unsloth

# 3. Training dependencies
pip install trl transformers datasets accelerate bitsandbytes

# 4. Tracking + eval
pip install wandb matplotlib seaborn pandas

# 5. Optional: LLM judge
pip install anthropic sentence-transformers
```

---

## Environment Compatibility

| Platform | Status |
|---|---|
| Kaggle Notebooks (T4/P100/L4) | ✓ Supported (4-bit) |
| Local (CUDA) | ✓ Supported |
| HF Spaces (CPU) | ✓ Env demo only (no training) |
| M1/M2 Mac | ✓ Env demo only |
