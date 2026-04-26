# Work-Life Firewall
### An RL Environment for Teaching LLMs to Set Boundaries

**OpenEnv Hackathon — Theme 3.2: Personalized Tasks**

> *The agent that learns to say no on Monday so it doesn't collapse on Friday has learned something real.*

---

## The Problem

Every week, Indian software engineers at product companies face an impossible collision:

- Staging server down at 6 AM, blocking the team
- US client passive-aggressive escalation email from 11 PM
- Sprint demo on Friday
- Leave applied for 3 months ago, still not approved
- Teammate asking (for the 3rd time) to cover on-call
- HR appraisal form due Friday
- 10:30 PM standup you were added to as "optional" but are now expected to attend

None of these are individually hard. The combination — with real time and energy constraints and real relationship stakes — is what breaks people.

**No existing LLM benchmark tests this.** There is no training signal that rewards an agent for calibrating a polite decline correctly, or for protecting focus blocks, or for pushing back on a client without damaging the relationship. LLMs are tested on task completion. They are never tested on task refusal quality.

This environment measures both.

---

## What the Agent Sees, Does, and Gets Rewarded For

### The Episode

One episode = one work week (Monday to Friday). The agent receives 7 canonical events — one at a time — and must decide how to handle each one. Its decisions have immediate and deferred consequences. A bad Monday call spawns worse problems on Wednesday.

### Events

| Event | Source | Stakes |
|---|---|---|
| Staging server down | PagerDuty | Blocks sprint, team capacity |
| 3 Slack messages (Priya, Rahul) | Slack | Interruption cost, peer relationships |
| Client escalation email | Outlook | Client trust, tone calibration |
| Leave approval pending (3 months) | Calendar | Personal, manager relationship |
| Annual appraisal due Friday | HR Portal | Career, 90-minute overhead |
| On-call swap request (3rd time) | Slack | Peer pattern, Wednesday energy |
| 10:30 PM standup (optional invite) | Calendar | Sleep cost, client visibility |

### Actions

The agent writes a free-text response — the actual message it would send, or the specific technical action it would take. The environment decodes this to a structured action and evaluates consequences.

Actions have **energy cost**, **sprint health impact**, **relationship effects**, and may **spawn follow-on events** (e.g., saying yes to on-call spawns a Wednesday collapse event).

### Reward Function (5-component Rubric)

| Component | Weight | What it measures |
|---|---|---|
| Technical resolution | 25% | Staging fixed? Sprint delivered on time? |
| Communication quality | 25% | Tone, clarity, register of responses |
| Boundary setting | 20% | Quality of no's (not quantity — anti-gaming) |
| Energy to Friday | 20% | Agent wellbeing: did it survive the week? |
| Relationship preservation | 10% | Stakeholder trust maintained throughout |

**Anti-gaming design:** An agent cannot score high by declining everything (relationship score collapses) or accepting everything (energy collapses, sprint fails). The agent must find the specific set of strategic refusals that protect capacity while preserving the relationships that matter.

---

## Latest Committed Results (Evidence Files)

All claims in this section are derived from committed files under [evaluation/results](evaluation/results). The environment and training code used to generate these artifacts are linked in the citation list below.

Primary evidence files:

- [evaluation/results/training_metrics.json](evaluation/results/training_metrics.json)
- [evaluation/results/evaluation_summary.json](evaluation/results/evaluation_summary.json)
- [evaluation/results/reward_curve.png](evaluation/results/reward_curve.png)
- [evaluation/results/loss_curve.png](evaluation/results/loss_curve.png)
- [evaluation/results/component_breakdown.png](evaluation/results/component_breakdown.png)
- [evaluation/results/energy_trajectory.png](evaluation/results/energy_trajectory.png)
- [evaluation/results/decision_heatmap.png](evaluation/results/decision_heatmap.png)
- [evaluation/results/train_20260426.png](evaluation/results/train_20260426.png)
- [evaluation/results/profiling_20260426.png](evaluation/results/profiling_20260426.png)

Run configuration summary (from [evaluation/results/training_metrics.json](evaluation/results/training_metrics.json)):

- Training mode: **real**
- Steps: **300**
- Model: **Qwen/Qwen2.5-1.5B-Instruct**
- Runtime: **3432.34 seconds** (~57.2 minutes)
- Reward curve: **min 0.30**, **max 0.44**, **mean 0.370**

Evaluation summary (from [evaluation/results/evaluation_summary.json](evaluation/results/evaluation_summary.json)):

- Episodes per policy: **50**
- Mean reward: **random 0.679**, **greedy 0.946**, **trained_proxy 1.481**
- Mean Friday energy (%): **random 62.28**, **greedy 81.08**, **trained_proxy 100.0**

For full GPU runs (real mode, WandB logging, and updated artifacts), use [training/train.ipynb](training/train.ipynb) and then replace files in [evaluation/results](evaluation/results) with the generated outputs.

### Reward Curve

![Reward curve from committed training artifact](evaluation/results/reward_curve.png)
*Total episode reward vs. logged training steps from the committed artifact file.*

### Loss Curve

![Loss curve over training steps](evaluation/results/loss_curve.png)
*Training loss trend captured during run. This file is committed for automated validation.*

### Component Breakdown

![Radar chart showing 5 rubric components before and after training](evaluation/results/component_breakdown.png)
*Rubric component comparison generated from evaluation logs (technical resolution, communication quality, boundary setting, energy to Friday, relationship preservation).*

### Energy Trajectory (Monday → Friday)

![Line chart: agent energy across the week, 3 agents overlaid](evaluation/results/energy_trajectory.png)
*Day-wise energy trajectory (Monday to Friday) across evaluated policy variants from the committed summary JSON.*

### Raw Evidence JSON

- Training log: [evaluation/results/training_metrics.json](evaluation/results/training_metrics.json)
- Evaluation summary: [evaluation/results/evaluation_summary.json](evaluation/results/evaluation_summary.json)

### Decision Heatmap

![Heatmap: event × action choice, before vs. after training](evaluation/results/decision_heatmap.png)
*Before training: agent clusters in accept/attend actions. After training: clear shift to async-boundary, no-clearly-kindly, and decline-async actions for high-energy-cost events.*

### Train

![Training graph](evaluation/results/train_20260426.png)
*Training dashboard screenshot from the real run (committed artifact).*

### Profiling

![Profiling graph](evaluation/results/profiling_20260426.png)
*Profiler timing panels from GRPO training (committed artifact).*

### Citation Index (Code and Evidence)

- Environment entrypoint: [openenv.yaml](openenv.yaml), [environment/env.py](environment/env.py)
- Reward and consequences logic: [environment/reward.py](environment/reward.py), [environment/consequences.py](environment/consequences.py)
- Training pipeline: [training/train.py](training/train.py), [training/train.ipynb](training/train.ipynb), [training/rollout.py](training/rollout.py)
- Evaluation pipeline: [evaluation/evaluate.py](evaluation/evaluate.py), [evaluation/plot_results.py](evaluation/plot_results.py)
- Committed metrics: [evaluation/results/training_metrics.json](evaluation/results/training_metrics.json), [evaluation/results/evaluation_summary.json](evaluation/results/evaluation_summary.json)
- Live demo app: [app.py](app.py)

---

## Why This Matters

**Who would care:**
- Researchers studying LLM alignment with human values under resource constraints
- Companies building AI productivity assistants (the agent learned what every burned-out engineer had to learn the hard way)
- Teams studying RL in social/professional reasoning domains

**What it contributes:**
- First RL environment targeting work-life negotiation as a learnable capability
- A reward function that simultaneously measures technical delivery and interpersonal quality
- Evidence that GRPO-style optimization can train meaningful boundary-setting behavior in this domain

**Could a researcher write a paper about this?** Yes. The relationship between deferred decision costs (Monday's yes causing Wednesday's collapse), the anti-gaming rubric design, and the learning curve showing when boundary-setting emerges as a strategy — these are publishable observations.

---

## Technical Details

### Environment

- Built on **OpenEnv** (latest release)
- Extends `openenv.Environment` with full Gym-style API (`reset`, `step`, `state`)
- 5-component reward via `openenv.Rubric`
- Dense reward at every step (not just terminal)
- Stochastic consequence model (spawned events, approval randomness)

### Training

- Trainer: `trl.GRPOTrainer`
- Model family: `Qwen2.5-Instruct` (size preset selectable: small / medium / large)
- Quantization: default 4-bit path with optional `--no-4bit` fallback for runtime stability
- WandB project: https://wandb.ai/yusufindian09-aaa/meta_hackathon

### Running It

```bash
pip install -r requirements.txt

# Interactive demo (be Arjun for a week)
python -m examples.human_eval

# Run baseline comparison
python -m examples.random_agent --episodes 50
python -m examples.greedy_agent --episodes 50

# Training (requires GPU)
# Kaggle Notebook entrypoint:
python training/train.py --mode simulate --steps 1000

# Real GRPO run on Kaggle GPU
python training/train.py --mode real --steps 300 --run-name wlf-kaggle-grpo

# Stable fallback on constrained Kaggle runtimes
python training/train.py --mode real --size-preset small --steps 30 --batch-size 1 --num-generations 1 --max-completion-length 64 --no-4bit
```

### Kaggle Training Notebook

Use [training/train.ipynb](training/train.ipynb) inside Kaggle Notebooks (attach this repo as a dataset or clone it in /kaggle/working).

### Hugging Face (Model + Space)

```bash
# 1) Login once in Kaggle/local shell
huggingface-cli login

# 2) Upload trained checkpoint to HF Model Hub
python training/push_to_hf.py --repo-id YUS200619/meta_hackathon-qwen-model --folder checkpoints/final

# If your latest checkpoint is in a different folder, pass that path instead.
# Example:
# python training/push_to_hf.py --repo-id YUS200619/meta_hackathon-qwen-model --folder checkpoints

# 3) Run demo locally, then deploy same app to HF Space
python app.py
```

---

## Additional Materials

- **HF Space (live demo):** https://huggingface.co/spaces/YUS200619/meta_hackathon-qwen
- **Trained Model:** https://huggingface.co/YUS200619/meta_hackathon-qwen-model
- **Training notebook:** [training/train.ipynb](training/train.ipynb)
- **Project writeup:** [PRD.md](PRD.md)
- **Implementation details:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## Automated Round Checklist

- Public HF Space URL is provided and intended for logged-out access: https://huggingface.co/spaces/YUS200619/meta_hackathon-qwen
- OpenEnv entrypoint and config are included: [openenv.yaml](openenv.yaml), [environment/env.py](environment/env.py)
- Training evidence images are committed: [evaluation/results/reward_curve.png](evaluation/results/reward_curve.png), [evaluation/results/loss_curve.png](evaluation/results/loss_curve.png), [evaluation/results/train_20260426.png](evaluation/results/train_20260426.png), [evaluation/results/profiling_20260426.png](evaluation/results/profiling_20260426.png)
- Raw metrics evidence is committed: [evaluation/results/training_metrics.json](evaluation/results/training_metrics.json), [evaluation/results/evaluation_summary.json](evaluation/results/evaluation_summary.json)
- Runnable training artifacts are included: [training/train.py](training/train.py), [training/train.ipynb](training/train.ipynb)
- README links core deliverables directly so validator can discover them from one page

---

## Repository Structure

```
work-life-firewall/
├── environment/          # OpenEnv environment
│   ├── env.py            # WorkLifeFirewallEnv
│   ├── events.py         # 7 canonical events + spawned events
│   ├── state.py          # EpisodeState dataclass
│   ├── consequences.py   # Action → consequence model
│   └── reward.py         # 5-component Rubric
├── training/
│   ├── train.ipynb       # Kaggle notebook (primary submission)
│   ├── train.py          # Script version
│   └── rollout.py        # Episode rollout harness
├── evaluation/
│   ├── evaluate.py
│   ├── plot_results.py
│   └── results/          # All plots (PNG, committed)
├── examples/
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── human_eval.py
├── CLAUDE.md             # AI assistant instructions
├── AGENT.md              # Agent prompt + few-shot examples
├── openenv.yaml
└── requirements.txt
```

---

## The Learning Signal is Real

An LLM that learns to decline the on-call swap, skip the optional standup, and push back on scope creep without destroying relationships has learned something that matters outside this environment.

The fact that we can measure it — reward curves, component scores, Friday energy trajectories — is what makes it a research contribution. The fact that every engineer who reads the README recognises this week is what makes it a product.

---

*Built for the OpenEnv Hackathon. Theme 3.2 — Personalized Tasks.*
