# CLAUDE.md
## Work-Life Firewall — OpenEnv RL Environment

This file tells Claude Code (or any AI assistant) how to work on this codebase.
Read this before touching any file.

---

## What this project is

An RL training environment built on OpenEnv where an LLM agent learns to navigate a real Indian software engineer's work week. The agent must triage competing demands — broken staging server, client escalation, peer requests, sprint deadline, leave approval — and learn that strategic boundary-setting on Monday is what protects Friday delivery.

**This is not a chatbot. This is not a productivity tool. This is an RL environment.**

The agent's actions get scored by a rubric. The rubric's signal trains a 7B LLM via GRPO. The goal is to make the trained model measurably better at a real capability: knowing when to say no, and how to say it well.

---

## Project structure

```
work-life-firewall/
├── environment/          ← The OpenEnv environment (core)
│   ├── env.py            ← WorkLifeFirewallEnv class
│   ├── events.py         ← Event catalogue (7 canonical events)
│   ├── state.py          ← EpisodeState dataclass
│   ├── consequences.py   ← Action → consequence model
│   ├── reward.py         ← 5-component Rubric
│   └── judge.py          ← Communication quality scorer
├── training/
│   ├── train.ipynb       ← PRIMARY SUBMISSION ARTIFACT (Colab)
│   ├── train.py          ← Script version
│   ├── rollout.py        ← Episode rollout harness
│   └── grpo_config.py    ← GRPO hyperparameters
├── evaluation/
│   ├── evaluate.py       ← Run trained vs baseline
│   ├── plot_results.py   ← Generate all plots
│   └── results/          ← PNGs committed here
├── examples/
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── human_eval.py     ← Interactive demo
├── CLAUDE.md             ← This file
├── README.md
├── AGENT.md
├── openenv.yaml
└── requirements.txt
```

---

## Non-negotiable constraints

### OpenEnv API compliance
- `WorkLifeFirewallEnv` MUST extend `openenv.Environment`
- `reset()` returns observation dict
- `step(action)` returns `(observation, reward, done, info)`
- `state()` returns current state dict
- Do NOT use any of these as MCP tool names: `reset`, `step`, `state`, `close`
- Never import server internals in client code

### Reward function design principles
- Reward must be anti-gameable: agent cannot score high by declining everything (relationship collapses) or accepting everything (energy collapses)
- Dense reward at every step — not just terminal
- 5 rubric components via `openenv.Rubric` — do not collapse to a single scalar
- Boundary-setting score must reward *quality* no's, not quantity

### Training pipeline
- Model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Trainer: `trl.GRPOTrainer`
- Log to WandB: every run, every metric
- Save plots as PNG to `evaluation/results/` — never leave them only in Colab

---

## When writing new code

### State mutations
- Never mutate `EpisodeState` in place. Always copy, modify, return.
- All energy values are `float` in `[0.0, 1.0]`. Clamp after every operation.
- All relationship scores are `float` in `[0.0, 1.0]`. Clamp after every operation.

### Adding a new event
1. Add `Event(...)` to `CANONICAL_EVENTS` list in `events.py`
2. Add decode keywords to `_decode_action()` in `env.py`
3. Add event-specific state handling to `consequences.py`
4. Test with `python -m examples.human_eval` before committing

### Adding a new rubric component
1. Write a scorer function: `(state, decisions) -> float` in `[0.0, 1.0]`
2. Add `Component(name, weight, scorer)` to `build_rubric()` in `reward.py`
3. All weights must sum to 1.0
4. Re-run baseline eval to verify the component produces variance

### Prompt format
The agent prompt lives in `training/rollout.py:build_prompt()`. When modifying:
- Keep the persona ("Arjun Sharma, Bangalore product company") — it primes cultural context
- Include current energy and sprint health — the agent needs these to reason about tradeoffs
- Never reveal the reward function in the prompt
- End with `"Response:"` and nothing after — clean generation target

---

## Common mistakes to avoid

1. **Putting business logic in `env.py`** — consequence logic belongs in `consequences.py`, scoring in `reward.py`
2. **Stochastic rewards** — randomness goes in the environment (e.g., leave approval), not in the reward function
3. **Terminal-only reward** — GRPO needs dense signal. Every `step()` call must return a non-zero reward for meaningful actions
4. **Importing `torch` in `environment/`** — the environment must be framework-agnostic
5. **Hardcoding action IDs in `reward.py`** — use the `decisions` list from state, not string matching in the scorer
6. **Leaving plots in Colab cells** — every plot must be `plt.savefig("evaluation/results/X.png")` AND committed

---

## Running locally

```bash
# Install
pip install -r requirements.txt

# Run interactive demo
python -m examples.human_eval

# Run random agent baseline (50 episodes)
python -m examples.random_agent --episodes 50

# Run training (requires GPU)
python training/train.py --steps 1000 --wandb

# Generate evaluation plots
python evaluation/plot_results.py --checkpoint checkpoints/final
```

---

## Hackathon submission checklist

Before submitting, verify ALL of these:

- [ ] `openenv.yaml` valid and matches actual entry point
- [ ] `WorkLifeFirewallEnv` passes OpenEnv validation: `openenv validate .`
- [ ] Training notebook runs end-to-end in fresh Colab (no local dependencies)
- [ ] Loss and reward plots saved as PNG in `evaluation/results/`
- [ ] Plots embedded in README with captions
- [ ] WandB run URL in README
- [ ] HF Space link in README
- [ ] All other materials (blog, slides, video) linked from README
- [ ] Repo size reasonable (no large video files)
- [ ] README readable in under 5 minutes

---

## Architecture decisions and why

**Why GRPO over PPO?**
GRPO (Group Relative Policy Optimisation) doesn't require a separate critic model. For a 7B model on Colab, removing the critic cuts memory by ~40%. The relative scoring within a group of completions also works well for our multi-component reward.

**Why action decoding instead of constrained generation?**
Constrained generation (forcing the model to pick from a menu) removes the capability we're training. We want the model to learn to *express* boundary-setting in natural language. The decoder maps that expression to structured actions for consequence evaluation.

**Why dense reward?**
Sparse (terminal-only) reward with 14 decision steps means 13 steps get no signal. GRPO needs signal to rank completions. Dense reward per step gives the trainer enough signal to learn meaningful decision ordering.

**Why 7 canonical events?**
Enough to create realistic combinatorial pressure (the collision of all 7 is what breaks people) but small enough for an episode to fit in a 4096-token context window with full history.
