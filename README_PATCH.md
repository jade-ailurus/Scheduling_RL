# Scheduling_RL – W&B logging + long-training runners (patch)

This folder contains **new Python scripts** you can copy into the root of the GitHub repo:
`https://github.com/jade-ailurus/Scheduling_RL/tree/main`

Goal
- Increase training iterations (episodes) without editing core simulator files
- Add optional **Weights & Biases** logging
- Add a new robust RL runner: **Domain Randomization + QR-DQN + CVaR policy**

## 0) Security note (important)
Do **not** hardcode W&B API keys into code.

Set your key as an environment variable instead:
```bash
export WANDB_API_KEY="YOUR_KEY"
```

If `WANDB_API_KEY` is not set, scripts will automatically run in **offline** mode.

## 1) Files to copy into the repo root

Copy these files:
- `agents.py`
- `state_utils.py`
- `dr_utils.py`
- `wandb_utils.py`
- `train_dqn_wandb.py`
- `train_dr_cvar_qrdqn_wandb.py`
- `benchmark_compare_wandb.py`

## 2) Install deps (once)

Inside your repo environment:
```bash
pip install -U simpy pandas numpy torch wandb
```

## 3) Run baseline DQN (long training)

```bash
python train_dqn_wandb.py \
  --episodes 2000 \
  --eval-every 50 \
  --eval-episodes 5 \
  --train-perturb basic \
  --eval-perturb strong \
  --wandb
```

Outputs:
- Checkpoint: `checkpoints/dqn_charging.pt` (best eval saved)
- W&B charts:
  - `train/total_delay_min` over episode
  - `eval/total_delay_min` every `eval-every`
  - baseline evaluation at step 0

## 4) Run robust DR-CVaR-QRDQN (new runner)

```bash
python train_dr_cvar_qrdqn_wandb.py \
  --episodes 2000 \
  --eval-every 50 \
  --eval-episodes 5 \
  --train-perturb strong \
  --eval-perturb strong \
  --risk-mode cvar \
  --cvar-alpha 0.2 \
  --n-quantiles 51 \
  --wandb
```

Notes:
- `--train-perturb strong` is the “domain randomization” part.
- CVaR is used for **action selection** (risk-averse).

### “Modified DR-CVaR-QRDQN”
If your variant is “reduce distributional complexity”, you can do:
- fewer quantiles:
  - `--n-quantiles 21` (or 11)
- or make it less risk-averse:
  - increase `--cvar-alpha` toward 0.5

## 5) Benchmark (heuristic vs RL) under basic/weak/strong perturbations

```bash
python benchmark_compare_wandb.py \
  --dqn-ckpt checkpoints/dqn_charging.pt \
  --qrdqn-ckpt checkpoints/dr_cvar_qrdqn.pt \
  --n-seeds 20 \
  --heuristic-dispatch BIDDING \
  --rl-dispatch FIFO \
  --risk-mode cvar \
  --cvar-alpha 0.2 \
  --wandb
```

Outputs:
- CSV summary: `benchmark_out/summary_<timestamp>.csv`
- W&B aggregated metrics logged as scalars.

## 6) “How many episodes until it improves?”
Both training scripts log an **untrained baseline eval at step 0**, and eval every N episodes.
In W&B:
- plot `eval/total_delay_min` vs step
- visually find the “knee” / stabilization point
- or export the history as CSV and compute the first step achieving a target improvement threshold
