# Autonomous Parcel Routing — Intelligent Warehouse Path-Planning with Reinforcement Learning

> **One-sentence pitch**  
> Train an RL-powered agent that learns to navigate cluttered warehouse grids, avoid dynamic obstacles, and deliver parcels with maximum throughput and minimum collision risk.

## Why this repo exists
Modern fulfilment centres lose **up to 15 %** of daily capacity to inefficient routing and traffic jams.  
`autonomous-parcel-routing` turns a university assignment into a production-grade research sandbox where you can:

* prototype **gym-compatible** grid environments that mimic real warehouses,  
* benchmark multiple RL algorithms (DQN, PPO, A2C — plug-and-play) under identical reward functions,  
* visualise learned policies as high-fps GIFs and interactive notebooks,  
* and run fully reproducible experiments via CI/CD-backed Docker images.

## Feature highlights
| Module | What it does | In code |
|--------|--------------|---------|
| `wprl.env.WarehouseEnv` | Grid-world with dynamic fork-lift obstacles, configurable traffic density, sparse & shaped rewards | `src/wprl/env.py` |
| `wprl.agent.*` | Modular agents (tabular Q-learning → PPO) built on **PyTorch ≥2.2** | `src/wprl/agent.py` |
| `train.py` | CLI training loop (`python -m wprl.train --config cfg/baseline.yaml`) | `src/wprl/train.py` |
| `evaluate.py` | Batch evaluation + metric export (success %, steps, total reward) | `src/wprl/evaluate.py` |
| `utils/viz.py` | Matplotlib & GIF helpers for side-by-side policy playback | `src/wprl/utils/viz.py` |

## Quick start
```bash
# 1. Install editable package + extras
pip install -e ".[dev]"

# 2. Train a PPO agent for 10 k episodes
python -m wprl.train --config cfg/baseline.yaml --algo ppo --episodes 10000

# 3. Evaluate & render
python -m wprl.evaluate checkpoints/ppo_latest.pt --render --gif out/ppo_run.gif



# codeflow for understanding
                                  ┌───────────────────┐
                                  │ Configuration     │
                                  │ (baseline.yaml)   │
                                  └─────────┬─────────┘
                                            │
                                            ▼
┌───────────────────┐             ┌───────────────────┐
│ Logging           │◄────────────┤ Training Loop     │
│ (logger.py)       │             │ (train.py)        │
└─────────┬─────────┘             └─────────┬─────────┘
          │                                  │
          │                                  │
          │                                  │ Runs episodes
          │                        ┌─────────┴─────────┐
          │                        │                   │
          │                        ▼                   ▼
          │         ┌───────────────────┐   ┌───────────────────┐
          │         │ Environment       │   │ Agent             │
Saves     │         │ (env.py)          │◄──┤ (q_learning.py)   │
metrics   │         └─────────┬─────────┘   └─────────┬─────────┘
& config  │                   │                       │
          │                   │ State,                │ 
          │                   │ Reward,               │ Action
          │                   │ Done                  │
          │                   │                       │
          │                   ▼                       │
          │         ┌───────────────────┐            │
          │         │ Warehouse Grid    │            │
          │         │ with obstacles,   │◄───────────┘
          │         │ pickup, dropoff   │
          │         └───────────────────┘
          │
          ▼
┌───────────────────┐
│ Outputs           │
│ - metrics.csv     │
│ - config.yaml     │
│ - checkpoints     │
└───────────────────┘