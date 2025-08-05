# Autonomous Parcel Routing (APR) - Setup & Usage Guide

This guide provides comprehensive instructions for setting up, running, and customizing the APR reinforcement learning framework for warehouse environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Detailed Setup Process](#detailed-setup-process)
- [Project Structure](#project-structure)
- [Running the System](#running-the-system)
- [Outputs & Logs](#outputs--logs)
- [Making Changes](#making-changes)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd autonomous-parcel-routing

# 2. Setup virtual environment
python3 -m venv .apr_venv
source .apr_venv/bin/activate

# 3. Install package
pip install -e .

# 4. Verify setup
python verify_setup.py

# 5. Run training
python -m apr.train --config cfg/baseline.yaml

# 6. Explore in notebooks
cd notebooks && jupyter notebook
```

---

## 🏗️ Detailed Setup Process

### Step 1: Environment Preparation

**Prerequisites:**
- Python 3.9+ installed
- Git installed
- Terminal/Command line access

**Clone the repository:**
```bash
git clone <your-repository-url>
cd autonomous-parcel-routing
```

### Step 2: Virtual Environment Setup

**Why virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system Python
- Ensures reproducible setup

**Create and activate:**
```bash
# Create virtual environment
python3 -m venv .apr_venv

# Activate (Linux/Mac)
source .apr_venv/bin/activate

# Activate (Windows)
.apr_venv\Scripts\activate

# Verify activation - should show path with .apr_venv
which python
echo $VIRTUAL_ENV
```

### Step 3: Package Installation

**Install in development mode:**
```bash
pip install -e .
```

**What this does:**
- Installs all dependencies from `pyproject.toml`
- Makes `apr` package importable
- Links to source code (changes reflect immediately)
- Installs required packages: numpy, matplotlib, pandas, gymnasium, torch, tqdm

### Step 4: Verification

**Run the setup checker:**
```bash
python verify_setup.py
```

**Expected output:**
```
🔍 Verifying APR package setup...

✅ apr package (v0.1.0)
✅ WarehouseEnv
✅ RunLogger
✅ QLearningAgent
✅ train module

🎉 All imports successful! Package is properly configured.

📁 File structure check:
✅ src/apr/__init__.py
✅ src/apr/env.py
✅ src/apr/agents/__init__.py
✅ src/apr/agents/q_learning.py
✅ cfg/baseline.yaml
✅ pyproject.toml

🚀 Setup verification complete - everything looks good!
```

---

## 📁 Project Structure

```
autonomous-parcel-routing/
├── README.md                              # Project overview
├── SETUP.md                               # This setup guide
├── LICENSE                                # License file
├── pyproject.toml                         # Package configuration & dependencies
├── verify_setup.py                       # Setup verification script
├── .gitignore                            # Git ignore rules
├── .apr_venv/                            # Virtual environment (git-ignored)
│
├── cfg/                                  # Configuration files
│   ├── baseline.yaml                     # Default training config
│   └── sweep/                            # Hyperparameter sweep configs (future)
│
├── src/                                  # Source code (importable package)
│   └── apr/                              # Main APR package
│       ├── __init__.py                   # Package init & public API
│       ├── env.py                        # WarehouseEnv class
│       ├── train.py                      # Training pipeline
│       ├── logger.py                     # RunLogger for metrics
│       ├── agents/                       # RL agent implementations
│       │   ├── __init__.py               # Agents package init
│       │   └── q_learning.py             # Q-Learning agent
│       ├── resources/                    # Static resources
│       │   ├── __init__.py               # Resources package init
│       │   └── sprites/                  # Image assets
│       │       ├── robot.png             # Robot sprite
│       │       ├── robo2_package.png     # Robot carrying package
│       │       ├── package.png           # Package sprite
│       │       ├── icon-destination.png  # Dropoff zone
│       │       └── trap.png              # Shelf/obstacle sprite
│       └── utils/                        # Utility modules
│           ├── __init__.py               # Utils package init
│           └── viz.py                    # Visualization utilities
│
├── notebooks/                            # Jupyter notebooks (exploration)
│   ├── 01_exploration.ipynb             # Environment demo & testing
│   ├── 02_training_walkthrough.ipynb    # Algorithm comparisons
│   └── 03_results_visualization.ipynb   # Results analysis & plots
│
├── outputs/                              # Training outputs (git-ignored)
│   └── runs/                             # Individual training runs
│       └── YYYY-MM-DD_HH-MM-SS_algo/    # Timestamped run directories
│           ├── config.yaml               # Copy of training config
│           ├── metrics.csv               # Episode metrics
│           └── checkpoints/              # Model checkpoints
│               ├── ckpt_ep00200.pt       # Checkpoint at episode 200
│               ├── ckpt_ep00400.pt       # Checkpoint at episode 400
│               └── ...
│
└── tests/                                # Unit tests (future)
    └── test_env.py                       # Environment tests
```

---

## 🎮 Running the System

### Training Pipeline

**Basic training:**
```bash
# Must be in project root directory
python -m apr.train --config cfg/baseline.yaml
```

**What happens:**
1. Loads configuration from `cfg/baseline.yaml`
2. Creates WarehouseEnv with specified parameters
3. Initializes Q-Learning agent
4. Runs training episodes with logging
5. Saves checkpoints every N episodes
6. Outputs metrics to CSV

**Expected output:**
```
[  200] R= 468.0  ε=0.246  → saved ckpt_ep00200.pt
[  400] R= 470.0  ε=0.201  → saved ckpt_ep00400.pt
[  600] R= 490.0  ε=0.165  → saved ckpt_ep00600.pt
```

### Interactive Environment Testing

**Console testing:**
```bash
python -c "
from apr import WarehouseEnv
env = WarehouseEnv()
state = env.reset()
env.render()  # ASCII rendering
env.print_state()  # Detailed state info
"
```

**Sprite visualization testing:**
```bash
python -c "
from apr import WarehouseEnv
env = WarehouseEnv()
env.reset()
env.render(mode='sprites')  # Rich sprite rendering
"
```

### Notebook Exploration

**Start Jupyter:**
```bash
# Navigate to notebooks directory
cd notebooks

# Start Jupyter (if installed)
jupyter notebook

# Or use your preferred notebook environment
```

**Run notebooks in order:**
1. `01_exploration.ipynb` - Environment basics
2. `02_training_walkthrough.ipynb` - Algorithm comparison
3. `03_results_visualization.ipynb` - Results analysis

---

## 📊 Outputs & Logs

### Training Outputs Location

**Base directory:** `outputs/runs/`

**Run directory format:** `YYYY-MM-DD_HH-MM-SS_algorithm/`

**Example:** `outputs/runs/2025-08-04_18-02-14_q_learning/`

### Output Files

#### `config.yaml`
- **Purpose:** Copy of training configuration for reproducibility
- **Content:** Exact parameters used for this run
- **Example:**
```yaml
env:
  size: [6, 6]
  max_steps: 200
  seed: 0
agent:
  algo: q_learning
  alpha: 0.1
  gamma: 0.95
```

#### `metrics.csv`
- **Purpose:** Episode-by-episode training metrics
- **Columns:** `episode`, `reward`, `epsilon`
- **Usage:** Plot training curves, analyze convergence
- **Example:**
```csv
episode,reward,epsilon
1,24.0,0.299
2,468.0,0.298
3,470.0,0.297
```

#### `checkpoints/`
- **Purpose:** Saved agent models at intervals
- **Format:** `ckpt_ep{episode:05d}.pt`
- **Content:** Pickled Q-table and agent state
- **Usage:** Resume training, evaluate specific checkpoints

### Log Monitoring

**Real-time training progress:**
```bash
# Watch training progress
python -m apr.train --config cfg/baseline.yaml

# Monitor latest metrics
tail -f outputs/runs/*/metrics.csv
```

**Analyze completed runs:**
```bash
# List all runs
ls -la outputs/runs/

# Check latest run metrics
cat outputs/runs/$(ls -t outputs/runs/ | head -1)/config.yaml
```

---

## 🔧 Making Changes

### Configuration Changes

**File:** `cfg/baseline.yaml`

**Environment parameters:**
```yaml
env:
  size: [6, 6]          # Grid dimensions
  max_steps: 200        # Episode length limit  
  seed: 0               # Random seed for reproducibility
  layout: "default"     # Warehouse layout type
```

**Agent parameters:**
```yaml
agent:
  algo: q_learning      # Algorithm type
  alpha: 0.1           # Learning rate
  gamma: 0.95          # Discount factor
  epsilon: 0.3         # Initial exploration rate
```

**Training parameters:**
```yaml
train:
  episodes: 1000       # Number of episodes
  log_every: 200       # Checkpoint frequency
```

### Code Changes

#### Adding New Algorithms

**1. Create agent file:**
```bash
# Create new agent file
touch src/apr/agents/sarsa.py
```

**2. Implement agent class:**
```python
# src/apr/agents/sarsa.py
class SarsaAgent:
    def __init__(self, obs_space, act_space, alpha=0.1, gamma=0.95, epsilon=0.3):
        # Implementation here
        pass
```

**3. Update agents/__init__.py:**
```python
from .q_learning import QLearningAgent
from .sarsa import SarsaAgent

__all__ = ["QLearningAgent", "SarsaAgent"]
```

**4. Update config:**
```yaml
agent:
  algo: sarsa  # Use new algorithm
```

#### Environment Modifications

**File:** `src/apr/env.py`

**Common modifications:**
- Reward structure: Modify `step()` method
- Layout: Update `_setup_warehouse_layout()`
- Rendering: Enhance `render()` methods
- Observation space: Modify state representation

#### Adding Visualization

**File:** `src/apr/utils/viz.py`

**Add new rendering functions:**
```python
def create_heatmap(q_table):
    # Implementation for Q-table heatmaps
    pass

def generate_gif(episode_frames):
    # Implementation for animated GIFs
    pass
```

### Creating New Configurations

**Create config file:**
```bash
cp cfg/baseline.yaml cfg/my_experiment.yaml
```

**Run with custom config:**
```bash
python -m apr.train --config cfg/my_experiment.yaml
```

---

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'apr'
```

**Solutions:**
```bash
# Ensure virtual environment is activated
source .apr_venv/bin/activate

# Reinstall package
pip install -e .

# Check installation
python -c "import apr; print(apr.__version__)"
```

#### Training Fails
```
Error in dynamic import or agent creation
```

**Solutions:**
```bash
# Check config file syntax
python -c "import yaml; yaml.safe_load(open('cfg/baseline.yaml'))"

# Verify agent exists
python -c "from apr.agents import QLearningAgent"

# Run with debugging
python -m apr.train --config cfg/baseline.yaml -v
```

#### Permission Errors
```
Permission denied when creating outputs/
```

**Solutions:**
```bash
# Check/create outputs directory
mkdir -p outputs/runs

# Check permissions
ls -la outputs/
```

#### Sprite Loading Issues
```
Warning: Sprite filename.png not found, using fallback
```

**Solutions:**
```bash
# Check sprites directory
ls -la src/apr/resources/sprites/

# Copy missing sprites from original project
cp /path/to/original/sprites/*.png src/apr/resources/sprites/
```

### Environment Verification

**Quick health check:**
```bash
python -c "
from apr import WarehouseEnv
from apr.agents import QLearningAgent

# Test environment
env = WarehouseEnv()
state = env.reset()
print(f'Environment OK: {state}')

# Test agent
agent = QLearningAgent(env.observation_space, env.action_space)
action = agent.act(state)
print(f'Agent OK: action={action}')

# Test step
next_state, reward, done, info = env.step(action)
print(f'Step OK: reward={reward}')
"
```

### Getting Help

**Check logs:**
```bash
# Training logs
ls -la outputs/runs/

# Python errors
python -m apr.train --config cfg/baseline.yaml 2>&1 | tee debug.log
```

**Verify setup:**
```bash
python verify_setup.py
```

**Check dependencies:**
```bash
pip list | grep -E "(numpy|matplotlib|torch|gymnasium)"
```

---

## 🎯 Next Steps

After successful setup:

1. **Explore notebooks** - Start with `01_exploration.ipynb`
2. **Run training** - Test with default configuration
3. **Analyze results** - Check training curves and metrics
4. **Customize environment** - Modify rewards, layouts, or mechanics
5. **Implement new algorithms** - Add SARSA, Double Q-Learning
6. **Hyperparameter tuning** - Optimize agent performance

---

## 📚 Additional Resources

- **Environment API:** See `src/apr/env.py` for full environment interface
- **Agent Interface:** See `src/apr/agents/q_learning.py` for agent structure  
- **Training Pipeline:** See `src/apr/train.py` for training loop details
- **Visualization:** See `src/apr/utils/viz.py` for rendering utilities

For specific implementation details, refer to the source code in `src/apr/` - it's designed to be readable and well-documented.