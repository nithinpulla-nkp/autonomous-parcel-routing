# Usage Guide - Autonomous Parcel Routing

> **Complete guide to running agents, notebooks, and customizing the framework**

This guide covers all aspects of using the Autonomous Parcel Routing framework, from basic training to advanced customization.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Running Agents via Python Scripts](#running-agents-via-python-scripts)
- [Running Agents via Notebooks](#running-agents-via-notebooks)
- [Configuration & Customization](#configuration--customization)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Ensure you have completed the setup process from [SETUP.md](SETUP.md):

```bash
# Quick setup verification
source .apr_venv/bin/activate
python verify_setup.py
```

Expected output: ✅ All imports successful!

---

## Running Agents via Python Scripts

### 🤖 Individual Agent Training

Each algorithm can be trained using the command-line interface:

#### Q-Learning
```bash
# Basic training
python src/apr/train.py --config cfg/baseline.yaml

# Expected output:
# [  200] R= 468.0  ε=0.246  → saved ckpt_ep00200.pt
# [  400] R= 470.0  ε=0.201  → saved ckpt_ep00400.pt
# [  600] R= 490.0  ε=0.165  → saved ckpt_ep00600.pt
```

#### Double Q-Learning
```bash
python src/apr/train.py --config cfg/double_q_learning.yaml

# Benefits: Reduces overestimation bias with dual Q-tables
# Expected: Slightly better final performance than Q-Learning
```

#### SARSA
```bash
python src/apr/train.py --config cfg/sarsa.yaml

# Benefits: On-policy learning, more conservative exploration
# Expected: More stable but potentially slower convergence
```

#### SARSA(λ) with Eligibility Traces
```bash
# Create configuration file first
cp cfg/sarsa.yaml cfg/sarsa_lambda.yaml

# Edit cfg/sarsa_lambda.yaml to add:
# agent:
#   algo: sarsa_lambda
#   lambda_: 0.9  # Eligibility trace decay

python src/apr/train.py --config cfg/sarsa_lambda.yaml

# Benefits: Faster credit assignment, better for delayed rewards
```

### 🔍 Agent Validation

Validate that agents are actually learning (not just memorizing):

```bash
# Validate Q-Learning
python src/validate_agent.py --algorithm q_learning --train-episodes 300 --test-episodes 50

# Validate Double Q-Learning
python src/validate_agent.py --algorithm double_q_learning --train-episodes 300 --test-episodes 50

# Validate SARSA
python src/validate_agent.py --algorithm sarsa --train-episodes 300 --test-episodes 50

# With visualization
python src/validate_agent.py --algorithm q_learning --visualize --save-plots outputs/validation_plots.png
```

**Validation Output:**
```
🔍 RL Agent Validation Framework
==================================================
Algorithm: q_learning

📊 VALIDATION SUMMARY REPORT
============================================================
Overall Assessment: PASS

Component Scores:
  ✅ Learning: PASS
  ✅ Convergence: PASS  
  ✅ Exploration: PASS
  ✅ Generalization: PASS
  ✅ Local_optima: PASS

Key Metrics:
  Learning vs Random: 817.2 reward improvement
  Statistical Significance: p=0.0000
  State Coverage: 72.2%
  Action Diversity: 65.4%

🎉 VALIDATION PASSED: Agent is learning properly!
```

### 🏁 Algorithm Comparison

Compare all algorithms head-to-head:

```bash
# Compare with default settings
python src/compare_algorithms.py

# Expected output:
# 🏁 Starting Algorithm Comparison
# 🤖 Training Q-Learning...
# 🤖 Training Double Q-Learning...
# 🤖 Training SARSA...
# 
# 📊 Comparison Summary:
# Q-Learning: Final Avg = 465.1, Max = 490.0
# Double Q-Learning: Final Avg = 475.7, Max = 490.0  
# SARSA: Final Avg = 364.8, Max = 488.0
```

### 🎯 Agent Evaluation

Evaluate trained agents with comprehensive metrics:

```bash
# Evaluate single agent
python src/evaluate_agents.py single q_learning --episodes 100 --visualize

# Compare multiple agents
python src/evaluate_agents.py compare q_learning double_q_learning sarsa --episodes 100 --visualize

# Save results to directory
python src/evaluate_agents.py compare q_learning double_q_learning sarsa --output-dir outputs/comparison_results --visualize
```

---

## Running Agents via Notebooks

### 📓 Individual Agent Notebooks

Each algorithm has a complete notebook with training, validation, and testing:

#### Q-Learning Complete Workflow
```bash
cd notebooks
jupyter notebook 07_q_learning_complete.ipynb
```

**What it includes:**
- Environment setup and visualization
- Complete training with metrics tracking
- Q-table analysis and policy visualization
- Rigorous validation with statistical testing
- Comprehensive evaluation across multiple seeds
- Agent demonstration episodes
- Results saving with proper directory structure

#### SARSA Complete Workflow
```bash
jupyter notebook 08_sarsa_complete.ipynb
```

**SARSA-specific features:**
- On-policy learning analysis
- Conservative exploration behavior
- Safety vs performance trade-offs
- Comparison with Q-Learning characteristics

#### SARSA(λ) Complete Workflow
```bash
jupyter notebook 09_sarsa_lambda_complete.ipynb
```

**SARSA(λ)-specific features:**
- Eligibility traces visualization and analysis
- Credit assignment improvements
- Lambda parameter effects
- Trace activity monitoring

#### Double Q-Learning Training
```bash
jupyter notebook 04_double_q_learning_training.ipynb
```

**Double Q-Learning features:**
- Dual Q-table analysis
- Overestimation bias reduction
- Q1 vs Q2 vs Combined Q-table comparisons

### 📊 Multi-Agent Analysis Notebooks

#### Agent Validation
```bash
jupyter notebook 05_agent_validation.ipynb
```

**Comprehensive validation of all agents:**
- Statistical significance testing
- Convergence and stability analysis
- Exploration vs exploitation balance
- Generalization across scenarios
- Local optima detection
- Policy analysis and comparison

#### Algorithm Comparison
```bash
jupyter notebook 06_agent_comparison.ipynb
```

**Head-to-head algorithm comparison:**
- Training performance curves
- Convergence characteristics analysis
- Statistical significance testing
- Sample efficiency comparison
- Final performance rankings
- Algorithm recommendation based on use case

### 🎮 Quick Interactive Testing

```python
# In any notebook or Python session
import sys
sys.path.append('../src')  # if in notebooks/ directory

from apr import WarehouseEnv
from apr.agents import create_agent

# Create environment
env = WarehouseEnv(seed=42)
env.reset()
env.render(mode='human')  # Visual rendering

# Create and test agent
agent = create_agent('q_learning', env.observation_space, env.action_space)
state = env.agent_pos
action = agent.act(state)
next_state, reward, done, info = env.step(action)

print(f"State: {state} → Action: {action} → Next: {next_state} (Reward: {reward})")
```

---

## Configuration & Customization

### 📁 Configuration Files Location

All configuration files are in the `cfg/` directory:

```
cfg/
├── baseline.yaml          # Default Q-Learning configuration
├── double_q_learning.yaml # Double Q-Learning configuration  
├── sarsa.yaml            # SARSA configuration
└── custom_config.yaml    # Your custom configurations
```

### ⚙️ Basic Configuration Structure

```yaml
# Example: cfg/baseline.yaml
env:
  size: [6, 6]          # Warehouse grid dimensions
  max_steps: 200        # Maximum steps per episode
  seed: 0               # Random seed for reproducibility
  layout: "default"     # Warehouse layout type

agent:
  algo: q_learning      # Algorithm: q_learning, double_q_learning, sarsa, sarsa_lambda
  alpha: 0.1           # Learning rate (0.01 - 0.5)
  gamma: 0.95          # Discount factor (0.9 - 0.99)
  epsilon: 0.3         # Initial exploration rate (0.1 - 0.5)

train:
  episodes: 1000       # Number of training episodes
  log_every: 200       # Checkpoint frequency
```

### 🎛️ Parameter Customization Examples

#### Environment Parameters

```yaml
env:
  size: [8, 8]          # Larger warehouse (more challenging)
  max_steps: 300        # Longer episodes
  seed: 42              # Different random seed
  layout: "complex"     # More obstacles (if implemented)
  
  # Advanced options (if you modify env.py):
  num_packages: 3       # Multiple packages to collect
  dynamic_obstacles: true   # Moving obstacles
  reward_shaping: true  # Additional reward signals
```

#### Agent Hyperparameters

**Q-Learning Tuning:**
```yaml
agent:
  algo: q_learning
  alpha: 0.05          # Lower learning rate (more stable)
  gamma: 0.99          # Higher discount (more future-focused)
  epsilon: 0.4         # Higher exploration
  epsilon_decay: 0.995 # Slower exploration decay
  epsilon_min: 0.01    # Lower minimum exploration
```

**Double Q-Learning Tuning:**
```yaml
agent:
  algo: double_q_learning
  alpha: 0.1           # Standard learning rate
  gamma: 0.95          # Standard discount
  epsilon: 0.3         # Balanced exploration
  # Double Q-Learning automatically handles overestimation bias
```

**SARSA Tuning:**
```yaml
agent:
  algo: sarsa
  alpha: 0.15          # Slightly higher for on-policy learning
  gamma: 0.95          # Standard discount
  epsilon: 0.2         # Lower exploration (more conservative)
  epsilon_decay: 0.999 # Slower decay for consistent exploration
```

**SARSA(λ) Tuning:**
```yaml
agent:
  algo: sarsa_lambda
  alpha: 0.1           # Standard learning rate
  gamma: 0.95          # Standard discount  
  epsilon: 0.3         # Standard exploration
  lambda_: 0.9         # Eligibility trace decay (0.7-0.95)
  # Higher lambda = more Monte Carlo-like
  # Lower lambda = more TD-like
```

#### Training Parameters

```yaml
train:
  episodes: 2000       # More training for complex environments
  log_every: 100       # More frequent checkpoints
  
  # Advanced options:
  eval_every: 500      # Evaluation frequency
  save_best: true      # Save best performing model
  early_stopping: true # Stop if converged
```

### 🔧 Creating Custom Configurations

#### Example: High-Performance Q-Learning
```yaml
# cfg/q_learning_optimized.yaml
env:
  size: [6, 6]
  max_steps: 200
  seed: 42

agent:
  algo: q_learning
  alpha: 0.05          # Lower for stability
  gamma: 0.99          # Higher for long-term planning
  epsilon: 0.5         # Higher initial exploration
  epsilon_decay: 0.998 # Slower decay
  epsilon_min: 0.02    # Lower minimum

train:
  episodes: 1500       # More training
  log_every: 150
```

#### Example: Conservative SARSA
```yaml
# cfg/sarsa_conservative.yaml
env:
  size: [6, 6]
  max_steps: 200
  seed: 42

agent:
  algo: sarsa
  alpha: 0.08          # Moderate learning rate
  gamma: 0.95          # Standard discount
  epsilon: 0.15        # Low exploration (safety-first)
  epsilon_decay: 0.9995 # Very slow decay
  epsilon_min: 0.05    # Higher minimum (always some exploration)

train:
  episodes: 1200
  log_every: 200
```

#### Example: Fast-Learning SARSA(λ)
```yaml
# cfg/sarsa_lambda_fast.yaml
env:
  size: [6, 6]
  max_steps: 200
  seed: 42

agent:
  algo: sarsa_lambda
  alpha: 0.15          # Higher learning rate
  gamma: 0.95          # Standard discount
  epsilon: 0.3         # Balanced exploration
  lambda_: 0.95        # High eligibility trace (more Monte Carlo-like)

train:
  episodes: 800        # Fewer episodes needed due to eligibility traces
  log_every: 100
```

### 🚀 Running Custom Configurations

```bash
# Create your configuration
cp cfg/baseline.yaml cfg/my_experiment.yaml
# Edit cfg/my_experiment.yaml with your parameters

# Train with custom config
python src/apr/train.py --config cfg/my_experiment.yaml

# Validate custom configuration
python src/validate_agent.py --algorithm q_learning --train-episodes 300

# Compare custom vs baseline
python src/compare_algorithms.py  # Modify script to include custom config
```

### 📊 Parameter Impact Guide

| Parameter | Effect | Recommended Range | Notes |
|-----------|--------|-------------------|-------|
| `alpha` (learning rate) | Higher = faster learning, less stable | 0.05 - 0.2 | Start with 0.1 |
| `gamma` (discount) | Higher = more future-focused | 0.9 - 0.99 | Use 0.95 for most cases |
| `epsilon` (exploration) | Higher = more exploration | 0.1 - 0.5 | Depends on environment complexity |
| `epsilon_decay` | Faster decay = quicker convergence | 0.995 - 0.999 | Balance exploration vs exploitation |
| `lambda_` (SARSA-λ only) | Higher = more Monte Carlo-like | 0.7 - 0.95 | Use 0.9 as starting point |
| `episodes` | More episodes = better learning | 500 - 2000 | Depends on environment complexity |

---

## Advanced Usage

### 🔬 Research & Experimentation

#### Hyperparameter Sweeps
```python
# Example hyperparameter sweep (create as Python script)
import yaml
import subprocess
from itertools import product

# Define parameter ranges
alphas = [0.05, 0.1, 0.15, 0.2]
gammas = [0.9, 0.95, 0.99]
epsilons = [0.2, 0.3, 0.4]

# Load base configuration
with open('cfg/baseline.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Run sweep
for alpha, gamma, epsilon in product(alphas, gammas, epsilons):
    # Modify configuration
    config = base_config.copy()
    config['agent']['alpha'] = alpha
    config['agent']['gamma'] = gamma
    config['agent']['epsilon'] = epsilon
    
    # Save configuration
    config_name = f'sweep_a{alpha}_g{gamma}_e{epsilon}.yaml'
    with open(f'cfg/{config_name}', 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    subprocess.run(['python', 'src/apr/train.py', '--config', f'cfg/{config_name}'])
```

#### Custom Environment Modifications

**Modify reward structure** (edit `src/apr/env.py`):
```python
# In WarehouseEnv.step() method
def step(self, action):
    # ... existing code ...
    
    # Custom reward modifications
    if (nr, nc) in self.packages_remaining:
        reward += 50  # Higher pickup reward
    
    if self.carrying_packages and (nr, nc) == self.dropoff:
        reward += 200  # Higher delivery reward
    
    # Distance-based reward shaping
    if self.carrying_packages:
        distance_to_dropoff = abs(nr - self.dropoff[0]) + abs(nc - self.dropoff[1])
        reward -= distance_to_dropoff * 0.1  # Penalty for being far from dropoff
    
    return next_state, reward, done, info
```

#### Custom Agent Implementation

**Create new agent** (example: `src/apr/agents/my_agent.py`):
```python
from .base import TabularAgent
import numpy as np

class MyCustomAgent(TabularAgent):
    def __init__(self, obs_space, act_space, alpha=0.1, gamma=0.95, epsilon=0.3, **kwargs):
        super().__init__(obs_space, act_space, alpha, gamma, epsilon)
        # Your custom initialization
        
    def act(self, state, training=True):
        # Your custom action selection
        pass
        
    def learn(self, state, action, reward, next_state, done):
        # Your custom learning rule
        pass
```

**Register new agent** (edit `src/apr/agents/registry.py`):
```python
from .my_agent import MyCustomAgent

AGENT_REGISTRY = {
    'q_learning': QLearningAgent,
    'double_q_learning': DoubleQLearningAgent,
    'sarsa': SarsaAgent,
    'sarsa_lambda': SarsaLambdaAgent,
    'my_custom': MyCustomAgent,  # Add your agent
}
```

### 📈 Performance Monitoring

#### Real-time Training Monitoring
```bash
# Monitor training progress in real-time
tail -f outputs/runs/*/metrics.csv

# Plot training curves during training
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find latest run
latest_run = max(glob.glob('outputs/runs/*/'), key=os.path.getctime)
df = pd.read_csv(f'{latest_run}/metrics.csv')

plt.plot(df['episode'], df['reward'])
plt.xlabel('Episode')  
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
"
```

#### Batch Evaluation
```python
# Evaluate multiple saved models
import glob
from apr.agents.q_learning import QLearningAgent
from apr.evaluate import AgentEvaluator

evaluator = AgentEvaluator()

# Load and evaluate all checkpoints
checkpoints = glob.glob('outputs/runs/*/checkpoints/*.pt')
results = {}

for checkpoint in checkpoints:
    agent = QLearningAgent.load(checkpoint)
    result = evaluator.evaluate_agent(agent, num_episodes=50)
    results[checkpoint] = result['aggregated_results']['overall_statistics']['mean_reward']

# Find best checkpoint
best_checkpoint = max(results.items(), key=lambda x: x[1])
print(f"Best checkpoint: {best_checkpoint[0]} with reward {best_checkpoint[1]}")
```

---

## Troubleshooting

### 🔧 Common Issues

#### Configuration Errors
```bash
# Error: Invalid YAML syntax
python -c "import yaml; yaml.safe_load(open('cfg/my_config.yaml'))"

# Error: Unknown algorithm
# Check that algorithm name matches registry in src/apr/agents/registry.py
python -c "from apr.agents import list_available_agents; print(list_available_agents())"
```

#### Training Issues
```bash
# Error: No improvement in training
# Solution: Check hyperparameters, increase episodes, verify environment

# Error: Agent not exploring
# Solution: Increase epsilon, slower epsilon_decay

# Error: Training too slow
# Solution: Decrease episodes for testing, check hardware
```

#### Validation Failures
```bash
# Warning: Low exploration coverage
# This is normal for SARSA (conservative), concerning for Q-Learning

# Error: No statistical significance
# Solution: Increase training episodes, check algorithm implementation

# Error: High variance across runs
# Solution: Check for randomness in environment/agent, increase episodes
```

#### Environment Issues
```bash
# Error: Sprite not found warnings
ls -la src/apr/resources/sprites/  # Check sprites exist

# Error: Environment not rendering
# Solution: Check matplotlib backend, install GUI libraries if needed
```

### 📞 Getting Help

1. **Check logs**: Look in `outputs/runs/` for detailed training logs
2. **Verify setup**: Run `python verify_setup.py`
3. **Check documentation**: Refer to [SETUP.md](SETUP.md) for installation issues
4. **Test components**: Use the interactive examples above to isolate issues
5. **Check configurations**: Validate YAML syntax and parameter ranges

---

## 📚 Next Steps

After mastering the basics:

1. **Experiment with hyperparameters** using the examples above
2. **Implement custom environments** with different reward structures  
3. **Add new algorithms** following the agent template
4. **Conduct research studies** with systematic evaluation
5. **Scale to larger environments** and multi-agent scenarios

For detailed implementation examples, explore the notebooks in `notebooks/` - they provide complete workflows for each algorithm with extensive documentation and analysis.

---

**Happy learning!** 🚀