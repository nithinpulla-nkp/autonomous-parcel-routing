# Autonomous Parcel Routing

> **A comprehensive reinforcement learning framework for warehouse automation**  
> Train intelligent agents to navigate complex warehouse environments, collect packages, and optimize delivery routes using multiple RL algorithms.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**Autonomous Parcel Routing** is a modern reinforcement learning research platform that simulates warehouse automation challenges. Agents learn to navigate gridworld environments with obstacles, collect packages, and deliver them efficiently while maximizing reward and minimizing collisions.

### Key Features

- 🤖 **4 RL Algorithms**: Q-Learning, Double Q-Learning, SARSA, SARSA(λ) with eligibility traces
- 🏭 **Realistic Warehouse Environment**: Package pickup/delivery with obstacles and sprites
- 📊 **Comprehensive Validation**: Statistical testing ensures agents actually learn (not memorize)
- 📓 **Complete Notebooks**: Individual training/validation/testing workflows for each algorithm
- 🎨 **Rich Visualizations**: Training curves, policy analysis, performance comparisons
- 🔬 **Professional Evaluation**: Multi-seed testing, generalization analysis, local optima detection

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd autonomous-parcel-routing

# 2. Create virtual environment
python3 -m venv .apr_venv
source .apr_venv/bin/activate  # Windows: .apr_venv\Scripts\activate

# 3. Install package
pip install -e .

# 4. Verify installation
python verify_setup.py

# 5. Train your first agent
python src/apr/train.py --config cfg/baseline.yaml

# 6. Explore interactive notebooks
cd notebooks && jupyter notebook
```

## 🏗️ Architecture

```
autonomous-parcel-routing/
├── src/apr/                    # Core RL framework
│   ├── agents/                # Algorithm implementations
│   │   ├── q_learning.py      # Classic Q-Learning
│   │   ├── double_q_learning.py # Overestimation bias reduction
│   │   ├── sarsa.py           # On-policy SARSA + SARSA(λ)
│   │   └── registry.py        # Agent factory system
│   ├── env.py                 # Warehouse environment
│   ├── validation.py          # Statistical validation framework
│   ├── evaluate.py            # Comprehensive evaluation tools
│   └── train.py               # Training pipeline
├── notebooks/                 # Interactive exploration
│   ├── 07_q_learning_complete.ipynb      # Q-Learning workflow
│   ├── 08_sarsa_complete.ipynb           # SARSA workflow  
│   ├── 09_sarsa_lambda_complete.ipynb    # SARSA(λ) workflow
│   ├── 05_agent_validation.ipynb         # Multi-agent validation
│   └── 06_agent_comparison.ipynb         # Algorithm comparison
├── cfg/                       # Configuration files
│   ├── baseline.yaml          # Default Q-Learning config
│   ├── double_q_learning.yaml # Double Q-Learning config
│   └── sarsa.yaml            # SARSA config
└── outputs/                   # Training results (auto-created)
    ├── runs/                  # Training metrics & checkpoints
    ├── models/               # Saved agents
    ├── validation_results/   # Validation reports
    └── comparison_results/   # Algorithm comparisons
```

## 🤖 Implemented Algorithms

| Algorithm | Type | Key Features | Notebook |
|-----------|------|--------------|----------|
| **Q-Learning** | Off-policy TD | Fast convergence, optimistic updates | `07_q_learning_complete.ipynb` |
| **Double Q-Learning** | Off-policy TD | Reduces overestimation bias, dual Q-tables | `04_double_q_learning_training.ipynb` |
| **SARSA** | On-policy TD | Conservative, safer exploration | `08_sarsa_complete.ipynb` |
| **SARSA(λ)** | On-policy TD + traces | Faster credit assignment, eligibility traces | `09_sarsa_lambda_complete.ipynb` |

## 🏭 Environment Features

- **Grid-based warehouse** with configurable size and obstacles
- **Package pickup/delivery mechanics** with inventory tracking
- **Rich reward structure**: +25 pickup, +100 delivery, -20 collision
- **Multiple rendering modes**: ASCII, sprites, human-readable
- **Gymnasium-compatible interface** for easy integration

## 📊 Validation & Testing

Our framework includes rigorous validation to ensure agents actually learn:

- **Statistical significance testing** vs random baseline
- **Convergence analysis** with Q-value and policy tracking  
- **Exploration analysis** measuring state coverage and action diversity
- **Generalization testing** across multiple random seeds
- **Local optima detection** through multiple training runs
- **Policy analysis** of learned behavior patterns

## 📈 Performance Results

Recent validation results show all algorithms successfully learn:

| Algorithm | Reward Improvement | Success Rate | State Coverage | Status |
|-----------|-------------------|--------------|----------------|--------|
| Q-Learning | +817.2 vs random | 95%+ | 72.2% | ✅ PASS |
| Double Q-Learning | +989.7 vs random | 98%+ | 63.9% | ✅ PASS |
| SARSA | +983.5 vs random | 90%+ | 33.3% | ⚠️ PASS_WITH_WARNINGS |

## 🎮 Usage Examples

### Command Line Training
```bash
# Train Q-Learning agent
python src/apr/train.py --config cfg/baseline.yaml

# Train Double Q-Learning 
python src/apr/train.py --config cfg/double_q_learning.yaml

# Validate any algorithm
python src/validate_agent.py --algorithm q_learning --episodes 300

# Compare all algorithms
python src/compare_algorithms.py
```

### Interactive Notebooks
```python
# Quick environment test
from apr import WarehouseEnv
env = WarehouseEnv()
env.reset()
env.render(mode='human')  # Visual rendering

# Train and evaluate agent
from apr.agents import create_agent
from apr.validation import RLAgentValidator

agent = create_agent('q_learning', env.observation_space, env.action_space)
validator = RLAgentValidator(agent, env)
results = validator.full_validation(training_episodes=500)
```

### Programmatic Usage
```python
# Complete training workflow
from apr import WarehouseEnv, AgentEvaluator
from apr.agents import create_agent

# Setup
env = WarehouseEnv(seed=42)
agent = create_agent('double_q_learning', env.observation_space, env.action_space)

# Train (see notebooks for complete implementation)
# ... training loop ...

# Evaluate
evaluator = AgentEvaluator(env)
results = evaluator.evaluate_agent(agent, num_episodes=100)
print(f"Performance: {results['aggregated_results']['overall_statistics']}")
```

## 📖 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup and installation guide
- **[USAGE.md](USAGE.md)** - Complete usage guide with examples and customization
- **Notebooks** - Interactive tutorials and complete workflows
- **Source code** - Well-documented implementation in `src/apr/`

## 🔬 Research Applications

This framework is designed for:

- **Algorithm development** - Test new RL algorithms in controlled environments
- **Hyperparameter optimization** - Systematic parameter tuning with statistical validation
- **Benchmark studies** - Fair comparison of multiple algorithms
- **Educational use** - Learn RL concepts through hands-on implementation
- **Warehouse automation research** - Realistic logistics simulation

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **New algorithms** - PPO, A2C, DQN implementations
- **Environment extensions** - Multi-agent scenarios, dynamic obstacles
- **Visualization enhancements** - Interactive policy visualization
- **Performance optimizations** - Vectorized environments, GPU acceleration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on **Gymnasium** for standardized RL environments
- Uses **PyTorch** for neural network components  
- Visualization powered by **Matplotlib** and **Seaborn**
- Statistical analysis with **SciPy**

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Questions**: Check the documentation in `SETUP.md` and `USAGE.md`
- **Examples**: Explore the notebooks in `notebooks/` directory

---

**Ready to get started?** Follow the [setup guide](SETUP.md) or jump into the [usage documentation](USAGE.md)!