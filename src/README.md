# PLC Error Detection - Source Code

## Folder Structure

### `dqn_solution/`
Deep Q-Network implementation (basic RL approach)
- Fast training (~5 minutes)
- Simple MLP architecture
- Good for quick prototyping
- See `dqn_solution/README.md` for details

### `ppo_gnn_solution/`
PPO + Graph Neural Network implementation (advanced approach)
- Graph-aware architecture
- Better accuracy potential
- Recommended for production
- See `ppo_gnn_solution/README.md` for details

### Root-level files (shared utilities)
- `error_detector.py` - XML parser and rule-based error detection
- `analyzer.py` - Code analysis utilities
- `ml_pipeline.py` - ML training pipeline utilities
- `demo_simple.py` - Quick demo script (rule-based)
- `demo_ml_quick.py` - ML demo script

## Quick Start

### Run DQN Solution
```bash
python -c "from src.dqn_solution.rl_trainer import train_dqn_agent; train_dqn_agent()"
```

### Run PPO+GNN Solution
```bash
python src/ppo_gnn_solution/train_ppo_gnn.py
```

### Run Rule-Based Detector (No ML)
```bash
python src/demo_simple.py
```

## Which Solution to Use?

| Use Case | Recommended Solution |
|----------|---------------------|
| Quick testing | DQN |
| Production deployment | PPO+GNN (after full training) |
| Immediate results | Rule-based (`demo_simple.py`) |
| Research/experimentation | Both (compare results) |

## Model Comparison
See `../MODEL_COMPARISON.md` for detailed comparison of both approaches.
