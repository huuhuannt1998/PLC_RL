# PPO+GNN Solution for PLC Error Detection

## Overview
Proximal Policy Optimization (PPO) with Graph Neural Networks (GNN) for detecting errors in PLC code.

## Files
- `ppo_gnn_model.py` - PPO agent with GNN architecture
- `train_ppo_gnn.py` - Training script
- `error_detector.py` - Rule-based error detection parser

## Architecture
- **Graph Encoder**: 3-layer GNN (6 → 64 → 64 → 128)
- **Actor Network**: 128 → 256 → 128 → 10 (policy)
- **Critic Network**: 128 → 256 → 128 → 1 (value)
- **Algorithm**: PPO with clipped objective and policy gradient

## Performance
- **Training Episodes**: 200 (needs 500-1000 for convergence)
- **Current Reward**: 90.60
- **Training Time**: 10-20 minutes
- **Model**: `../../models/ppo_gnn_20251119_165549.pth`

## How to Use

### Train the Model
```bash
cd c:\Users\hbui11\Desktop\PLC_RL
python src/ppo_gnn_solution/train_ppo_gnn.py
```

### Load Trained Model
```python
from src.ppo_gnn_solution.ppo_gnn_model import PPOAgent
import torch

agent = PPOAgent(state_dim=128, action_dim=10)
agent.load('models/ppo_gnn_20251119_165549.pth')
```

### Train Longer (Recommended)
Edit `train_ppo_gnn.py`:
```python
# Change num_episodes to 1000
train_ppo_gnn(num_episodes=1000, update_interval=50)
```

## Pros
✓ Understands code structure (graph-based)
✓ Stable training (PPO algorithm)
✓ Better exploration/exploitation
✓ Scalable to complex patterns

## Cons
✗ Longer training time
✗ More complex to tune
✗ Needs more training data

## Recommended Settings
- **Episodes**: 500-1000
- **Update Interval**: 50
- **Learning Rate (Graph)**: 1e-4
- **Learning Rate (Policy)**: 3e-4
- **PPO Clip**: 0.2
- **Entropy Coefficient**: 0.01
