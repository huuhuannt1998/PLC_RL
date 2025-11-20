# DQN Solution for PLC Error Detection

## Overview
Deep Q-Network (DQN) implementation for detecting errors in PLC code.

## Files
- `rl_trainer.py` - Main DQN agent and training logic
- `error_detector.py` - Rule-based error detection parser

## Architecture
- **Input**: 12-dimensional feature vector
- **Network**: MLP with 2 hidden layers (128 neurons each)
- **Output**: 10 Q-values (one per action)
- **Algorithm**: Q-Learning with ε-greedy exploration and experience replay

## Performance
- **Precision**: ~10%
- **Recall**: ~100%
- **Training Time**: 2-5 minutes (50 episodes)
- **Model**: `../../models/rl_agent_20251119_165142.pth`

## How to Use

### Train the Model
```bash
python -c "from src.dqn_solution.rl_trainer import train_dqn_agent; train_dqn_agent()"
```

### Load Trained Model
```python
from src.dqn_solution.rl_trainer import RLAgent
import torch

agent = RLAgent(state_dim=12, action_dim=10)
agent.q_network.load_state_dict(torch.load('models/rl_agent_20251119_165142.pth'))
agent.q_network.eval()
```

## Pros
✓ Fast training
✓ Simple to understand
✓ High recall (finds all errors)
✓ Good for prototyping

## Cons
✗ High false positive rate
✗ Doesn't understand code structure
✗ Limited to feature vectors
