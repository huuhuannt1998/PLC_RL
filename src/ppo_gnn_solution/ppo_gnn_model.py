"""
Advanced RL Models for PLC Error Detection
Option B: PPO (Proximal Policy Optimization) + Graph Neural Network

This provides:
1. More stable training than DQN
2. Better structure understanding via GNN
3. Policy gradient optimization
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from error_detector import PLCXMLParser


# ============================================================================
# Graph Neural Network Components
# ============================================================================

class GraphNode:
    """Node in the PLC control flow graph"""
    def __init__(self, node_id: int, node_type: str, features: np.ndarray):
        self.id = node_id
        self.type = node_type  # 'contact', 'coil', 'if', 'assignment', etc.
        self.features = features
        self.neighbors: List[int] = []


class PLCGraph:
    """Graph representation of PLC code"""
    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[Tuple[int, int]] = []
    
    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node
    
    def add_edge(self, from_id: int, to_id: int):
        self.edges.append((from_id, to_id))
        if from_id in self.nodes:
            self.nodes[from_id].neighbors.append(to_id)
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Convert to adjacency matrix for GNN"""
        n = len(self.nodes)
        adj = torch.zeros(n, n)
        for from_id, to_id in self.edges:
            adj[from_id, to_id] = 1
        return adj
    
    def get_node_features(self) -> torch.Tensor:
        """Get all node features as tensor"""
        features = np.array([node.features for node in self.nodes.values()], dtype=np.float32)
        return torch.from_numpy(features)


class GraphBuilder:
    """Build graph representation from PLC XML"""
    
    @staticmethod
    def build_from_parser(parser: PLCXMLParser) -> PLCGraph:
        """Convert PLC code to graph structure"""
        graph = PLCGraph()
        node_id = 0
        feature_dim = 6  # Fixed feature dimension
        
        block_type = parser.get_block_type()
        
        if block_type == 'OB':
            # Build graph from ladder logic
            networks = parser.extract_ladder_logic()
            for network in networks:
                prev_node_id = None
                for part in network['parts']:
                    # Create node features (fixed 6 dimensions)
                    features = np.array([
                        1.0 if part['name'] == 'Contact' else 0.0,
                        1.0 if part['name'] == 'Coil' else 0.0,
                        1.0 if part.get('negated', False) else 0.0,
                        float(part.get('uid', 0)) / 100.0,  # Normalized
                        0.0,  # Padding
                        0.0   # Padding
                    ], dtype=np.float32)
                    
                    node = GraphNode(node_id, part['name'], features)
                    graph.add_node(node)
                    
                    if prev_node_id is not None:
                        graph.add_edge(prev_node_id, node_id)
                    
                    prev_node_id = node_id
                    node_id += 1
        
        elif block_type in ['FC', 'FB']:
            # Build graph from SCL code
            code_blocks = parser.extract_scl_code()
            for code in code_blocks:
                lines = code.split('\n')
                prev_node_id = None
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('//'):
                        continue
                    
                    # Classify line type
                    node_type = 'unknown'
                    if 'IF' in line:
                        node_type = 'condition'
                    elif ':=' in line:
                        node_type = 'assignment'
                    elif 'THEN' in line or 'ELSE' in line:
                        node_type = 'control'
                    
                    # Create features (fixed 6 dimensions)
                    features = np.array([
                        1.0 if node_type == 'condition' else 0.0,
                        1.0 if node_type == 'assignment' else 0.0,
                        1.0 if 'NOT' in line else 0.0,
                        1.0 if 'AND' in line else 0.0,
                        1.0 if 'OR' in line else 0.0,
                        len(line) / 100.0,  # Normalized length
                    ], dtype=np.float32)
                    
                    node = GraphNode(node_id, node_type, features)
                    graph.add_node(node)
                    
                    if prev_node_id is not None:
                        graph.add_edge(prev_node_id, node_id)
                    
                    prev_node_id = node_id
                    node_id += 1
        
        # Ensure at least one node exists
        if len(graph.nodes) == 0:
            # Create dummy node
            features = np.zeros(feature_dim, dtype=np.float32)
            graph.add_node(GraphNode(0, 'dummy', features))
        
        return graph


class GNNLayer(nn.Module):
    """Graph Neural Network layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_features]
        adj: [num_nodes, num_nodes]
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0))
        
        # Normalize adjacency matrix
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Graph convolution
        out = self.linear(adj_normalized @ x)
        return self.activation(out)


class GraphEncoder(nn.Module):
    """Encode PLC graph into fixed-size representation"""
    
    def __init__(self, node_feature_dim: int = 6, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.gnn1 = GNNLayer(node_feature_dim, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.gnn3 = GNNLayer(hidden_dim, output_dim)
        
        self.readout = nn.Linear(output_dim, output_dim)
    
    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Encode graph to fixed-size vector
        Returns: [output_dim] vector
        """
        h = self.gnn1(node_features, adj)
        h = self.gnn2(h, adj)
        h = self.gnn3(h, adj)
        
        # Global pooling (mean over all nodes)
        graph_embedding = h.mean(dim=0)
        
        return self.readout(graph_embedding)


# ============================================================================
# PPO Agent
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor):
        """Sample action from policy"""
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        action_probs, state_values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    """PPO Agent with Graph Neural Network encoder"""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_feature_dim=6,
            hidden_dim=64,
            output_dim=state_dim
        )
        
        # Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=256)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim=256)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.graph_encoder.parameters(), 'lr': 1e-4},
            {'params': self.policy.parameters(), 'lr': 3e-4}
        ])
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.c1 = 0.5  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        
        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }
    
    def select_action(self, graph: PLCGraph) -> Tuple[int, float]:
        """Select action using policy"""
        # Encode graph to state
        node_features = graph.get_node_features()
        adj = graph.get_adjacency_matrix()
        
        with torch.no_grad():
            state = self.graph_encoder(node_features, adj)
        
        # Get action from policy
        action, logprob = self.policy_old.get_action(state.unsqueeze(0))
        
        # Store for training
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        
        return action, logprob.item()
    
    def store_reward(self, reward: float, is_terminal: bool):
        """Store reward and terminal flag"""
        self.memory['rewards'].append(reward)
        self.memory['is_terminals'].append(is_terminal)
    
    def update(self):
        """PPO update"""
        if len(self.memory['rewards']) == 0:
            return 0.0
        
        # Convert to tensors
        old_states = torch.stack(self.memory['states']).detach()
        old_actions = torch.tensor(self.memory['actions']).detach()
        old_logprobs = torch.stack(self.memory['logprobs']).detach()
        
        # Calculate rewards-to-go
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.memory['rewards']),
            reversed(self.memory['is_terminals'])
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # PPO update for K epochs
        total_loss = 0.0
        for _ in range(self.K_epochs):
            # Evaluate actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Total loss
            loss = -torch.min(surr1, surr2).mean() + \
                   self.c1 * F.mse_loss(state_values, rewards) - \
                   self.c2 * dist_entropy.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Copy new policy to old
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }
        
        return total_loss / self.K_epochs
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'graph_encoder': self.graph_encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy_old'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# ============================================================================
# Training Environment for PPO
# ============================================================================

class PPOEnvironment:
    """Environment for PPO training"""
    
    def __init__(self, xml_files: List[str], ground_truth: Dict):
        self.xml_files = xml_files
        self.ground_truth = ground_truth
        self.current_file_idx = 0
        self.current_graph = None
        self.detected_errors = []
    
    def reset(self) -> PLCGraph:
        """Reset to new file"""
        self.current_file_idx = np.random.randint(0, len(self.xml_files))
        file_path = self.xml_files[self.current_file_idx]
        
        parser = PLCXMLParser(file_path)
        self.current_graph = GraphBuilder.build_from_parser(parser)
        self.detected_errors = []
        
        return self.current_graph
    
    def step(self, action: int) -> Tuple[PLCGraph, float, bool]:
        """Take action"""
        reward = 0.0
        done = False
        
        file_name = os.path.basename(self.xml_files[self.current_file_idx])
        true_errors = self.ground_truth.get(file_name, [])
        
        if action == 9:  # Done
            done = True
            # Calculate final reward
            tp = len([e for e in self.detected_errors if e in true_errors])
            fp = len(self.detected_errors) - tp
            fn = len(true_errors) - tp
            reward = 100 * tp - 20 * fp - 50 * fn
        elif action <= 5:  # Detection action
            # Check if correct
            from error_detector import ErrorType
            error_types = [
                ErrorType.MISSING_LOGIC,
                ErrorType.LOGIC_INVERSION,
                ErrorType.BOOLEAN_LOGIC,
                ErrorType.SIGN_ERROR,
                ErrorType.INCOMPLETE_STATE,
                ErrorType.INVALID_DEFAULT
            ]
            target_type = error_types[action]
            matching = [e for e in true_errors if e.error_type == target_type]
            
            if matching:
                reward = 20
                self.detected_errors.extend(matching)
            else:
                reward = -10
        
        return self.current_graph, reward, done


def main():
    """Test the PPO + GNN agent"""
    print("=" * 80)
    print("PPO + GNN Agent for PLC Error Detection")
    print("=" * 80)
    
    # This would be integrated into the main training pipeline
    print("\nModel Architecture:")
    print("  1. Graph Neural Network (3 layers)")
    print("  2. PPO with Actor-Critic")
    print("  3. Graph encoding -> Policy/Value networks")
    print("\nAdvantages over basic DQN:")
    print("  ✓ Understands code structure")
    print("  ✓ More stable training")
    print("  ✓ Better exploration/exploitation balance")
    print("  ✓ Can handle sequential dependencies")


if __name__ == "__main__":
    main()
