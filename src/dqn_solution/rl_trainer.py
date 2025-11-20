import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from error_detector import LogicError, ErrorSeverity, ErrorType, PLCXMLParser
import xml.etree.ElementTree as ET


@dataclass
class ErrorFeatures:
    """Feature vector for machine learning"""
    # Structural features
    has_negation: float
    num_contacts: float
    num_coils: float
    has_emergency_stop: float
    has_start_stop: float
    
    # Semantic features
    has_or_operator: float
    has_and_operator: float
    has_self_assignment: float
    has_pid_pattern: float
    
    # Code metrics
    code_complexity: float
    num_if_statements: float
    num_variables: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.has_negation, self.num_contacts, self.num_coils,
            self.has_emergency_stop, self.has_start_stop,
            self.has_or_operator, self.has_and_operator,
            self.has_self_assignment, self.has_pid_pattern,
            self.code_complexity, self.num_if_statements, self.num_variables
        ], dtype=np.float32)


class FeatureExtractor:
    """Extract features from PLC XML for ML models"""
    
    @staticmethod
    def extract_features(parser: PLCXMLParser) -> ErrorFeatures:
        """Extract feature vector from parsed XML"""
        
        # Initialize features
        features = ErrorFeatures(
            has_negation=0.0,
            num_contacts=0.0,
            num_coils=0.0,
            has_emergency_stop=0.0,
            has_start_stop=0.0,
            has_or_operator=0.0,
            has_and_operator=0.0,
            has_self_assignment=0.0,
            has_pid_pattern=0.0,
            code_complexity=0.0,
            num_if_statements=0.0,
            num_variables=0.0
        )
        
        # Extract from ladder logic
        networks = parser.extract_ladder_logic()
        for network in networks:
            for part in network['parts']:
                if part['name'] == 'Contact':
                    features.num_contacts += 1.0
                    if part['negated']:
                        features.has_negation += 1.0
                elif part['name'] == 'Coil':
                    features.num_coils += 1.0
            
            # Check variable names
            for var in network['variables']:
                if 'emergency' in var.lower() or 'estop' in var.lower():
                    features.has_emergency_stop = 1.0
                if 'start' in var.lower() or 'stop' in var.lower():
                    features.has_start_stop = 1.0
        
        # Extract from SCL code
        code_blocks = parser.extract_scl_code()
        for code in code_blocks:
            features.num_if_statements += code.count('IF ')
            features.has_or_operator = 1.0 if ' OR ' in code else 0.0
            features.has_and_operator = 1.0 if ' AND ' in code else 0.0
            
            # Check for PID patterns
            if 'Setpoint' in code and 'ProcessValue' in code:
                features.has_pid_pattern = 1.0
            
            # Check for self-assignment
            import re
            if re.search(r'#(\w+)\s*:=\s*#\1', code):
                features.has_self_assignment = 1.0
            
            # Code complexity (simple metric)
            features.code_complexity = len(code.split('\n'))
        
        # Variable count
        features.num_variables = len(parser.extract_data_block_variables())
        
        return features


class ErrorDetectionNetwork(nn.Module):
    """Neural network for error detection"""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_error_types: int = 6):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-task outputs
        self.error_detector = nn.Linear(hidden_dim, 2)  # Binary: has_error or not
        self.error_classifier = nn.Linear(hidden_dim, num_error_types)  # Error type
        self.severity_classifier = nn.Linear(hidden_dim, 3)  # Severity level
        
    def forward(self, x):
        features = self.feature_encoder(x)
        
        has_error = self.error_detector(features)
        error_type = self.error_classifier(features)
        severity = self.severity_classifier(features)
        
        return has_error, error_type, severity


class RLAgent:
    """Reinforcement Learning agent for error detection and correction"""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Experience replay
        self.memory = []
        self.batch_size = 32
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def train_step(self):
        """Train on batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in batch:
            s, a, r, ns, d = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Q-learning update
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class ErrorDetectionEnvironment:
    """RL Environment for error detection tasks"""
    
    def __init__(self, xml_files: List[str], ground_truth: Dict[str, List[LogicError]]):
        self.xml_files = xml_files
        self.ground_truth = ground_truth
        self.current_file_idx = 0
        self.current_parser = None
        self.current_features = None
        self.detected_errors = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to new file"""
        self.current_file_idx = np.random.randint(0, len(self.xml_files))
        file_path = self.xml_files[self.current_file_idx]
        
        self.current_parser = PLCXMLParser(file_path)
        extractor = FeatureExtractor()
        self.current_features = extractor.extract_features(self.current_parser)
        self.detected_errors = []
        
        return self.current_features.to_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action and return (next_state, reward, done)"""
        
        # Action mapping
        actions = {
            0: "check_missing_negation",
            1: "check_logic_inversion",
            2: "check_boolean_error",
            3: "check_sign_error",
            4: "check_incomplete_state",
            5: "check_invalid_default",
            6: "no_error_found",
            7: "flag_critical",
            8: "flag_functional",
            9: "done"
        }
        
        action_name = actions.get(action, "unknown")
        reward = 0.0
        done = False
        
        file_name = self.xml_files[self.current_file_idx].split('\\')[-1]
        true_errors = self.ground_truth.get(file_name, [])
        
        if action == 9:  # Done
            done = True
            # Calculate final reward based on detection accuracy
            true_positives = len([e for e in self.detected_errors if e in true_errors])
            false_positives = len([e for e in self.detected_errors if e not in true_errors])
            false_negatives = len([e for e in true_errors if e not in self.detected_errors])
            
            reward = 50 * true_positives - 10 * false_positives - 20 * false_negatives
        
        elif action <= 5:  # Error detection actions
            # Check if this error type exists
            error_types = [ErrorType.MISSING_LOGIC, ErrorType.LOGIC_INVERSION, 
                          ErrorType.BOOLEAN_LOGIC, ErrorType.SIGN_ERROR,
                          ErrorType.INCOMPLETE_STATE, ErrorType.INVALID_DEFAULT]
            
            target_type = error_types[action]
            matching_errors = [e for e in true_errors if e.error_type == target_type]
            
            if matching_errors:
                reward = 10  # Correct detection
                self.detected_errors.extend(matching_errors)
            else:
                reward = -5  # False positive
        
        elif action in [7, 8]:  # Severity flagging
            if self.detected_errors:
                last_error = self.detected_errors[-1]
                if (action == 7 and last_error.severity == ErrorSeverity.CRITICAL) or \
                   (action == 8 and last_error.severity == ErrorSeverity.FUNCTIONAL):
                    reward = 5  # Correct severity
                else:
                    reward = -2  # Wrong severity
        
        # Next state (same features for now, could be updated)
        next_state = self.current_features.to_vector()
        
        return next_state, reward, done


def train_rl_agent(num_episodes: int = 1000):
    """Train RL agent on error detection"""
    from error_detector import analyze_file
    
    # Load buggy files
    buggy_dir = r"C:\Users\hbui11\Desktop\PLC_RL\Export\Buggy"
    xml_files = [os.path.join(buggy_dir, f) for f in os.listdir(buggy_dir) if f.endswith('.xml')]
    
    # Build ground truth from actual error detection
    print("Building ground truth from error detection...")
    ground_truth = {}
    for file_path in xml_files:
        file_name = os.path.basename(file_path)
        errors, _ = analyze_file(file_path)
        ground_truth[file_name] = errors
        print(f"  {file_name}: {len(errors)} errors")
    
    # Create environment and agent
    env = ErrorDetectionEnvironment(xml_files, ground_truth)
    agent = RLAgent(state_dim=12, action_dim=10)
    
    print("Training RL Agent for Error Detection")
    print("=" * 80)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:  # Max 20 steps per episode
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode}/{num_episodes} - Avg Reward: {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
    
    print("\nTraining Complete!")
    print(f"Final Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    return agent


if __name__ == "__main__":
    print("=" * 80)
    print("PLC LOGIC ERROR DETECTION - RL Training")
    print("=" * 80)
    print()
    
    # Train the agent
    trained_agent = train_rl_agent(num_episodes=500)
    
    # Save model
    model_path = r"C:\Users\hbui11\Desktop\PLC_RL\models\trained_agent.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(trained_agent.q_network.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
