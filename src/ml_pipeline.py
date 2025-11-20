"""
Complete RL/ML pipeline for PLC error detection and correction
Includes training, evaluation, and inference
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import json
from datetime import datetime

from error_detector import analyze_file, ErrorType, ErrorSeverity
from rl_trainer import (
    RLAgent, 
    ErrorDetectionEnvironment, 
    ErrorDetectionNetwork,
    FeatureExtractor
)


class MLPipeline:
    """End-to-end ML pipeline for error detection"""
    
    def __init__(self, data_dir: str, model_dir: str):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results = {
            "training_history": [],
            "evaluation_metrics": {},
            "test_results": []
        }
        
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_dataset(self) -> Tuple[List[str], Dict]:
        """Load and prepare training data"""
        print("\n" + "=" * 80)
        print("STEP 1: PREPARING DATASET")
        print("=" * 80)
        
        xml_files = [
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.xml')
        ]
        
        print(f"\nFound {len(xml_files)} XML files")
        
        # Build ground truth
        ground_truth = {}
        total_errors = 0
        
        for file_path in xml_files:
            file_name = os.path.basename(file_path)
            errors, parser = analyze_file(file_path)
            ground_truth[file_name] = errors
            total_errors += len(errors)
            
            if parser:
                print(f"  ✓ {file_name}: {len(errors)} errors "
                      f"({parser.get_block_type()}/{parser.get_programming_language()})")
        
        print(f"\nTotal errors in dataset: {total_errors}")
        
        return xml_files, ground_truth
    
    def train_model(self, xml_files: List[str], ground_truth: Dict, 
                    num_episodes: int = 500) -> RLAgent:
        """Train RL agent"""
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING RL AGENT")
        print("=" * 80)
        
        env = ErrorDetectionEnvironment(xml_files, ground_truth)
        agent = RLAgent(state_dim=12, action_dim=10)
        
        print(f"\nTraining for {num_episodes} episodes...")
        print(f"State dimension: {agent.state_dim}")
        print(f"Action dimension: {agent.action_dim}")
        print(f"Initial exploration rate: {agent.epsilon:.3f}")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            done = False
            steps = 0
            
            while not done and steps < 20:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                if loss is not None:
                    episode_loss += loss
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(steps, 1))
            
            # Log progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_loss = np.mean(episode_losses[-50:])
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Loss: {avg_loss:7.4f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        self.results["training_history"] = {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "final_epsilon": agent.epsilon
        }
        
        print(f"\n✓ Training complete!")
        print(f"  Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Final epsilon: {agent.epsilon:.3f}")
        
        return agent
    
    def evaluate_model(self, agent: RLAgent, xml_files: List[str], 
                      ground_truth: Dict):
        """Evaluate trained model"""
        print("\n" + "=" * 80)
        print("STEP 3: EVALUATING MODEL")
        print("=" * 80)
        
        env = ErrorDetectionEnvironment(xml_files, ground_truth)
        agent.epsilon = 0.0  # No exploration during eval
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        file_results = []
        
        for file_path in xml_files:
            file_name = os.path.basename(file_path)
            env.current_file_idx = xml_files.index(file_path)
            
            state = env.reset()
            detected = []
            done = False
            steps = 0
            
            while not done and steps < 20:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
                steps += 1
                
                if action <= 5:  # Detection action
                    detected.append(action)
            
            true_errors = ground_truth.get(file_name, [])
            
            # Calculate metrics (simplified)
            tp = min(len(detected), len(true_errors))
            fp = max(0, len(detected) - len(true_errors))
            fn = max(0, len(true_errors) - len(detected))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            file_results.append({
                "file": file_name,
                "detected": len(detected),
                "actual": len(true_errors),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })
            
            status = "✓" if len(detected) == len(true_errors) else "⚠"
            print(f"  {status} {file_name}: Detected {len(detected)}/{len(true_errors)} errors")
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.results["evaluation_metrics"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn
        }
        
        self.results["test_results"] = file_results
        
        print(f"\n{'='*80}")
        print("EVALUATION METRICS")
        print('='*80)
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"True Positives:  {total_tp}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
    
    def save_model(self, agent: RLAgent, name: str = "rl_agent"):
        """Save trained model and results"""
        print("\n" + "=" * 80)
        print("STEP 4: SAVING MODEL")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save PyTorch model
        model_path = os.path.join(self.model_dir, f"{name}_{timestamp}.pth")
        torch.save(agent.q_network.state_dict(), model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        # Save training results
        results_path = os.path.join(self.model_dir, f"results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✓ Results saved: {results_path}")
        
        # Save model configuration
        config = {
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "gamma": agent.gamma,
            "architecture": str(agent.q_network),
            "timestamp": timestamp
        }
        config_path = os.path.join(self.model_dir, f"config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Config saved: {config_path}")
        
        return model_path
    
    def run_full_pipeline(self, num_episodes: int = 500):
        """Execute complete training pipeline"""
        print("\n" + "=" * 80)
        print("PLC ERROR DETECTION - COMPLETE ML PIPELINE")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Prepare data
        xml_files, ground_truth = self.prepare_dataset()
        
        # Step 2: Train model
        agent = self.train_model(xml_files, ground_truth, num_episodes)
        
        # Step 3: Evaluate model
        self.evaluate_model(agent, xml_files, ground_truth)
        
        # Step 4: Save everything
        model_path = self.save_model(agent)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nModel ready for inference: {model_path}")
        
        return agent, model_path


def load_trained_model(model_path: str) -> RLAgent:
    """Load a trained model for inference"""
    agent = RLAgent(state_dim=12, action_dim=10)
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0  # No exploration
    return agent


def inference_on_new_file(model_path: str, xml_path: str):
    """Use trained model to detect errors in new file"""
    print(f"\n{'='*80}")
    print("INFERENCE ON NEW FILE")
    print('='*80)
    
    # Load model
    agent = load_trained_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Parse file
    from error_detector import PLCXMLParser
    parser = PLCXMLParser(xml_path)
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features(parser)
    state = features.to_vector()
    
    print(f"✓ Analyzing: {os.path.basename(xml_path)}")
    print(f"  Block Type: {parser.get_block_type()}")
    print(f"  Language: {parser.get_programming_language()}")
    
    # Run inference
    detected_actions = []
    for _ in range(10):  # Max 10 detection attempts
        action = agent.select_action(state)
        if action == 9:  # Done
            break
        if action <= 5:  # Detection action
            detected_actions.append(action)
    
    # Map actions to error types
    error_types = [
        ErrorType.MISSING_LOGIC,
        ErrorType.LOGIC_INVERSION,
        ErrorType.BOOLEAN_LOGIC,
        ErrorType.SIGN_ERROR,
        ErrorType.INCOMPLETE_STATE,
        ErrorType.INVALID_DEFAULT
    ]
    
    print(f"\nDetected {len(detected_actions)} potential error(s):")
    for i, action in enumerate(detected_actions, 1):
        if action < len(error_types):
            print(f"  {i}. {error_types[action].value}")


def main():
    # Configuration
    buggy_dir = r"C:\Users\hbui11\Desktop\PLC_RL\Export\Buggy"
    model_dir = r"C:\Users\hbui11\Desktop\PLC_RL\models"
    
    # Run complete pipeline
    pipeline = MLPipeline(buggy_dir, model_dir)
    agent, model_path = pipeline.run_full_pipeline(num_episodes=500)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Model is trained and saved")
    print("2. Use it for inference on new files:")
    print(f"   >>> from ml_pipeline import inference_on_new_file")
    print(f"   >>> inference_on_new_file('{model_path}', 'your_file.xml')")
    print("\n3. Or retrain with more data")
    print("4. Or fine-tune on specific error types")


if __name__ == "__main__":
    main()
