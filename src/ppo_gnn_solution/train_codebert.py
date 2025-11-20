"""
Training script for CodeBERT model on PLC error detection
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from codebert_model import CodeBERTErrorDetector, CodeBERTTrainer, PLCCodeParser
from typing import Dict, List, Tuple
import glob


def create_training_dataset():
    """
    Create training data from our buggy XML files
    
    Error type mapping (from ERROR_CATALOG.md):
    0: MISSING_LOGIC
    1: LOGIC_INVERSION  
    2: BOOLEAN_LOGIC_ERROR
    3: SIGN_ERROR
    4: INCOMPLETE_STATE_MACHINE
    5: INVALID_DEFAULT_VALUE
    6: TIMER_VALUE_ERROR
    7: COMPARISON_OPERATOR_ERROR
    8: MISSING_SAFETY_INTERLOCK
    9: INCORRECT_DATA_TYPE
    """
    
    # Annotate which errors exist in each file
    error_annotations = {
        'Main_OB1_buggy.xml': [0, 8],  # Missing safety logic, missing interlock
        'FC_MotorControl_buggy.xml': [1, 2],  # Logic inversion, boolean error
        'FB_PID_Controller_buggy.xml': [3, 4, 1, 0],  # Sign error, incomplete state, inversion, missing logic
        'DB_GlobalData_buggy.xml': [5],  # Invalid default value
    }
    
    training_data = []
    export_dir = os.path.join(os.path.dirname(__file__), "../../Export/Buggy")
    
    for filename, error_types in error_annotations.items():
        xml_path = os.path.join(export_dir, filename)
        if not os.path.exists(xml_path):
            print(f"Warning: {xml_path} not found, skipping...")
            continue
        
        parser = PLCCodeParser(xml_path)
        code_text = parser.extract_code_as_text()
        
        # Create error label vector
        error_labels = torch.zeros(10)
        for error_type in error_types:
            error_labels[error_type] = 1.0
        
        # Create location labels (simplified: uniform if errors exist)
        max_tokens = 512
        if len(error_types) > 0:
            # Errors could be anywhere, use low uniform probability
            location_labels = torch.ones(max_tokens) * 0.2
        else:
            location_labels = torch.zeros(max_tokens)
        
        training_data.append((code_text, error_labels, location_labels))
        print(f"✓ Loaded {filename}: {len(error_types)} error types")
    
    # Also add correct files (negative examples)
    export_dir_clean = os.path.join(os.path.dirname(__file__), "../../Export")
    clean_files = ['Main_OB1.xml', 'FC_MotorControl.xml', 'FB_PID_Controller.xml', 'DB_GlobalData.xml']
    
    for filename in clean_files:
        xml_path = os.path.join(export_dir_clean, filename)
        if not os.path.exists(xml_path):
            continue
        
        parser = PLCCodeParser(xml_path)
        code_text = parser.extract_code_as_text()
        
        # No errors in clean files
        error_labels = torch.zeros(10)
        location_labels = torch.zeros(512)
        
        training_data.append((code_text, error_labels, location_labels))
        print(f"✓ Loaded {filename}: 0 errors (clean)")
    
    return training_data


def train_codebert_model(num_epochs: int = 10, learning_rate: float = 2e-5):
    """
    Fine-tune CodeBERT for PLC error detection
    """
    print("=" * 70)
    print("Training CodeBERT for PLC Error Detection")
    print("=" * 70)
    
    # Initialize model
    print("\n[1/4] Loading pre-trained CodeBERT...")
    model = CodeBERTErrorDetector(num_error_types=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create training data
    print("\n[2/4] Preparing training dataset...")
    training_data = create_training_dataset()
    print(f"   Dataset size: {len(training_data)} examples")
    
    # Initialize trainer
    print(f"\n[3/4] Initializing trainer (lr={learning_rate})...")
    trainer = CodeBERTTrainer(model, learning_rate=learning_rate)
    
    # Training loop
    print(f"\n[4/4] Training for {num_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(training_data)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    print("-" * 70)
    print("\n✓ Training complete!")
    
    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "../../models/codebert_finetuned.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Model saved to: {save_path}")
    
    return model


def evaluate_model(model: CodeBERTErrorDetector):
    """Evaluate model on test files"""
    print("\n" + "=" * 70)
    print("Evaluating Model")
    print("=" * 70)
    
    test_files = [
        ('Export/Buggy/Main_OB1_buggy.xml', [0, 8]),
        ('Export/Buggy/FC_MotorControl_buggy.xml', [1, 2]),
        ('Export/Buggy/FB_PID_Controller_buggy.xml', [3, 4, 1, 0]),
        ('Export/Buggy/DB_GlobalData_buggy.xml', [5]),
    ]
    
    correct = 0
    total = 0
    
    for filename, true_errors in test_files:
        xml_path = os.path.join(os.path.dirname(__file__), "../..", filename)
        if not os.path.exists(xml_path):
            continue
        
        result = model.predict(xml_path, threshold=0.5)
        predicted_errors = set(result['error_types'])
        true_errors_set = set(true_errors)
        
        # Calculate accuracy
        correct_predictions = len(predicted_errors & true_errors_set)
        total_predictions = len(predicted_errors | true_errors_set)
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
        else:
            accuracy = 1.0 if len(true_errors_set) == 0 else 0.0
        
        print(f"\n{os.path.basename(filename)}:")
        print(f"  True errors: {sorted(true_errors_set)}")
        print(f"  Predicted: {sorted(predicted_errors)}")
        print(f"  Accuracy: {accuracy:.1%}")
        
        if accuracy > 0.5:
            correct += 1
        total += 1
    
    overall = correct / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"Overall: {correct}/{total} files correctly classified ({overall:.1%})")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Train the model
    model = train_codebert_model(num_epochs=20, learning_rate=2e-5)
    
    # Evaluate
    evaluate_model(model)
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Model is now fine-tuned on PLC error patterns")
    print("2. Use model.predict(xml_path) to detect errors in new files")
    print("3. For better accuracy, collect more training examples")
    print("4. Adjust threshold (default 0.5) based on precision/recall needs")
    print("=" * 70)
