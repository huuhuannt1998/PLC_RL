"""
PLC Error Detection using Pre-trained Code Model
Using Microsoft's CodeBERT (free, open-source model from Hugging Face)

CodeBERT is specifically trained on code and understands programming patterns.
We'll fine-tune it for PLC error detection.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
import re
import numpy as np


class PLCCodeParser:
    """Simple XML parser without rule-based error detection"""
    
    def __init__(self, xml_path: str = None):
        self.xml_path = xml_path
        if xml_path:
            self.tree = ET.parse(xml_path)
            self.root = self.tree.getroot()
        else:
            self.tree = None
            self.root = None
    
    @staticmethod
    def extract_code_from_xml(xml_path: str) -> str:
        """Static method to extract code from XML file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            code_lines = []
            
            # Extract from SCL code (FC/FB blocks) - StructuredText
            for code_block in root.findall('.//{http://www.siemens.com/automation/Openness/SW/NetworkSource/StructuredText/v3}StructuredText'):
                if code_block.text:
                    code_lines.append(code_block.text)
            
            # Also try without namespace
            for code_block in root.findall('.//StructuredText'):
                if code_block.text:
                    code_lines.append(code_block.text)
            
            return '\n'.join(code_lines).strip() if code_lines else ""
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return ""
    
    def extract_code_as_text(self) -> str:
        """Extract PLC code as plain text for CodeBERT"""
        code_lines = []
        
        # Extract from ladder logic (OB blocks)
        for network in self.root.findall('.//SW.Blocks.CompileUnit//NetworkSource'):
            for line in network.findall('.//FlgNet'):
                parts = line.findall('.//Part')
                for part in parts:
                    name = part.get('Name', 'Unknown')
                    negated = part.get('Negated', 'false')
                    uid = part.get('UId', '0')
                    code_lines.append(f"{name} uid={uid} negated={negated}")
        
        # Extract from SCL code (FC/FB blocks) - try both SourceText and StructuredText
        for code_block in self.root.findall('.//{http://www.siemens.com/automation/Openness/SW/NetworkSource/StructuredText/v3}StructuredText'):
            if code_block.text:
                code_lines.append(code_block.text)
        
        # Also try without namespace
        for code_block in self.root.findall('.//StructuredText'):
            if code_block.text:
                code_lines.append(code_block.text)
        
        # Try SourceText (legacy format)
        for code_block in self.root.findall('.//SourceText'):
            if code_block.text:
                code_lines.append(code_block.text)
        
        # Extract from data blocks
        for member in self.root.findall('.//Member'):
            name = member.get('Name', '')
            datatype = member.get('Datatype', '')
            version = member.get('Version', '')
            code_lines.append(f"VAR {name}: {datatype} VERSION={version}")
        
        return '\n'.join(code_lines) if code_lines else "EMPTY_CODE"
    
    def get_block_name(self) -> str:
        """Get the name of this PLC block"""
        for obj in self.root.findall('.//SW.Blocks.CompileUnit'):
            name = obj.get('Name', 'Unknown')
            if name != 'Unknown':
                return name
        return 'Unknown'


class CodeBERTErrorDetector(nn.Module):
    """
    Error detection using Microsoft's CodeBERT
    CodeBERT: Pre-trained model on code from GitHub (125M parameters)
    Free and open-source from Hugging Face
    """
    
    def __init__(self, num_error_types: int = 10):
        super().__init__()
        
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained CodeBERT
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        
        # CodeBERT outputs 768-dimensional embeddings
        hidden_size = 768
        
        # Error detection head
        self.error_classifier = nn.Sequential(
            nn.Linear(hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, num_error_types),
            nn.Sigmoid()  # Multi-label classification (multiple errors possible)
        )
        
        # Error localization head (where in code is the error?)
        self.location_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, code_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns:
            error_probs: [num_error_types] - probability of each error type
            location_probs: [num_tokens] - probability each token is error location
        """
        # Tokenize code
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get CodeBERT embeddings
        outputs = self.codebert(**inputs)
        
        # Sequence output: [batch, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # Pooled output: [batch, hidden_size] (CLS token)
        pooled_output = outputs.pooler_output
        
        # Classify error types
        error_probs = self.error_classifier(pooled_output).squeeze(0)
        
        # Locate errors in code
        location_logits = self.location_head(sequence_output).squeeze(-1).squeeze(0)
        
        return error_probs, location_logits
    
    def detect_error(self, code_text: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Simple binary error detection (has error or not)
        
        Args:
            code_text: Code to analyze
            threshold: Detection threshold
        
        Returns:
            (has_error, confidence)
        """
        if not code_text:
            return False, 0.0
        
        self.eval()
        with torch.no_grad():
            error_probs, _ = self.forward(code_text)
            max_prob = error_probs.max().item()
            has_error = max_prob > threshold
        
        return has_error, max_prob
    
    def train_model(self, training_data: List[Dict], epochs: int = 100, batch_size: int = 8):
        """
        Simple training method for binary classification (buggy vs clean)
        
        Args:
            training_data: List of {'code': str, 'label': int} where label is 0 (clean) or 1 (buggy)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Shuffle data
            import random
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                batch_loss = 0.0
                for sample in batch:
                    code = sample['code']
                    label = float(sample['label'])
                    
                    if not code:
                        continue
                    
                    # Forward pass
                    optimizer.zero_grad()
                    error_probs, _ = self.forward(code)
                    
                    # Use max probability as prediction
                    pred_prob = error_probs.max()
                    
                    # Binary loss
                    target = torch.tensor([label]).to(self.device)
                    loss = criterion(pred_prob.unsqueeze(0), target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item()
                    
                    # Accuracy
                    pred_label = 1 if pred_prob.item() > 0.5 else 0
                    if pred_label == label:
                        correct += 1
                    total += 1
                
                total_loss += batch_loss
            
            # Print progress
            avg_loss = total_loss / (len(training_data) / batch_size)
            accuracy = correct / total * 100 if total > 0 else 0
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print(f"✓ Training complete!")

    def predict(self, xml_path: str, threshold: float = 0.5) -> Dict:
        """
        Predict errors in a PLC XML file
        
        Args:
            xml_path: Path to PLC XML file
            threshold: Confidence threshold for error detection
        
        Returns:
            Dictionary with:
                - has_errors: bool
                - error_types: List of detected error type indices
                - error_locations: List of token indices where errors likely are
                - confidence: List of confidence scores
        """
        parser = PLCCodeParser(xml_path)
        code_text = parser.extract_code_as_text()
        block_name = parser.get_block_name()
        
        self.eval()
        with torch.no_grad():
            error_probs, location_probs = self.forward(code_text)
        
        # Find errors above threshold
        error_types = (error_probs > threshold).nonzero(as_tuple=True)[0].tolist()
        error_confidences = error_probs[error_types].tolist()
        
        # Find error locations (top-k most suspicious tokens)
        k = min(10, len(location_probs))
        top_k_locations = torch.topk(location_probs, k)
        error_locations = top_k_locations.indices.tolist()
        location_confidences = top_k_locations.values.tolist()
        
        return {
            'block_name': block_name,
            'has_errors': len(error_types) > 0,
            'error_types': error_types,
            'error_confidences': error_confidences,
            'error_locations': error_locations,
            'location_confidences': location_confidences,
            'code_preview': code_text[:200] + '...' if len(code_text) > 200 else code_text
        }


class CodeBERTTrainer:
    """Training loop for fine-tuning CodeBERT on PLC error detection"""
    
    def __init__(self, model: CodeBERTErrorDetector, learning_rate: float = 2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.error_criterion = nn.BCELoss()
        self.location_criterion = nn.BCELoss()
    
    def train_step(self, code_text: str, error_labels: torch.Tensor, 
                    location_labels: torch.Tensor) -> float:
        """
        Single training step
        
        Args:
            code_text: PLC code as string
            error_labels: [num_error_types] - binary labels for each error type
            location_labels: [num_tokens] - binary labels for error locations
        
        Returns:
            loss: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        error_probs, location_probs = self.model(code_text)
        
        # Compute losses
        error_loss = self.error_criterion(error_probs, error_labels)
        
        # Match location_probs length with location_labels
        min_len = min(len(location_probs), len(location_labels))
        location_loss = self.location_criterion(
            location_probs[:min_len], 
            location_labels[:min_len]
        )
        
        # Combined loss
        total_loss = error_loss + 0.5 * location_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def train_epoch(self, training_data: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> float:
        """
        Train for one epoch
        
        Args:
            training_data: List of (code_text, error_labels, location_labels)
        
        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        for code_text, error_labels, location_labels in training_data:
            loss = self.train_step(code_text, error_labels, location_labels)
            total_loss += loss
        
        return total_loss / len(training_data)


def create_training_data(xml_files: List[str], error_annotations: Dict[str, List[int]]) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Create training dataset from annotated XML files
    
    Args:
        xml_files: List of paths to PLC XML files
        error_annotations: Dict mapping filename to list of error type indices
    
    Returns:
        List of training examples
    """
    training_data = []
    
    for xml_path in xml_files:
        parser = PLCCodeParser(xml_path)
        code_text = parser.extract_code_as_text()
        
        # Get error labels
        filename = os.path.basename(xml_path)
        error_types = error_annotations.get(filename, [])
        
        # Create binary label vector
        error_labels = torch.zeros(10)
        for error_type in error_types:
            error_labels[error_type] = 1.0
        
        # Create location labels (simplified: uniform distribution if errors exist)
        num_tokens = 512  # Max sequence length
        if len(error_types) > 0:
            # Assume errors are spread throughout (you'd need more precise annotations)
            location_labels = torch.ones(num_tokens) * 0.1
        else:
            location_labels = torch.zeros(num_tokens)
        
        training_data.append((code_text, error_labels, location_labels))
    
    return training_data


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("PLC Error Detection with CodeBERT")
    print("=" * 60)
    print("Model: microsoft/codebert-base (125M parameters)")
    print("Source: Hugging Face (free, open-source)")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading CodeBERT model...")
    model = CodeBERTErrorDetector(num_error_types=10)
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example: Predict on a file
    test_file = os.path.join(os.path.dirname(__file__), "../../Export/Buggy/FC_MotorControl_buggy.xml")
    test_file = os.path.abspath(test_file)
    if os.path.exists(test_file):
        print(f"\nAnalyzing: {test_file}")
        result = model.predict(test_file, threshold=0.3)
        
        print(f"\nBlock: {result['block_name']}")
        print(f"Has Errors: {result['has_errors']}")
        if result['has_errors']:
            print(f"Error Types Detected: {result['error_types']}")
            print(f"Confidence Scores: {[f'{c:.2%}' for c in result['error_confidences']]}")
            print(f"Top Error Locations (token indices): {result['error_locations'][:5]}")
        
        print(f"\nCode Preview:")
        print(result['code_preview'])
    else:
        print(f"\nTest file not found: {test_file}")
    
    print("\n" + "=" * 60)
    print("To fine-tune the model on your data:")
    print("1. Annotate XML files with error types")
    print("2. Create training dataset")
    print("3. Run training loop")
    print("4. Save fine-tuned model")
    print("=" * 60)
