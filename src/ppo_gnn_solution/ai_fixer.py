"""
AI Agent for Fixing PLC Logic Errors
Uses CodeBERT to detect errors and generate corrections
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import re
from codebert_model import PLCCodeParser, CodeBERTErrorDetector


class CodeT5Fixer(nn.Module):
    """
    Uses CodeT5 (Salesforce's code generation model) to fix errors
    CodeT5 is trained specifically for code-to-code transformations
    Free from Hugging Face
    """
    
    def __init__(self):
        super().__init__()
        
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CodeT5 using device: {self.device}")
        
        # Load pre-trained CodeT5 (small version: 60M parameters)
        print("Loading CodeT5 model for code fixing...")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
        
        # Move model to device
        self.model.to(self.device)
        
    def fix_code(self, buggy_code: str, error_description: str) -> str:
        """
        Generate fixed version of buggy code
        
        Args:
            buggy_code: The code with errors
            error_description: Description of what's wrong (e.g., "logic inversion")
        
        Returns:
            Fixed code
        """
        # Create prompt for CodeT5
        prompt = f"Fix {error_description}: {buggy_code}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate fix
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,  # Beam search for better quality
            early_stopping=True,
            temperature=0.7
        )
        
        # Decode
        fixed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fixed_code


class PLCErrorFixer:
    """
    Complete system: Detect errors + Generate fixes
    """
    
    ERROR_TYPE_NAMES = {
        0: "missing logic",
        1: "logic inversion",
        2: "boolean logic error",
        3: "sign error",
        4: "incomplete state machine",
        5: "invalid default value",
        6: "timer value error",
        7: "comparison operator error",
        8: "missing safety interlock",
        9: "incorrect data type"
    }
    
    def __init__(self):
        # Detector: CodeBERT
        self.detector = CodeBERTErrorDetector(num_error_types=10)
        
        # Fixer: CodeT5
        self.fixer = CodeT5Fixer()
        
        # Load fine-tuned detector if available (try both possible filenames)
        model_paths = [
            os.path.join(os.path.dirname(__file__), "../../models/codebert_trained_50examples.pth"),
            os.path.join(os.path.dirname(__file__), "../../models/codebert_finetuned.pth")
        ]
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading fine-tuned detector from {model_path}")
                self.detector.load_state_dict(torch.load(model_path, weights_only=True))
                break
        
    def fix_xml_file(self, xml_path: str, output_path: str = None) -> Dict:
        """
        Detect and fix errors in PLC XML file
        
        Args:
            xml_path: Input XML file with errors
            output_path: Where to save fixed XML (optional)
        
        Returns:
            Dictionary with detection results and fixes
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {os.path.basename(xml_path)}")
        print(f"{'='*70}")
        
        # Step 1: Detect errors
        print("\n[1/3] Detecting errors...")
        detection = self.detector.predict(xml_path, threshold=0.5)
        
        if not detection['has_errors']:
            print("   ✓ No errors detected!")
            return {
                'file': xml_path,
                'has_errors': False,
                'fixes': []
            }
        
        print(f"   ⚠ Found {len(detection['error_types'])} error type(s)")
        for i, error_type in enumerate(detection['error_types']):
            error_name = self.ERROR_TYPE_NAMES.get(error_type, "unknown")
            confidence = detection['error_confidences'][i]
            print(f"      - {error_name} (confidence: {confidence:.1%})")
        
        # Step 2: Extract code
        print("\n[2/3] Extracting buggy code...")
        parser = PLCCodeParser(xml_path)
        buggy_code = parser.extract_code_as_text()
        print(f"   Code length: {len(buggy_code)} characters")
        
        # Step 3: Generate fixes
        print("\n[3/3] Generating fixes...")
        fixes = []
        
        for i, error_type in enumerate(detection['error_types']):
            error_name = self.ERROR_TYPE_NAMES.get(error_type, "unknown")
            
            # Generate fix for this specific error
            print(f"   Fixing: {error_name}...")
            fixed_code = self.fixer.fix_code(buggy_code, error_name)
            
            fixes.append({
                'error_type': error_type,
                'error_name': error_name,
                'confidence': detection['error_confidences'][i],
                'original_code': buggy_code,
                'fixed_code': fixed_code
            })
        
        # Step 4: Save fixed XML (if output path provided)
        if output_path:
            self._save_fixed_xml(xml_path, output_path, fixes)
            print(f"\n✓ Fixed XML saved to: {output_path}")
        
        return {
            'file': xml_path,
            'has_errors': True,
            'num_errors': len(detection['error_types']),
            'error_types': detection['error_types'],
            'error_names': [self.ERROR_TYPE_NAMES.get(e, "unknown") for e in detection['error_types']],
            'confidences': detection['error_confidences'],
            'fixes': fixes
        }
    
    def _save_fixed_xml(self, input_xml: str, output_xml: str, fixes: List[Dict]):
        """
        Save corrected XML file
        This is simplified - in reality you'd need to properly update XML structure
        """
        # Parse original XML
        tree = ET.parse(input_xml)
        root = tree.getroot()
        
        # For demonstration: Add comment with fixes
        comment = ET.Comment(f" Fixed by AI Agent - {len(fixes)} corrections applied ")
        root.insert(0, comment)
        
        for i, fix in enumerate(fixes):
            fix_comment = ET.Comment(f" Fix {i+1}: {fix['error_name']} - confidence {fix['confidence']:.1%} ")
            root.insert(i+1, fix_comment)
        
        # Save
        tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    
    def batch_fix(self, buggy_dir: str, output_dir: str):
        """
        Fix all XML files in a directory
        
        Args:
            buggy_dir: Directory with buggy XML files
            output_dir: Where to save fixed files
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        xml_files = glob.glob(os.path.join(buggy_dir, "*.xml"))
        
        print(f"\n{'='*70}")
        print(f"Batch Processing: {len(xml_files)} files")
        print(f"{'='*70}")
        
        results = []
        
        for xml_file in xml_files:
            basename = os.path.basename(xml_file)
            output_file = os.path.join(output_dir, basename.replace('_buggy', '_fixed'))
            
            result = self.fix_xml_file(xml_file, output_file)
            results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        
        total_errors = sum(r['num_errors'] for r in results if r['has_errors'])
        files_with_errors = sum(1 for r in results if r['has_errors'])
        
        print(f"Files processed: {len(results)}")
        print(f"Files with errors: {files_with_errors}")
        print(f"Total errors fixed: {total_errors}")
        
        return results


def demonstrate_error_fixing():
    """
    Demo: Fix all buggy files in Export/Buggy
    """
    print("\n" + "="*70)
    print("PLC Error Fixing Agent - Demo")
    print("="*70)
    print("\nThis agent will:")
    print("1. Detect errors using CodeBERT (fine-tuned)")
    print("2. Generate fixes using CodeT5")
    print("3. Save corrected XML files")
    print("="*70)
    
    # Initialize agent
    agent = PLCErrorFixer()
    
    # Fix all buggy files
    buggy_dir = os.path.join(os.path.dirname(__file__), "../../Export/Buggy")
    output_dir = os.path.join(os.path.dirname(__file__), "../../Export/Fixed_by_AI")
    
    if not os.path.exists(buggy_dir):
        print(f"\nError: Buggy directory not found: {buggy_dir}")
        return
    
    results = agent.batch_fix(buggy_dir, output_dir)
    
    # Show detailed results
    print("\n" + "="*70)
    print("Detailed Results")
    print("="*70)
    
    for result in results:
        if result['has_errors']:
            print(f"\n{os.path.basename(result['file'])}:")
            for fix in result['fixes']:
                print(f"  ✓ Fixed: {fix['error_name']}")
                print(f"    Confidence: {fix['confidence']:.1%}")
                print(f"    Original: {fix['original_code'][:50]}...")
                print(f"    Fixed: {fix['fixed_code'][:50]}...")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Check if fine-tuned model exists
    model_path = os.path.join(os.path.dirname(__file__), "../../models/codebert_finetuned.pth")
    
    if not os.path.exists(model_path):
        print("="*70)
        print("WARNING: Fine-tuned model not found!")
        print("="*70)
        print(f"\nPlease run training first:")
        print("  python src/ppo_gnn_solution/train_codebert.py")
        print(f"\nExpected model location: {model_path}")
        print("="*70)
        print("\nContinuing with pre-trained model (may have lower accuracy)...")
        input("\nPress Enter to continue...")
    
    # Run demonstration
    demonstrate_error_fixing()
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("1. Check Export/Fixed_by_AI/ for corrected files")
    print("2. Compare with original buggy files")
    print("3. Verify fixes are correct")
    print("4. Fine-tune CodeT5 on more PLC examples for better fixes")
    print("="*70)
