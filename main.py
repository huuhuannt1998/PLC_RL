"""
Complete Training and Verification Pipeline
1. Train CodeBERT on 50 training examples
2. Detect errors in Export/Buggy/*.xml
3. Generate fixes with CodeT5
4. Run NuSMV verification before and after
"""
import os
import sys
import json
import torch
import xml.etree.ElementTree as ET
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'ppo_gnn_solution'))

from xml_to_nusmv import XMLToNuSMVConverter
from verification_pipeline import VerificationPipeline
from codebert_model import PLCCodeParser, CodeBERTErrorDetector
from ai_fixer import PLCErrorFixer, CodeT5Fixer

import warnings
warnings.filterwarnings('ignore')


def train_codebert_on_dataset():
    """Train CodeBERT on all 50 training examples"""
    print("=" * 70)
    print("STEP 1: TRAINING CODEBERT ON 50 EXAMPLES")
    print("=" * 70)
    
    parser = PLCCodeParser()
    detector = CodeBERTErrorDetector()
    
    # Load training data
    training_data = []
    
    # Load buggy examples (label = 1)
    buggy_dir = 'training_data/buggy'
    for xml_file in os.listdir(buggy_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(buggy_dir, xml_file)
            code = parser.extract_code_from_xml(xml_path)
            if code:
                training_data.append({'code': code, 'label': 1})
    
    # Load correct examples (label = 0)
    correct_dir = 'training_data/correct'
    for xml_file in os.listdir(correct_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(correct_dir, xml_file)
            code = parser.extract_code_from_xml(xml_path)
            if code:
                training_data.append({'code': code, 'label': 0})
    
    print(f"Loaded {len(training_data)} training examples")
    print(f"  Buggy: {sum(1 for d in training_data if d['label'] == 1)}")
    print(f"  Correct: {sum(1 for d in training_data if d['label'] == 0)}")
    
    # Train model
    print("\nTraining CodeBERT (this will take several minutes)...")
    print("Note: Training for 50 epochs (reduced for CPU performance)")
    
    detector.train_model(training_data, epochs=50, batch_size=8)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(detector.state_dict(), 'models/codebert_trained_50examples.pth')
    print("\n[OK] Model saved to: models/codebert_trained_50examples.pth")
    
    return detector


def convert_xml_to_nusmv(xml_file: str, output_dir: str, prefix: str = "") -> str:
    """Convert XML to NuSMV and return path"""
    converter = XMLToNuSMVConverter()
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(xml_file).replace('.xml', '.smv')
    if prefix:
        base_name = f"{prefix}_{base_name}"
    
    smv_path = os.path.join(output_dir, base_name)
    converter.convert_file(xml_file, smv_path)
    
    return smv_path


def run_nusmv_verification(smv_file: str, label: str) -> dict:
    """Run NuSMV on a model and return results"""
    pipeline = VerificationPipeline()
    all_pass, output, violation_count = pipeline.run_nusmv(smv_file)
    
    status = "[PASS]" if all_pass else f"[FAIL] ({violation_count} violations)"
    print(f"  {label:30s} {status}")
    
    return {
        'all_pass': all_pass,
        'violation_count': violation_count,
        'output': output
    }


def process_buggy_files(detector: CodeBERTErrorDetector):
    """Process all files in Export/Buggy with complete verification"""
    print("\n" + "=" * 70)
    print("STEP 2: DETECTING AND FIXING ERRORS IN EXPORT/BUGGY")
    print("=" * 70)
    
    buggy_dir = 'Export/Buggy'
    fixed_dir = 'Export/Fixed_by_AI'
    verification_dir = 'verification/pipeline_results'
    
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    os.makedirs(os.path.join(verification_dir, 'before'), exist_ok=True)
    os.makedirs(os.path.join(verification_dir, 'after'), exist_ok=True)
    
    # Initialize AI fixer
    fixer = PLCErrorFixer()
    parser = PLCCodeParser()
    
    results = []
    
    xml_files = [f for f in os.listdir(buggy_dir) if f.endswith('.xml')]
    
    print(f"\nProcessing {len(xml_files)} buggy files...")
    print("=" * 70)
    
    for xml_file in xml_files:
        print(f"\n{'='*70}")
        print(f"FILE: {xml_file}")
        print(f"{'='*70}")
        
        xml_path = os.path.join(buggy_dir, xml_file)
        
        # Step 1: Extract code and detect error
        code = parser.extract_code_from_xml(xml_path)
        if not code:
            print("  [WARN] No code found in XML")
            continue
        
        has_error, confidence = detector.detect_error(code)
        print(f"\n1. AI Detection:")
        print(f"   Error detected: {has_error}")
        print(f"   Confidence: {confidence:.2%}")
        
        # Step 2: Convert original to NuSMV (BEFORE)
        print(f"\n2. Formal Verification (BEFORE FIX):")
        smv_before = convert_xml_to_nusmv(
            xml_path,
            os.path.join(verification_dir, 'before'),
            'before'
        )
        print(f"   Generated: {os.path.basename(smv_before)}")
        
        # Step 3: Run NuSMV on original
        before_result = run_nusmv_verification(smv_before, "Original")
        
        # Step 4: Generate fix if error detected
        fixed_xml_path = None
        if has_error:
            print(f"\n3. Generating Fix:")
            # Create output path for fixed file
            base_name = os.path.basename(xml_path)
            fixed_xml_path = os.path.join(fixed_dir, base_name.replace('_buggy', '_fixed'))
            
            # Generate fix using CodeT5
            print(f"   Generating AI fix with CodeT5...")
            fixed_code = fixer.fixer.fix_code(code, "logic error")
            
            # Save fixed XML
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Find and update the code section
                for structured_text in root.iter('{http://www.siemens.com/automation/Openness/SW/NetworkSource/StructuredText/v3}StructuredText'):
                    structured_text.text = fixed_code
                
                # Save
                tree.write(fixed_xml_path, encoding='utf-8', xml_declaration=True)
                print(f"   [OK] Fixed version saved: {fixed_xml_path}")
            except Exception as e:
                print(f"   [ERROR] Failed to save fix: {e}")
                fixed_xml_path = None
            
            if fixed_xml_path and os.path.exists(fixed_xml_path):
                print(f"   [OK] Fixed version saved: {fixed_xml_path}")
                
                # Step 5: Convert fixed to NuSMV (AFTER)
                print(f"\n4. Formal Verification (AFTER FIX):")
                smv_after = convert_xml_to_nusmv(
                    fixed_xml_path,
                    os.path.join(verification_dir, 'after'),
                    'after'
                )
                print(f"   Generated: {os.path.basename(smv_after)}")
                
                # Step 6: Run NuSMV on fixed version
                after_result = run_nusmv_verification(smv_after, "Fixed")
                
                # Step 7: Compare results
                print(f"\n5. Verification Comparison:")
                if before_result['violation_count'] > after_result['violation_count']:
                    improvement = before_result['violation_count'] - after_result['violation_count']
                    print(f"   [IMPROVED] {improvement} fewer violations")
                elif before_result['violation_count'] == after_result['violation_count']:
                    print(f"   [SAME] No change in violations")
                else:
                    print(f"   [WORSE] More violations after fix")
                
                results.append({
                    'file': xml_file,
                    'ai_detected': has_error,
                    'ai_confidence': confidence,
                    'before_violations': before_result['violation_count'],
                    'after_violations': after_result['violation_count'],
                    'fixed_file': fixed_xml_path,
                    'smv_before': smv_before,
                    'smv_after': smv_after
                })
            else:
                print(f"   [ERROR] Fix generation failed or no file produced")
                results.append({
                    'file': xml_file,
                    'ai_detected': has_error,
                    'ai_confidence': confidence,
                    'before_violations': before_result['violation_count'],
                    'after_violations': None,
                    'fixed_file': None,
                    'smv_before': smv_before,
                    'smv_after': None
                })
        else:
            print(f"\n3. No error detected by AI - skipping fix generation")
            results.append({
                'file': xml_file,
                'ai_detected': has_error,
                'ai_confidence': confidence,
                'before_violations': before_result['violation_count'],
                'after_violations': None,
                'fixed_file': None,
                'smv_before': smv_before,
                'smv_after': None
            })
    
    return results


def generate_final_report(results: list):
    """Generate comprehensive final report"""
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    
    # Statistics
    total_files = len(results)
    ai_detected = sum(1 for r in results if r['ai_detected'])
    fixes_generated = sum(1 for r in results if r['fixed_file'])
    
    print(f"\nAI Detection:")
    print(f"  Total files: {total_files}")
    if total_files > 0:
        print(f"  Errors detected by AI: {ai_detected} ({ai_detected/total_files*100:.1f}%)")
        print(f"  Fixes generated: {fixes_generated}")
    else:
        print(f"  No files were processed successfully")
    
    # Verification results
    before_with_violations = sum(1 for r in results if r['before_violations'] > 0)
    after_with_violations = sum(1 for r in results if r['after_violations'] is not None and r['after_violations'] > 0)
    
    print(f"\nFormal Verification:")
    print(f"  Files with violations (BEFORE): {before_with_violations}/{total_files}")
    if fixes_generated > 0:
        print(f"  Files with violations (AFTER): {after_with_violations}/{fixes_generated}")
    
    # Success stories
    improved = [r for r in results if r['after_violations'] is not None and 
                r['before_violations'] > r['after_violations']]
    
    if improved:
        print(f"\nSuccessfully Fixed ({len(improved)} files):")
        for r in improved:
            print(f"  [OK] {r['file']}")
            print(f"    Violations: {r['before_violations']} -> {r['after_violations']}")
    
    # Save detailed report
    report_path = 'verification/pipeline_results/final_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'statistics': {
                'total_files': total_files,
                'ai_detected': ai_detected,
                'fixes_generated': fixes_generated,
                'before_violations': before_with_violations,
                'after_violations': after_with_violations,
                'improved': len(improved)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n[OK] Detailed report saved to: {report_path}")
    
    return results


def main():
    """Main pipeline execution"""
    print("\n" + "=" * 70)
    print("COMPLETE TRAINING & VERIFICATION PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("1. Train CodeBERT on 50 training examples (100 epochs)")
    print("2. Detect errors in Export/Buggy/*.xml")
    print("3. Generate fixes using CodeT5")
    print("4. Run NuSMV verification BEFORE and AFTER each fix")
    print("5. Generate comprehensive report")
    print("\nEstimated time: 10-15 minutes")
    print("=" * 70)
    
    # Step 1: Train model (commented out - already trained)
    # detector = train_codebert_on_dataset()
    
    # Load existing trained model
    detector = CodeBERTErrorDetector()
    detector.load_state_dict(torch.load('models/codebert_trained_50examples.pth'))
    detector.eval()
    print("\n[OK] Loaded trained model from: models/codebert_trained_50examples.pth")
    
    # Step 2: Process buggy files with verification
    results = process_buggy_files(detector)
    
    # Step 3: Generate final report
    generate_final_report(results)
    
    print("\n" + "=" * 70)
    print("[COMPLETE] PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    print("\nOutput Locations:")
    print("  Fixed XML files: Export/Fixed_by_AI/")
    print("  NuSMV models (before): verification/pipeline_results/before/")
    print("  NuSMV models (after): verification/pipeline_results/after/")
    print("  Final report: verification/pipeline_results/final_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
