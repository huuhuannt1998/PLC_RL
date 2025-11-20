"""
Automated Verification Pipeline
Runs NuSMV on all auto-generated models and creates verification report
"""
import os
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Tuple


class VerificationPipeline:
    """Automated formal verification pipeline using NuSMV"""
    
    def __init__(self, nusmv_path: str = r"NuSMV-2.7.1-win64\bin\NuSMV.exe"):
        self.nusmv_path = nusmv_path
        self.results = {}
        
    def run_nusmv(self, smv_file: str) -> Tuple[bool, str, int]:
        """
        Run NuSMV on a model file
        Returns: (all_pass, output, violation_count)
        """
        try:
            result = subprocess.run(
                [self.nusmv_path, smv_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout
            
            # Count violations
            violation_count = output.count("is false")
            all_pass = violation_count == 0
            
            return all_pass, output, violation_count
            
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", -1
        except Exception as e:
            return False, f"ERROR: {str(e)}", -1
    
    def verify_directory(self, smv_dir: str, category: str) -> Dict:
        """Verify all models in a directory"""
        results = {}
        
        smv_files = [f for f in os.listdir(smv_dir) if f.endswith('.smv')]
        
        print(f"\nVerifying {category} models ({len(smv_files)} files)...")
        print("=" * 70)
        
        for smv_file in sorted(smv_files):
            smv_path = os.path.join(smv_dir, smv_file)
            all_pass, output, violation_count = self.run_nusmv(smv_path)
            
            status = "✓ PASS" if all_pass else f"✗ FAIL ({violation_count} violations)"
            print(f"  {smv_file:50s} {status}")
            
            results[smv_file] = {
                'all_pass': all_pass,
                'violation_count': violation_count,
                'output': output
            }
        
        return results
    
    def verify_all(self):
        """Verify all buggy and correct models"""
        print("=" * 70)
        print("FORMAL VERIFICATION PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Verify buggy models
        buggy_results = self.verify_directory(
            'verification/auto_generated/buggy',
            'BUGGY'
        )
        
        # Verify correct models
        correct_results = self.verify_directory(
            'verification/auto_generated/correct',
            'CORRECT'
        )
        
        # Generate statistics
        self.generate_statistics(buggy_results, correct_results)
        
        # Save results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'buggy': buggy_results,
            'correct': correct_results
        }
        
        return self.results
    
    def generate_statistics(self, buggy_results: Dict, correct_results: Dict):
        """Generate verification statistics"""
        print("\n" + "=" * 70)
        print("VERIFICATION STATISTICS")
        print("=" * 70)
        
        # Buggy models statistics
        buggy_total = len(buggy_results)
        buggy_violations = sum(1 for r in buggy_results.values() if not r['all_pass'])
        buggy_pass = buggy_total - buggy_violations
        
        print(f"\nBuggy Models:")
        print(f"  Total: {buggy_total}")
        print(f"  Violations detected: {buggy_violations} ({buggy_violations/buggy_total*100:.1f}%)")
        print(f"  False negatives: {buggy_pass} ({buggy_pass/buggy_total*100:.1f}%)")
        
        # Correct models statistics
        correct_total = len(correct_results)
        correct_pass = sum(1 for r in correct_results.values() if r['all_pass'])
        correct_violations = correct_total - correct_pass
        
        print(f"\nCorrect Models:")
        print(f"  Total: {correct_total}")
        print(f"  All properties passed: {correct_pass} ({correct_pass/correct_total*100:.1f}%)")
        print(f"  False positives: {correct_violations} ({correct_violations/correct_total*100:.1f}%)")
        
        # Overall accuracy
        total = buggy_total + correct_total
        correct_detections = buggy_violations + correct_pass
        accuracy = correct_detections / total * 100
        
        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct_detections}/{total})")
        
        # Detailed violation counts
        total_violations = sum(r['violation_count'] for r in buggy_results.values() if r['violation_count'] > 0)
        avg_violations = total_violations / buggy_violations if buggy_violations > 0 else 0
        
        print(f"\nViolation Analysis:")
        print(f"  Total violations found: {total_violations}")
        print(f"  Average per buggy model: {avg_violations:.2f}")
        
    def save_report(self, output_file: str = "verification_report.json"):
        """Save verification report to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Verification report saved to: {output_file}")
    
    def generate_detailed_report(self, output_file: str = "verification_report.txt"):
        """Generate human-readable verification report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("FORMAL VERIFICATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Buggy models
            f.write("BUGGY MODELS - Expected to have violations\n")
            f.write("-" * 70 + "\n")
            for name, result in sorted(self.results['buggy'].items()):
                status = "✓ DETECTED" if not result['all_pass'] else "✗ MISSED"
                f.write(f"\n{name}: {status}\n")
                if not result['all_pass']:
                    f.write(f"  Violations: {result['violation_count']}\n")
                    # Extract violation details
                    lines = result['output'].split('\n')
                    for line in lines:
                        if 'is false' in line:
                            f.write(f"  - {line.strip()}\n")
            
            # Correct models
            f.write("\n\nCORRECT MODELS - Expected to pass all properties\n")
            f.write("-" * 70 + "\n")
            for name, result in sorted(self.results['correct'].items()):
                status = "✓ VERIFIED" if result['all_pass'] else "✗ FALSE POSITIVE"
                f.write(f"\n{name}: {status}\n")
                if not result['all_pass']:
                    f.write(f"  Unexpected violations: {result['violation_count']}\n")
        
        print(f"✓ Detailed report saved to: {output_file}")


def compare_before_after(buggy_file: str, correct_file: str):
    """Compare verification results before and after fix"""
    pipeline = VerificationPipeline()
    
    print("=" * 70)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 70)
    
    print(f"\nBEFORE (Buggy): {buggy_file}")
    buggy_pass, buggy_output, buggy_violations = pipeline.run_nusmv(
        os.path.join('verification/auto_generated/buggy', buggy_file)
    )
    print(f"  Result: {'✓ PASS' if buggy_pass else f'✗ FAIL ({buggy_violations} violations)'}")
    
    print(f"\nAFTER (Fixed): {correct_file}")
    correct_pass, correct_output, correct_violations = pipeline.run_nusmv(
        os.path.join('verification/auto_generated/correct', correct_file)
    )
    print(f"  Result: {'✓ PASS' if correct_pass else f'✗ FAIL ({correct_violations} violations)'}")
    
    if not buggy_pass and correct_pass:
        print("\n✓ FIX VERIFIED: Bug was present, now fixed!")
    elif not buggy_pass and not correct_pass:
        print("\n⚠ FIX INCOMPLETE: Bug still present")
    elif buggy_pass and correct_pass:
        print("\n? NO BUG DETECTED: Both versions pass")
    
    return buggy_pass, correct_pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Compare specific before/after
        compare_before_after(sys.argv[1], sys.argv[2])
    else:
        # Run full verification pipeline
        pipeline = VerificationPipeline()
        pipeline.verify_all()
        pipeline.save_report()
        pipeline.generate_detailed_report()
