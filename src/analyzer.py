import os
import sys
from error_detector import analyze_file, ErrorCorrector, PLCXMLParser
import json
from typing import Dict, List


def generate_report(buggy_dir: str, output_file: str = "error_analysis_report.json"):
    """Generate comprehensive error analysis report"""
    
    results = {
        "summary": {
            "total_files": 0,
            "total_errors": 0,
            "critical_errors": 0,
            "functional_errors": 0,
            "performance_errors": 0
        },
        "files": {}
    }
    
    if not os.path.exists(buggy_dir):
        print(f"Directory not found: {buggy_dir}")
        return
    
    for filename in os.listdir(buggy_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(buggy_dir, filename)
            errors, parser = analyze_file(file_path)
            
            if errors:
                results["summary"]["total_files"] += 1
                results["summary"]["total_errors"] += len(errors)
                
                file_info = {
                    "block_type": parser.get_block_type() if parser else "Unknown",
                    "block_name": parser.get_block_name() if parser else "Unknown",
                    "language": parser.get_programming_language() if parser else "Unknown",
                    "errors": []
                }
                
                for error in errors:
                    results["summary"][f"{error.severity.value}_errors"] += 1
                    
                    error_info = {
                        "type": error.error_type.value,
                        "severity": error.severity.value,
                        "location": error.location,
                        "description": error.description,
                        "fix": error.fix
                    }
                    
                    if error.line_number:
                        error_info["line_number"] = error.line_number
                    
                    file_info["errors"].append(error_info)
                
                results["files"][filename] = file_info
    
    # Save report
    output_path = os.path.join(buggy_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return results


def print_summary(results: Dict):
    """Print summary statistics"""
    summary = results["summary"]
    
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Files Analyzed: {summary['total_files']}")
    print(f"Total Errors Found: {summary['total_errors']}")
    print(f"  - Critical: {summary['critical_errors']}")
    print(f"  - Functional: {summary['functional_errors']}")
    print(f"  - Performance: {summary['performance_errors']}")
    print("=" * 80)
    
    print("\nERRORS BY FILE:")
    print("-" * 80)
    for filename, file_info in results["files"].items():
        print(f"\n{filename} ({file_info['block_type']} - {file_info['language']})")
        print(f"  Errors: {len(file_info['errors'])}")
        for i, error in enumerate(file_info['errors'], 1):
            print(f"    {i}. [{error['severity'].upper()}] {error['type']}")
            print(f"       {error['description']}")


def compare_with_correct_versions(buggy_dir: str, correct_dir: str):
    """Compare buggy versions with correct versions"""
    print("\n" + "=" * 80)
    print("COMPARING BUGGY vs CORRECT VERSIONS")
    print("=" * 80)
    
    buggy_files = [f for f in os.listdir(buggy_dir) if f.endswith('.xml')]
    
    for buggy_file in buggy_files:
        # Try to find corresponding correct file
        correct_file = buggy_file.replace('_buggy', '')
        correct_path = os.path.join(correct_dir, correct_file)
        buggy_path = os.path.join(buggy_dir, buggy_file)
        
        if os.path.exists(correct_path):
            print(f"\n{buggy_file}")
            print("-" * 40)
            
            buggy_errors, _ = analyze_file(buggy_path)
            correct_errors, _ = analyze_file(correct_path)
            
            print(f"Buggy version: {len(buggy_errors)} errors")
            print(f"Correct version: {len(correct_errors)} errors")
            
            if len(buggy_errors) > len(correct_errors):
                print(f"✓ Fixes would resolve {len(buggy_errors) - len(correct_errors)} errors")
        else:
            print(f"\nNo correct version found for {buggy_file}")


def generate_training_dataset(buggy_dir: str, correct_dir: str, output_file: str = "training_data.json"):
    """Generate training dataset in format suitable for ML"""
    from rl_trainer import FeatureExtractor
    
    dataset = {
        "features": [],
        "labels": [],
        "metadata": []
    }
    
    buggy_files = [f for f in os.listdir(buggy_dir) if f.endswith('.xml')]
    
    for buggy_file in buggy_files:
        buggy_path = os.path.join(buggy_dir, buggy_file)
        
        try:
            errors, parser = analyze_file(buggy_path)
            
            if parser:
                extractor = FeatureExtractor()
                features = extractor.extract_features(parser)
                
                dataset["features"].append(features.to_vector().tolist())
                dataset["labels"].append({
                    "has_errors": len(errors) > 0,
                    "num_errors": len(errors),
                    "error_types": [e.error_type.value for e in errors],
                    "severities": [e.severity.value for e in errors]
                })
                dataset["metadata"].append({
                    "filename": buggy_file,
                    "block_type": parser.get_block_type(),
                    "language": parser.get_programming_language()
                })
        except Exception as e:
            print(f"Error processing {buggy_file}: {e}")
    
    # Save dataset
    output_path = os.path.join(buggy_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nTraining dataset saved to: {output_path}")
    print(f"Total samples: {len(dataset['features'])}")


def main():
    buggy_dir = r"C:\Users\hbui11\Desktop\PLC_RL\Export\Buggy"
    correct_dir = r"C:\Users\hbui11\Desktop\PLC_RL\Export"
    
    print("=" * 80)
    print("PLC LOGIC ERROR ANALYSIS TOOL")
    print("=" * 80)
    
    # Generate detailed report
    results = generate_report(buggy_dir)
    
    if results:
        # Print summary
        print_summary(results)
        
        # Compare with correct versions
        compare_with_correct_versions(buggy_dir, correct_dir)
        
        # Generate ML training dataset
        generate_training_dataset(buggy_dir, correct_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the generated files:")
    print("  - error_analysis_report.json (detailed report)")
    print("  - training_data.json (ML training dataset)")
    print("=" * 80)


if __name__ == "__main__":
    main()
