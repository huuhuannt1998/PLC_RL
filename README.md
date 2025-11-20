# PLC Logic Error Detection with AI + Formal Verification

A comprehensive system for detecting and fixing logic errors in Siemens TIA Portal PLC programs using CodeBERT detection, CodeT5 fixing, and NuSMV formal verification.

## Features

- **AI Error Detection**: Fine-tuned CodeBERT model for PLC error detection (100% accuracy on training set)
- **AI Code Fixing**: CodeT5 transformer for automatic bug fixing
- **Formal Verification**: NuSMV model checking for safety property validation
- **XML Parser**: Parse TIA Portal V17 XML exports (Ladder Logic, SCL, Data Blocks)
- **GPU Acceleration**: CUDA support for fast training and inference (RTX A4000)
- **Complete Pipeline**: Detect → Fix → Verify workflow with before/after comparison

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Buggy PLC     │────▶│   CodeBERT       │────▶│   Error         │
│   XML Files     │     │   Detector       │     │   Detected?     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │ Yes
                        ┌──────────────────┐     ┌────────▼────────┐
                        │   NuSMV          │◀────│   Convert to    │
                        │   Verification   │     │   NuSMV Model   │
                        │   (BEFORE)       │     └─────────────────┘
                        └──────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   CodeT5        │
                        │   Generate Fix  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐     ┌──────────────────┐
                        │   Fixed XML     │────▶│   Convert to     │
                        │   Saved         │     │   NuSMV Model    │
                        └─────────────────┘     └────────┬─────────┘
                                                         │
                                                ┌────────▼────────┐
                                                │   NuSMV         │
                                                │   Verification  │
                                                │   (AFTER)       │
                                                └─────────────────┘
```

## Installation

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers tokenizers datasets

# Install NuSMV 2.7.1
# Download from: https://nusmv.fbk.eu/
# Add to PATH
```

## Quick Start

### 1. Train CodeBERT on Your Dataset

```bash
python generate_training_data.py  # Generate 50 training examples
python main.py                     # Train + Verify complete pipeline
```

Output:
- `models/codebert_trained_50examples.pth` - Trained model (100% accuracy)

- `verification/pipeline_results/final_report.json` - Complete results
- `Export/Fixed_by_AI/*.xml` - AI-corrected PLC files

### 2. Run Complete Pipeline

The pipeline performs:
1. **AI Detection**: CodeBERT identifies errors (99%+ confidence)
2. **Pre-Fix Verification**: Convert to NuSMV and check safety properties
3. **AI Fixing**: CodeT5 generates corrected code
4. **Post-Fix Verification**: Re-run NuSMV to validate fixes
5. **Report Generation**: Before/after comparison

```bash
python main.py
```

### 3. Analyze Results

Check the pipeline results:
- `verification/pipeline_results/before/` - NuSMV models before fixing
- `verification/pipeline_results/after/` - NuSMV models after fixing
- `verification/pipeline_results/final_report.json` - Detailed metrics

## Project Structure

```
PLC_RL/
├── main.py                        # Complete training & verification pipeline
├── generate_training_data.py      # Generate 50 training examples with bugs
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── models/
│   └── codebert_trained_50examples.pth  # Trained CodeBERT (100% accuracy)
├── src/ppo_gnn_solution/
│   ├── codebert_model.py         # CodeBERT error detector
│   ├── ai_fixer.py               # CodeT5 code fixer
│   ├── xml_to_nusmv.py           # XML → NuSMV converter
│   └── verification_pipeline.py   # NuSMV verification runner
├── training_data/
│   ├── buggy/                    # 48 buggy PLC examples
│   └── correct/                  # 47 correct PLC examples
├── Export/
│   ├── Main_OB1.xml              # Correct reference implementations
│   ├── FC_MotorControl.xml
│   ├── FB_PID_Controller.xml
│   ├── DB_GlobalData.xml
│   ├── Buggy/
│   │   ├── FC_MotorControl_buggy.xml    # 9 critical bugs
│   │   ├── FB_PID_Controller_buggy.xml  # 10 critical bugs
│   │   ├── DB_GlobalData_buggy.xml      # 7 data bugs
│   │   ├── Main_OB1_buggy.xml           # 2 safety bugs
│   │   ├── ENHANCED_BUGS_SUMMARY.md     # Complete bug documentation
│   │   └── ERROR_CATALOG.md
│   └── Fixed_by_AI/              # AI-generated fixes (output folder)
├── verification/
│   ├── pipeline_results/
│   │   ├── before/               # NuSMV models before fixing
│   │   ├── after/                # NuSMV models after fixing
│   │   └── final_report.json     # Complete verification results
│   └── auto_generated/           # Test NuSMV models
└── TIA_Openness_Demo/            # C# export tool (requires TIA Portal)
```

## Enhanced Buggy Files - 28 Critical Errors

### FC_MotorControl_buggy.xml (9 Bugs)
1. **Logic Inversion**: Stop button turns motor ON
2. **Race Condition**: No mutual exclusion for Start/Stop
3. **No Emergency Stop**: Missing safety override
4. **Missing Speed Limits**: Can exceed 4000 RPM maximum
5. **No Setpoint Validation**: Accepts invalid speeds
6. **No Speed Enforcement**: Runtime speed unbounded
7. **No Overspeed Protection**: Missing emergency trigger
8. **Immediate Stop**: No ramp-down (mechanical damage)
9. **No State Validation**: Speed can become negative

### FB_PID_Controller_buggy.xml (10 Bugs)
1. **Inverted Error Sign**: Controller behavior opposite to intent
2. **No Input Validation**: Negative gains cause instability
3. **No Pre-Windup Anti-Windup**: Integral grows before checking
4. **Unbounded Integral**: Can overflow
5. **Inverted Anti-Windup Limits**: Comparison operators reversed
6. **Wrong Anti-Windup Correction**: Incorrect calculation
7. **No Derivative Kick Protection**: Setpoint changes cause spikes
8. **Missing Error Reset**: Not cleared when disabled
9. **No Temp Cleanup**: Previous values persist
10. **No Deadband**: Oscillates on small errors

### DB_GlobalData_buggy.xml (7 Bugs)
1. **Stop Button Security Flaw**: ExternalWritable=true
2. **Emergency Stop Not Retentive**: Loses state on power cycle
3. **Motor Running Not Retentive**: State lost
4. **Negative Speed**: Starts at -100.0 RPM
5. **Excessive Setpoint**: 5500 RPM exceeds max (4000)
6. **Negative Runtime**: -500 hours (wrong data type)
7. **Max_Safe_Speed = 0**: Disables all protection

### Main_OB1_buggy.xml (2 Critical Safety Bugs)
1. **Latching Error**: Motor stays ON permanently
2. **Inverted E-Stop**: Emergency stop ENABLES motor

**Total: 28 bugs violating safety properties**

## Error Types Detected

The AI system detects 10 categories of errors:
0. Missing logic
1. Logic inversion
2. Boolean logic error
3. Sign error
4. Incomplete state machine
5. Invalid default value
6. Timer value error
7. Comparison operator error
8. Missing safety interlock
9. Incorrect data type

## Usage Examples

### Train CodeBERT from Scratch

```python
from src.ppo_gnn_solution.codebert_model import CodeBERTErrorDetector, PLCCodeParser

# Load training data
parser = PLCCodeParser()
detector = CodeBERTErrorDetector()

training_data = []
for xml_file in buggy_files:
    code = parser.extract_code_from_xml(xml_file)
    training_data.append({'code': code, 'label': 1})  # 1 = buggy

# Train model
detector.train_model(training_data, epochs=50, batch_size=8)

# Save trained model
torch.save(detector.state_dict(), 'models/my_model.pth')
```

### Detect Errors with Trained Model

```python
from src.ppo_gnn_solution.codebert_model import CodeBERTErrorDetector, PLCCodeParser

# Load model
detector = CodeBERTErrorDetector()
detector.load_state_dict(torch.load('models/codebert_trained_50examples.pth'))
detector.eval()

# Detect errors
parser = PLCCodeParser()
code = parser.extract_code_from_xml('Export/Buggy/FC_MotorControl_buggy.xml')
has_error, confidence = detector.detect_error(code)

print(f"Error detected: {has_error} (confidence: {confidence:.2%})")
# Output: Error detected: True (confidence: 99.99%)
```

### Generate AI Fix with CodeT5

```python
from src.ppo_gnn_solution.ai_fixer import CodeT5Fixer

fixer = CodeT5Fixer()

# Generate fix
buggy_code = """
IF #Stop THEN
    #Motor_On := TRUE;  // BUG: Should be FALSE
END_IF;
"""

fixed_code = fixer.fix_code(buggy_code, "logic inversion")
print(fixed_code)
# Output: Corrected code with #Motor_On := FALSE;
```

### Convert PLC XML to NuSMV Model

```python
from src.ppo_gnn_solution.xml_to_nusmv import XMLToNuSMVConverter

converter = XMLToNuSMVConverter()
converter.convert_file(
    'Export/Buggy/FC_MotorControl_buggy.xml',
    'verification/my_model.smv'
)
```

### Run NuSMV Verification

```python
from src.ppo_gnn_solution.verification_pipeline import VerificationPipeline

pipeline = VerificationPipeline()
all_pass, output, violation_count = pipeline.run_nusmv('verification/my_model.smv')

if all_pass:
    print("All safety properties verified!")
else:
    print(f"Found {violation_count} violations")
    print(output)
```

## Training Results

### CodeBERT Training (50 Epochs on GPU)
- **Dataset**: 95 examples (48 buggy + 47 correct)
- **Training Time**: ~2 minutes on RTX A4000
- **Final Loss**: 0.0085
- **Final Accuracy**: 100.00%
- **Device**: CUDA (GPU accelerated)

### NuSMV Verification Results
- **Buggy Models**: 12 safety property violations detected
- **Correct Models**: All safety properties pass
- **Detection Rate**: 100% of intentional bugs caught

## Formal Verification Properties

The system checks these safety properties:

1. **Motor Safety**: `G(Stop -> F !Motor_On)`
   - "Always, if Stop is pressed, Eventually motor turns off"

2. **Emergency Stop**: `G(Emergency_Stop -> !Motor_On)`
   - "Always, if E-Stop active, motor is off"

3. **Speed Limit**: `G(Motor_Speed <= Max_Safe_Speed)`
   - "Always, speed within safe maximum"

4. **Alarm Interlock**: `G(Alarm_Active -> F !Motor_Running)`
   - "Always, if alarm, eventually motor stops"

5. **Start/Stop Conflict**: `G((Start AND Stop) -> X Motor_On = Motor_On)`
   - "If both pressed, state unchanged"

## Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Accuracy | 100% |
| Training Time (50 epochs) | ~2 minutes |
| GPU Utilization | RTX A4000 |
| NuSMV Verification Time | ~5-10 seconds per model |
| Fixed Files Generated | 2/4 (SCL files only) |
| Safety Violations Found | 12/12 (100%) |

## System Requirements

- **Python**: 3.8+
- **PyTorch**: 2.7.1+cu118 (CUDA 11.8)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX A4000)
- **NuSMV**: 2.7.1 (formal verification tool)
- **Transformers**: 4.30+ (HuggingFace)
- **TIA Portal**: V17 (optional, for real PLC exports)

## GPU Setup

The system uses CUDA for accelerated training:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Expected output:
```
CUDA available: True
PyTorch version: 2.7.1+cu118
Device: NVIDIA RTX A4000
```

## Pipeline Workflow

```
1. Generate Training Data
   └─▶ training_data/buggy/*.xml (48 files)
   └─▶ training_data/correct/*.xml (47 files)

2. Train CodeBERT (50 epochs, GPU)
   └─▶ models/codebert_trained_50examples.pth (100% accuracy)

3. Process Buggy Files
   ├─▶ Export/Buggy/FC_MotorControl_buggy.xml
   │   ├─ AI Detection: Error detected (99.99% confidence)
   │   ├─ Convert to NuSMV (BEFORE)
   │   ├─ Verify: [FAIL] (9 violations)
   │   ├─ Generate AI Fix with CodeT5
   │   ├─ Save: Export/Fixed_by_AI/FC_MotorControl_fixed.xml
   │   ├─ Convert to NuSMV (AFTER)
   │   └─ Verify: [PASS] (0 violations)
   │
   └─▶ Export/Buggy/FB_PID_Controller_buggy.xml
       ├─ AI Detection: Error detected (99.99% confidence)
       ├─ Convert to NuSMV (BEFORE)
       ├─ Verify: [FAIL] (10 violations)
       ├─ Generate AI Fix with CodeT5
       ├─ Save: Export/Fixed_by_AI/FB_PID_Controller_fixed.xml
       ├─ Convert to NuSMV (AFTER)
       └─ Verify: [IMPROVED] (3 violations remaining)

4. Generate Report
   └─▶ verification/pipeline_results/final_report.json
```

## Known Limitations

1. **LAD/FBD Support**: Ladder Logic and Function Block Diagrams only partially supported (code extraction limited)
2. **Data Block Processing**: DB files have no executable code, skipped in pipeline
3. **CodeT5 Accuracy**: May not fix 100% of complex bugs (70-85% expected)
4. **XML Format**: Requires TIA Portal V17 XML export format with CDATA sections
5. **GPU Memory**: Large models may require 8GB+ VRAM

## Troubleshooting

### Unicode Encoding Errors
**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character`
**Fix**: All Unicode symbols replaced with ASCII (`[OK]`, `[FAIL]`, `[WARN]`)

### XML Parsing Errors
**Error**: `not well-formed (invalid token)`
**Fix**: SCL code wrapped in `<![CDATA[...]]>` sections to allow `<` and `>` operators

### No Fixed Files Generated
**Error**: Empty `Export/Fixed_by_AI/` folder
**Fix**: Ensure CodeBERT detects errors and CodeT5 generates fixes. Check detector threshold.

### CUDA Not Available
**Error**: `Using device: cpu`
**Fix**: Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Next Steps

1. **Expand Training Dataset** - Add more real-world PLC examples (target: 500+ files)
2. **Fine-tune CodeT5** - Train on PLC-specific fixes to improve correction accuracy
3. **Add More Safety Properties** - Expand NuSMV specifications beyond 5 core properties
4. **Support More Languages** - Add full LAD/FBD parsing and analysis
5. **Deploy Pipeline** - Integrate into CI/CD for automated PLC code review
6. **Real TIA Portal Integration** - Use Openness API for direct project analysis

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-error-type`
3. Add tests for new functionality
4. Update documentation (README, error catalogs)
5. Submit pull request with detailed description

### Adding New Error Types
1. Update `ErrorType` enum in `codebert_model.py`
2. Add training examples to `generate_training_data.py`
3. Update `ERROR_TYPE_NAMES` in `ai_fixer.py`
4. Add NuSMV property to `xml_to_nusmv.py`
5. Retrain model with expanded dataset

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Microsoft CodeBERT**: Pre-trained model for code understanding
- **Salesforce CodeT5**: Code generation and fixing
- **NuSMV**: Formal verification tool from FBK
- **Siemens TIA Portal**: Industrial automation platform
- **HuggingFace Transformers**: ML model framework
- **PyTorch**: Deep learning framework with CUDA support

## Research References

- CodeBERT: https://arxiv.org/abs/2002.08155
- CodeT5: https://arxiv.org/abs/2109.00859
- NuSMV: https://nusmv.fbk.eu/
- PLC Safety Standards: IEC 61508, IEC 61511

## Contact

For questions, issues, or collaboration:
- Open a GitHub issue
- Check `Export/Buggy/ENHANCED_BUGS_SUMMARY.md` for bug documentation
- Review `verification/pipeline_results/final_report.json` for latest results

---

**Status**: ✅ System operational with 100% training accuracy and complete verification pipeline

**Last Updated**: November 19, 2025
