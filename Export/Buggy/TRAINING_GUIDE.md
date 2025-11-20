# Training Dataset Structure

## Directory Layout

```
Export/
├── Main_OB1.xml                    # Correct version
├── FC_MotorControl.xml             # Correct version
├── FB_PID_Controller.xml           # Correct version
├── DB_GlobalData.xml               # Correct version
└── Buggy/
    ├── Main_OB1_buggy.xml          # 2 errors
    ├── FC_MotorControl_buggy.xml   # 2 errors
    ├── FB_PID_Controller_buggy.xml # 4 errors
    ├── DB_GlobalData_buggy.xml     # 1 error
    └── ERROR_CATALOG.md            # This file's details
```

## Error Distribution

### By Severity
- **Critical (Safety/Control):** 6 errors
- **Functional:** 2 errors
- **Configuration:** 1 error

### By Type
- **Logic Inversion:** 3 errors
- **Missing Logic:** 2 errors
- **Boolean Logic:** 1 error
- **Sign Error:** 1 error
- **Incomplete State:** 1 error
- **Invalid Default:** 1 error

## RL Training Phases

### Phase 1: Error Detection
**Input:** XML file with bugs  
**Output:** List of error locations and types  
**Metrics:** Precision, Recall, F1-Score

### Phase 2: Error Classification
**Input:** Error location  
**Output:** Error category and severity  
**Metrics:** Classification accuracy

### Phase 3: Fix Generation
**Input:** Error location and type  
**Output:** Code patch/correction  
**Metrics:** Fix correctness, side effects

### Phase 4: Verification
**Input:** Original code + proposed fix  
**Output:** Validation result  
**Metrics:** Test pass rate, safety compliance

## Recommended Training Order

1. Start with **Main_OB1_buggy.xml** (clearest patterns)
2. Move to **FC_MotorControl_buggy.xml** (logic errors)
3. Progress to **FB_PID_Controller_buggy.xml** (most complex)
4. Finally **DB_GlobalData_buggy.xml** (domain knowledge)

## Feature Extraction for ML

Key features to extract:
- **AST structure** (XML parsing)
- **Control flow** (IF/ELSE branching)
- **Data flow** (variable usage)
- **Domain patterns** (PID, ladder logic conventions)
- **Safety keywords** (Emergency, Stop, Enable)
- **Temporal logic** (ramp up/down symmetry)

## Evaluation Metrics

```python
{
    "detection": {
        "true_positives": 9,  # All 9 errors found
        "false_positives": 0,  # No false alarms
        "false_negatives": 0   # No missed errors
    },
    "classification": {
        "correct_severity": 9,     # Correct severity assigned
        "correct_type": 9          # Correct error type
    },
    "correction": {
        "valid_fixes": 9,          # Syntactically valid
        "correct_fixes": 9,        # Logically correct
        "no_side_effects": 9       # Doesn't break other code
    }
}
```

## Ground Truth Labels

Each error has ground truth:
- **Location:** File, line/element
- **Type:** Error category
- **Severity:** Critical/Functional/Performance
- **Fix:** Exact correction needed
- **Impact:** What goes wrong
- **Detection Method:** How to find it

## Next Steps for RL Implementation

1. **Parse XML to AST** - Create structured representation
2. **Extract Features** - Build feature vectors for ML
3. **Create Dataset** - Pairs of (buggy, correct) versions
4. **Define Reward Function** - Based on error catalog
5. **Train Agent** - Use PPO or similar RL algorithm
6. **Evaluate** - Test on held-out examples
7. **Deploy** - Real-world validation with TIA Portal exports

Good luck with your RL training!
