# Logic Errors Reference - Training Data for RL Agent

This document catalogs intentional logic errors in the buggy PLC code for RL agent training.

## Overview

The buggy versions contain **9 distinct logic errors** across 3 categories:
1. **Safety Critical** - Could cause equipment damage or safety hazards
2. **Functional** - Causes incorrect behavior
3. **Performance** - Suboptimal but not immediately dangerous

---

## Error Catalog

### File: Main_OB1_buggy.xml

#### Error 1: Missing Stop Button Logic (CRITICAL)
**Location:** Network 1 - Motor Control  
**Severity:** Safety Critical  
**Type:** Missing Logic Element

**Bug:**
```xml
<Part Name="Contact" UId="23" />  <!-- Missing negation -->
```

**Problem:**
- Contact should check for normally-closed stop button
- Missing `<Negated Name="operand" />` attribute
- Motor cannot be stopped once started (latch condition with no reset)

**Symptoms:**
- Motor runs continuously after first start
- Stop button has no effect
- Only emergency stop or power cycle can stop motor

**Fix:**
Add negation to the contact:
```xml
<Part Name="Contact" UId="23">
  <Negated Name="operand" />
</Part>
```

**Detection Strategy for RL:**
- Look for ladder rungs with only set coils and no reset logic
- Check for missing negated contacts in safety circuits
- Verify start/stop button pairing

---

#### Error 2: Inverted Emergency Stop Logic (CRITICAL)
**Location:** Network 2 - Safety Logic  
**Severity:** Safety Critical  
**Type:** Logic Inversion

**Bug:**
```xml
<Part Name="Contact" UId="33" />  <!-- Should be negated -->
```

**Problem:**
- Emergency stop contact is normally-open instead of normally-closed
- Pressing E-stop activates motor instead of stopping it
- Violates safety standards (emergency stops must be fail-safe)

**Symptoms:**
- System unsafe when E-stop is pressed
- Motor starts when E-stop activated
- Inverse of expected behavior

**Fix:**
Add negation to emergency stop contact:
```xml
<Part Name="Contact" UId="33">
  <Negated Name="operand" />
</Part>
```

**Detection Strategy for RL:**
- Emergency stop variables should always use negated contacts
- Safety logic should stop/disable when activated (TRUE state)
- Look for patterns where safety signals enable rather than disable

---

### File: FC_MotorControl_buggy.xml

#### Error 3: Stop Button Acts as Start (CRITICAL)
**Location:** Line 2 - Condition Logic  
**Severity:** Functional Critical  
**Type:** Boolean Logic Error

**Bug:**
```scl
IF #Start OR #Stop THEN  // Wrong operator!
```

**Problem:**
- Uses OR instead of AND NOT
- Pressing Stop button starts the motor
- Should be: `IF #Start AND NOT #Stop THEN`

**Symptoms:**
- Motor starts when stop button pressed
- Cannot have proper stop control
- Contradictory logic

**Fix:**
```scl
IF #Start AND NOT #Stop THEN
```

**Detection Strategy for RL:**
- Start and Stop should be in opposition (AND NOT)
- Never use OR with Start/Stop pairs
- Semantic understanding: Stop should prevent, not enable

---

#### Error 4: Missing Ramp-Down Logic (FUNCTIONAL)
**Location:** ELSE branch, Lines 14-16  
**Severity:** Functional  
**Type:** Missing Implementation

**Bug:**
```scl
ELSE
    #Motor_On := FALSE;
    #Current_Speed := #Current_Speed;  // No-op, does nothing!
END_IF;
```

**Problem:**
- Speed value preserved when motor stops
- Missing deceleration ramp
- Asymmetric behavior (ramps up but not down)

**Symptoms:**
- Speed indicator shows non-zero when motor off
- No controlled deceleration
- Potential mechanical stress from instant stops

**Fix:**
```scl
ELSE
    #Motor_On := FALSE;
    IF #Current_Speed > 0.0 THEN
        #Current_Speed := #Current_Speed - 10.0;
        IF #Current_Speed < 0.0 THEN
            #Current_Speed := 0.0;
        END_IF;
    END_IF;
END_IF;
```

**Detection Strategy for RL:**
- Look for asymmetric IF/ELSE branches
- Self-assignments (`x := x`) are usually bugs
- Ramp-up logic should have matching ramp-down

---

### File: FB_PID_Controller_buggy.xml

#### Error 5: Inverted Error Calculation (CRITICAL)
**Location:** Line 3 - Error Calculation  
**Severity:** Control Critical  
**Type:** Sign Error

**Bug:**
```scl
#Error := #ProcessValue - #Setpoint;  // Wrong order!
```

**Problem:**
- Error sign is inverted
- Controller responds backwards (increases output when should decrease)
- Causes unstable control (positive feedback instead of negative)

**Symptoms:**
- System diverges instead of converges
- Oscillations grow instead of dampen
- Process value moves away from setpoint

**Fix:**
```scl
#Error := #Setpoint - #ProcessValue;
```

**Detection Strategy for RL:**
- PID error should always be: Setpoint - ProcessValue
- Check for control stability patterns
- Look for standard control equation forms

---

#### Error 6: Premature Integral Accumulation (PERFORMANCE)
**Location:** Lines 9-10 - Integral Term  
**Severity:** Performance  
**Type:** Order of Operations

**Bug:**
```scl
// Adds to integral before checking limits
#IntegralSum := #IntegralSum + #Error;
#IntTerm := #Ki * #IntegralSum;
```

**Problem:**
- Integral accumulates before anti-windup check
- Can overshoot limits by one cycle
- Not catastrophic but reduces control quality

**Symptoms:**
- Slight overshoot in control response
- Slower recovery from saturation
- Less precise control

**Fix:**
Move anti-windup check before integral accumulation, or use conditional accumulation

**Detection Strategy for RL:**
- Anti-windup should prevent accumulation, not just limit output
- Check ordering of state updates vs. limit checks

---

#### Error 7: Inverted Anti-Windup Conditions (CRITICAL)
**Location:** Lines 24-30 - Anti-windup  
**Severity:** Control Critical  
**Type:** Logic Inversion

**Bug:**
```scl
IF #ControlOutput < 100.0 THEN  // Should be >
    #ControlOutput := 100.0;
    #IntegralSum := #IntegralSum - #Error;
ELSIF #ControlOutput > 0.0 THEN  // Should be <
    #ControlOutput := 0.0;
    #IntegralSum := #IntegralSum - #Error;
END_IF;
```

**Problem:**
- Comparison operators inverted
- Anti-windup activates in normal range, not at limits
- Output clamped incorrectly
- Integral wind-up prevention doesn't work

**Symptoms:**
- Output always clamped to limits
- No proportional control
- Severe integral windup
- Bang-bang control instead of smooth

**Fix:**
```scl
IF #ControlOutput > 100.0 THEN
    #ControlOutput := 100.0;
    #IntegralSum := #IntegralSum - #Error;
ELSIF #ControlOutput < 0.0 THEN
    #ControlOutput := 0.0;
    #IntegralSum := #IntegralSum - #Error;
END_IF;
```

**Detection Strategy for RL:**
- Anti-windup should trigger OUTSIDE normal range
- Upper limit check: `>` max value
- Lower limit check: `<` min value
- Look for inverted guard conditions

---

#### Error 8: Missing Error Reset on Disable (FUNCTIONAL)
**Location:** ELSE branch, Lines 34-39  
**Severity:** Functional  
**Type:** Incomplete State Reset

**Bug:**
```scl
ELSE
    #ControlOutput := 0.0;
    #IntegralSum := 0.0;
    #LastError := 0.0;
    #Initialized := FALSE;
    // Missing: #Error := 0.0;
END_IF;
```

**Problem:**
- Error output not reset when controller disabled
- Stale error value persists
- Could confuse monitoring systems

**Symptoms:**
- Error display shows old value when controller off
- Misleading operator information
- Potential for incorrect alarm states

**Fix:**
```scl
ELSE
    #ControlOutput := 0.0;
    #IntegralSum := 0.0;
    #LastError := 0.0;
    #Initialized := FALSE;
    #Error := 0.0;
END_IF;
```

**Detection Strategy for RL:**
- All outputs should be explicitly reset
- Check for completeness in reset logic
- Output variables should have assignment in all branches

---

### File: DB_GlobalData_buggy.xml

#### Error 9: Unsafe Speed Setpoint (CONFIGURATION)
**Location:** Speed_Setpoint StartValue  
**Severity:** Safety/Configuration  
**Type:** Invalid Default Value

**Bug:**
```xml
<Member Name="Speed_Setpoint" Datatype="Real">
  <StartValue>3500.0</StartValue>  <!-- Exceeds motor rating! -->
</Member>
```

**Problem:**
- Default speed (3500 RPM) exceeds typical motor rating (max 3000 RPM)
- Could damage equipment on first startup
- No validation or range checking

**Symptoms:**
- Over-speed condition at startup
- Mechanical damage to motor
- Safety risk

**Fix:**
```xml
<StartValue>1500.0</StartValue>  <!-- Safe default value -->
```

**Detection Strategy for RL:**
- Check default values against typical ranges
- Speed setpoints should be conservative
- Look for domain-specific constraints
- Compare with other similar variables

---

## Summary Statistics

| Category | Count | Files Affected |
|----------|-------|----------------|
| Safety Critical | 4 | OB1, FC, FB |
| Functional | 3 | FC, FB, DB |
| Performance | 2 | FB |
| **Total** | **9** | **4** |

## RL Training Approach

### Detection Phase
Train agent to identify:
1. Pattern matching (missing negations, inverted logic)
2. Semantic understanding (Start/Stop relationships)
3. Asymmetric code structures
4. Domain knowledge violations (PID conventions)

### Correction Phase
Train agent to:
1. Propose specific fixes
2. Verify fix maintains other functionality
3. Generate test cases for the fix
4. Explain the reasoning

### Reward Function Design

**Positive Rewards:**
- Correctly identifying error location: +10
- Correct error classification: +5
- Valid fix that compiles: +20
- Fix solves problem without side effects: +50

**Negative Rewards:**
- False positive detection: -5
- Invalid fix (syntax error): -10
- Fix breaks other functionality: -30
- Missed critical safety error: -50

### Difficulty Progression

**Level 1:** Syntax patterns (missing negations)  
**Level 2:** Logic inversions (wrong operators)  
**Level 3:** Missing implementations (incomplete code)  
**Level 4:** Domain knowledge (PID conventions)  
**Level 5:** Multi-error scenarios (multiple interacting bugs)

---

## Test Cases

Each error should have test cases demonstrating:
1. **Normal Operation** (with correct code)
2. **Failure Mode** (with bug present)
3. **Edge Cases** (boundary conditions)
4. **Safety Scenarios** (worst-case outcomes)

See `test_cases/` directory for detailed test scenarios.
