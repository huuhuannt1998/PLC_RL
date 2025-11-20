# Enhanced Buggy Files - Critical Error Summary

## Overview
All files in the Export/Buggy/ folder have been enhanced with multiple sophisticated logical errors that violate safety properties, cause race conditions, boundary violations, and state machine errors.

---

## FC_MotorControl_buggy.xml - 9 Critical Bugs

### Safety-Critical Bugs:
1. **Logic Inversion (CRITICAL)**: Stop button turns motor ON instead of OFF
   - `IF #Stop THEN #Motor_On := TRUE;` should be `FALSE`
   - Violates: G(Stop -> F !Motor_On)

2. **Race Condition**: No mutual exclusion for Start and Stop
   - Both buttons can be TRUE simultaneously
   - Missing conflict resolution logic

3. **No Emergency Stop Override**: System cannot be shut down in emergency
   - Missing: `IF #Emergency_Stop THEN #Motor_On := FALSE; RETURN; END_IF;`

### Speed Control Bugs:
4. **Missing Upper Speed Limit**: Can exceed safe maximum (4000 RPM)
   - No validation of Speed_Setpoint bounds

5. **No Setpoint Validation**: Accepts invalid negative or excessive speeds
   - Missing: `IF #temp_speed > 4000.0 OR #temp_speed < 0.0 THEN ...`

6. **No Maximum Speed Enforcement**: Runtime speed can grow unbounded
   - Missing absolute safety limit check

7. **No Overspeed Protection**: System doesn't trigger emergency stop on overspeed

### Mechanical Safety Bugs:
8. **Immediate Stop Without Ramp-Down**: Causes mechanical stress
   - `#Current_Speed := 0.0;` should ramp down gradually
   - High-speed immediate stop can damage equipment

9. **No State Validation**: Current_Speed can become negative
   - Missing: `IF #Current_Speed < 0.0 THEN #Current_Speed := 0.0; END_IF;`

---

## FB_PID_Controller_buggy.xml - 10 Critical Bugs

### Algorithm Bugs:
1. **Inverted Error Sign (CRITICAL)**: Controller behaves opposite to intent
   - `#Error := #ProcessValue - #Setpoint;` should be `#Setpoint - #ProcessValue`
   - Increases output when should decrease and vice versa

2. **No Input Validation**: Negative gain constants cause instability
   - Missing: `IF #Kp < 0.0 OR #Ki < 0.0 OR #Kd < 0.0 THEN disable`

3. **No Pre-Windup Anti-Windup**: Integral sum grows before checking limits
   - Should check limits BEFORE adding to `#IntegralSum`

4. **Unbounded Integral Accumulation**: IntegralSum can overflow
   - Missing: `IF #IntegralSum > 10000.0 THEN #IntegralSum := 10000.0;`

### Control Bugs:
5. **Inverted Anti-Windup Limits (CRITICAL)**: Comparison operators reversed
   - `IF #ControlOutput < 100.0` should be `> 100.0`
   - `IF #ControlOutput > 0.0` should be `< 0.0`
   - Applies limits at wrong boundaries

6. **Wrong Anti-Windup Correction**: Subtracts incorrect amount
   - Should subtract `(#ControlOutput - limit) / #Ki` not just `#Error`

7. **No Derivative Kick Protection**: Large setpoint changes cause spikes
   - Should use filtered derivative or error rate limiting

8. **Missing Error Reset**: Error not cleared when disabled
   - Missing: `#Error := 0.0;` in ELSE branch

9. **No Temp Variable Cleanup**: Previous calculation values persist
   - Missing: Clear #PropTerm, #IntTerm, #DiffTerm when disabled

10. **No Deadband**: Oscillates on small errors
    - Missing: `IF ABS(#Error) < 0.5 THEN #Error := 0.0;`

---

## DB_GlobalData_buggy.xml - 7 Critical Bugs

### Data Type and Attribute Bugs:
1. **Stop Button Security Flaw**: ExternalWritable=true allows external modification
   - Critical safety inputs should be read-only from network

2. **Emergency Stop Not Retentive**: Loses state on power cycle
   - Emergency stop state should be retained (Retain=true)

3. **Motor Running Not Retentive**: Loses critical state information
   - Should remember motor state across power cycles

### Invalid Data Values:
4. **Negative Speed**: Motor_Speed starts at -100.0 RPM
   - Invalid physical state, should be >= 0.0

5. **Excessive Speed Setpoint**: Speed_Setpoint = 5500.0 exceeds max (4000 RPM)
   - Violates mechanical safety limits

6. **Negative Runtime**: RunTime_Hours = -500
   - Should use UDInt (unsigned) to prevent negative values
   - Time cannot be negative

### Missing Safety Parameters:
7. **Max_Safe_Speed = 0.0**: Critical safety limit set to zero
   - Disables all overspeed protection
   - Should be 4000.0 RPM

8. **Wrong Alarm Code Type**: Alarm_Code is Word (16-bit) should be DWord
   - Limited to 65536 codes, modern systems need more

9. **No Mutual Exclusion Logic**: Alarm_Active and Motor_Running can both be TRUE
   - Safety violation: motor should stop when alarm active

---

## Main_OB1_buggy.xml - 2 Critical Safety Bugs

### Ladder Logic Bugs:
1. **Latching Error**: Missing negation on Start contact
   - Motor stays ON permanently once started
   - Cannot be turned off except by E-stop

2. **Inverted Emergency Stop (CRITICAL)**: E-stop ENABLES motor
   - Contact logic reversed: pressing E-stop starts motor
   - Violates fundamental safety principle
   - Most dangerous bug in entire system

---

## NuSMV Detectable Violations

These bugs violate the following formal safety properties:

1. **Safety Property 1**: `G(Stop -> F !Motor_On)`
   - "Always, if Stop is pressed, Eventually motor turns off"
   - VIOLATED by FC_MotorControl (Stop turns motor ON)

2. **Safety Property 2**: `G(Emergency_Stop -> !Motor_On)`
   - "Always, if Emergency Stop is active, motor is off"
   - VIOLATED by Main_OB1 (E-stop enables motor)

3. **Safety Property 3**: `G(Motor_Speed <= Max_Safe_Speed)`
   - "Always, motor speed is less than or equal to maximum"
   - VIOLATED by FC_MotorControl (no speed limit)

4. **Safety Property 4**: `G(Alarm_Active -> F !Motor_Running)`
   - "Always, if alarm is active, eventually motor stops"
   - VIOLATED by missing interlock logic

5. **Safety Property 5**: `G((Start AND Stop) -> X Motor_On = Motor_On)`
   - "If Start and Stop both pressed, state unchanged"
   - VIOLATED by race condition

---

## Complexity Analysis

| File | Lines of Code | Critical Bugs | Safety Violations | Control Bugs | Data Bugs |
|------|---------------|---------------|-------------------|--------------|-----------|
| FC_MotorControl | ~50 | 9 | 5 | 3 | 1 |
| FB_PID_Controller | ~60 | 10 | 2 | 5 | 3 |
| DB_GlobalData | ~90 | 7 | 3 | 0 | 4 |
| Main_OB1 | ~150 | 2 | 2 | 0 | 0 |
| **TOTAL** | **~350** | **28** | **12** | **8** | **8** |

---

## Expected AI Detection Performance

With 28 sophisticated bugs across 4 files:
- **CodeBERT** should detect 70-85% (20-24 bugs)
- **CodeT5 Fixer** should correct 60-75% (17-21 bugs)
- **NuSMV** should verify 100% of safety property violations (12 safety bugs)

Combined AI + Formal Verification should catch **95%+ of all bugs**.

---

## Testing Strategy

1. **Phase 1 - Detection**: Run CodeBERT on all 4 files
2. **Phase 2 - Verification (Before)**: Convert to NuSMV, expect 12 violations
3. **Phase 3 - Fixing**: Apply CodeT5 fixes
4. **Phase 4 - Verification (After)**: Re-run NuSMV, expect 0-2 violations
5. **Phase 5 - Report**: Generate before/after comparison

---

## Files Ready for Pipeline

✓ FC_MotorControl_buggy.xml (9 bugs)
✓ FB_PID_Controller_buggy.xml (10 bugs)
✓ DB_GlobalData_buggy.xml (7 bugs)
✓ Main_OB1_buggy.xml (2 bugs)

**Total: 28 critical bugs across 4 files**

Run: `python main.py` to execute complete AI + Formal Verification pipeline.
