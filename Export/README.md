# TIA Portal Logic Export - Example Files

These are example XML files that represent typical TIA Portal logic exports. They are structured to match the actual format that TIA Openness API would generate.

## Files Included

### 1. Main_OB1.xml
- **Type**: Organization Block (OB1)
- **Language**: Ladder Logic (LAD)
- **Purpose**: Main program cycle block
- **Networks**: 
  - Motor control logic with start button
  - Emergency stop safety logic

### 2. FC_MotorControl.xml
- **Type**: Function (FC)
- **Number**: 10
- **Language**: Structured Control Language (SCL)
- **Purpose**: Motor control with speed ramping
- **Inputs**: Start, Stop, Speed_Setpoint
- **Outputs**: Motor_On, Current_Speed

### 3. FB_PID_Controller.xml
- **Type**: Function Block (FB)
- **Number**: 20
- **Language**: Structured Control Language (SCL)
- **Purpose**: PID controller implementation
- **Features**: 
  - Proportional, Integral, Derivative control
  - Anti-windup protection
  - State retention in static variables

### 4. DB_GlobalData.xml
- **Type**: Global Data Block (DB)
- **Number**: 1
- **Purpose**: Global variables and process data
- **Variables**:
  - Start_Button, Stop_Button, Emergency_Stop (Bool)
  - Motor_Running, Motor_Speed, Speed_Setpoint (Real)
  - Temperature, RunTime_Hours, Alarm data

## XML Structure

Each file follows the TIA Portal XML schema:
- `<Document>` root element
- `<Engineering version="V17" />` version info
- `<DocumentInfo>` metadata
- Block-specific elements (OB, FC, FB, DB)
- `<Interface>` with Input/Output/Static sections
- `<NetworkSource>` with ladder logic (FlgNet) or text (StructuredText)

## Use Cases

These files can be used for:
1. **Testing parsers** without needing TIA Portal access
2. **Developing RL agents** that work with PLC logic
3. **Understanding the XML schema** structure
4. **Creating mock data** for training/testing

## Notes

- Files use TIA Portal V17 format
- All timestamps are set to 2025-11-19
- Logic represents a simple motor control system
- Comments and titles are in English (en-US)
- Data is structured to be realistic and representative

## Next Steps

Once you have TIA Portal permissions:
1. Run the actual export using `TIA_Openness_Demo.exe`
2. Compare the structure with these examples
3. Update parsing logic if there are differences
4. Train your RL model with real production data
