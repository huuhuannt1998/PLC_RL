import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    CRITICAL = "critical"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"


class ErrorType(Enum):
    LOGIC_INVERSION = "logic_inversion"
    MISSING_LOGIC = "missing_logic"
    BOOLEAN_LOGIC = "boolean_logic"
    SIGN_ERROR = "sign_error"
    INCOMPLETE_STATE = "incomplete_state"
    INVALID_DEFAULT = "invalid_default"


@dataclass
class LogicError:
    file_name: str
    location: str
    error_type: ErrorType
    severity: ErrorSeverity
    description: str
    fix: str
    line_number: Optional[int] = None


class PLCXMLParser:
    """Parse PLC XML files and extract logic structures"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.namespaces = {
            'flg': 'http://www.siemens.com/automation/Openness/SW/NetworkSource/FlgNet/v4',
            'scl': 'http://www.siemens.com/automation/Openness/SW/NetworkSource/StructuredText/v3',
            'intf': 'http://www.siemens.com/automation/Openness/SW/Interface/v5'
        }
    
    def get_block_type(self) -> str:
        """Identify the block type (OB, FC, FB, DB)"""
        if self.root.find('.//SW.Blocks.OB') is not None:
            return 'OB'
        elif self.root.find('.//SW.Blocks.FC') is not None:
            return 'FC'
        elif self.root.find('.//SW.Blocks.FB') is not None:
            return 'FB'
        elif self.root.find('.//SW.Blocks.GlobalDB') is not None:
            return 'DB'
        return 'Unknown'
    
    def get_block_name(self) -> str:
        """Extract block name"""
        name_elem = self.root.find('.//Name')
        return name_elem.text if name_elem is not None else 'Unknown'
    
    def get_programming_language(self) -> str:
        """Get the programming language"""
        lang_elem = self.root.find('.//ProgrammingLanguage')
        return lang_elem.text if lang_elem is not None else 'Unknown'
    
    def extract_ladder_logic(self) -> List[Dict]:
        """Extract ladder logic networks (LAD/FBD)"""
        networks = []
        for network in self.root.findall('.//NetworkSource/flg:FlgNet', self.namespaces):
            parts = []
            for part in network.findall('.//flg:Part', self.namespaces):
                part_data = {
                    'name': part.get('Name'),
                    'uid': part.get('UId'),
                    'negated': part.find('.//flg:Negated', self.namespaces) is not None
                }
                parts.append(part_data)
            
            contacts = []
            for contact in network.findall('.//flg:Access', self.namespaces):
                symbol = contact.find('.//flg:Symbol/flg:Component', self.namespaces)
                if symbol is not None:
                    contacts.append(symbol.get('Name'))
            
            networks.append({
                'parts': parts,
                'variables': contacts
            })
        return networks
    
    def extract_scl_code(self) -> List[str]:
        """Extract SCL/Structured Text code"""
        code_blocks = []
        for code in self.root.findall('.//NetworkSource/scl:StructuredText', self.namespaces):
            if code.text:
                code_blocks.append(code.text.strip())
        return code_blocks
    
    def extract_data_block_variables(self) -> List[Dict]:
        """Extract variables from data blocks"""
        variables = []
        for member in self.root.findall('.//intf:Member', self.namespaces):
            var_data = {
                'name': member.get('Name'),
                'datatype': member.get('Datatype'),
                'start_value': None
            }
            start_val = member.find('.//intf:StartValue', self.namespaces)
            if start_val is not None:
                var_data['start_value'] = start_val.text
            variables.append(var_data)
        return variables
    
    def get_comments(self) -> List[str]:
        """Extract all comments from the code"""
        comments = []
        for comment in self.root.findall('.//MultilingualTextItem'):
            text_elem = comment.find('.//Text')
            if text_elem is not None and text_elem.text:
                comments.append(text_elem.text)
        return comments


class ErrorDetector:
    """Detect logic errors in PLC code"""
    
    def __init__(self):
        self.errors: List[LogicError] = []
    
    def detect_all_errors(self, parser: PLCXMLParser, file_name: str) -> List[LogicError]:
        """Run all error detection methods"""
        self.errors = []
        
        block_type = parser.get_block_type()
        
        if block_type == 'OB':
            self._detect_ladder_errors(parser, file_name)
        elif block_type in ['FC', 'FB']:
            self._detect_scl_errors(parser, file_name)
        elif block_type == 'DB':
            self._detect_data_block_errors(parser, file_name)
        
        return self.errors
    
    def _detect_ladder_errors(self, parser: PLCXMLParser, file_name: str):
        """Detect errors in ladder logic"""
        networks = parser.extract_ladder_logic()
        
        for idx, network in enumerate(networks):
            # Check for missing negations on contacts
            parts = network['parts']
            variables = network['variables']
            
            # Check for start/stop button patterns without proper negation
            if any('Start' in v for v in variables):
                contacts = [p for p in parts if p['name'] == 'Contact']
                if contacts and not any(c['negated'] for c in contacts):
                    # Check if this is a set coil with no reset logic
                    coils = [p for p in parts if p['name'] == 'Coil']
                    if coils and not any(c['negated'] for c in coils):
                        self.errors.append(LogicError(
                            file_name=file_name,
                            location=f"Network {idx + 1}",
                            error_type=ErrorType.MISSING_LOGIC,
                            severity=ErrorSeverity.CRITICAL,
                            description="Missing stop button logic - motor cannot be stopped once started",
                            fix="Add negated contact for stop button or add reset coil"
                        ))
            
            # Check for emergency stop patterns
            if any('Emergency' in v or 'E_Stop' in v or 'ESTOP' in v for v in variables):
                contacts = [p for p in parts if p['name'] == 'Contact']
                # Emergency stops should typically be negated (normally closed)
                if contacts and not any(c['negated'] for c in contacts):
                    self.errors.append(LogicError(
                        file_name=file_name,
                        location=f"Network {idx + 1}",
                        error_type=ErrorType.LOGIC_INVERSION,
                        severity=ErrorSeverity.CRITICAL,
                        description="Emergency stop logic inverted - pressing E-stop may enable instead of disable",
                        fix="Add negation to emergency stop contact (normally closed logic)"
                    ))
    
    def _detect_scl_errors(self, parser: PLCXMLParser, file_name: str):
        """Detect errors in SCL/Structured Text"""
        code_blocks = parser.extract_scl_code()
        
        for code in code_blocks:
            lines = code.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Check for Start OR Stop pattern (should be AND NOT)
                if re.search(r'#Start\s+OR\s+#Stop', line, re.IGNORECASE):
                    self.errors.append(LogicError(
                        file_name=file_name,
                        location=f"Line {line_num}",
                        error_type=ErrorType.BOOLEAN_LOGIC,
                        severity=ErrorSeverity.CRITICAL,
                        description="Stop button acts as Start - wrong boolean operator",
                        fix="Change 'OR' to 'AND NOT' (should be: #Start AND NOT #Stop)",
                        line_number=line_num
                    ))
                
                # Check for inverted PID error calculation
                if re.search(r'#Error\s*:=\s*#ProcessValue\s*-\s*#Setpoint', line, re.IGNORECASE):
                    self.errors.append(LogicError(
                        file_name=file_name,
                        location=f"Line {line_num}",
                        error_type=ErrorType.SIGN_ERROR,
                        severity=ErrorSeverity.CRITICAL,
                        description="PID error calculation inverted - controller will respond backwards",
                        fix="Change to: #Error := #Setpoint - #ProcessValue",
                        line_number=line_num
                    ))
                
                # Check for self-assignment (often a bug)
                match = re.search(r'#(\w+)\s*:=\s*#\1\s*;', line)
                if match:
                    self.errors.append(LogicError(
                        file_name=file_name,
                        location=f"Line {line_num}",
                        error_type=ErrorType.MISSING_LOGIC,
                        severity=ErrorSeverity.FUNCTIONAL,
                        description=f"Self-assignment detected: #{match.group(1)} := #{match.group(1)} - likely missing implementation",
                        fix="Implement the intended logic (e.g., ramp-down calculation)",
                        line_number=line_num
                    ))
                
                # Check for inverted anti-windup logic
                if 'ControlOutput' in line and '<' in line and '100.0' in line:
                    if ':= 100.0' in lines[line_num] if line_num < len(lines) else '':
                        self.errors.append(LogicError(
                            file_name=file_name,
                            location=f"Line {line_num}",
                            error_type=ErrorType.LOGIC_INVERSION,
                            severity=ErrorSeverity.CRITICAL,
                            description="Anti-windup condition inverted - should check '>' not '<' for upper limit",
                            fix="Change IF #ControlOutput < 100.0 to IF #ControlOutput > 100.0",
                            line_number=line_num
                        ))
            
            # Check for missing variable resets in ELSE block
            if 'ELSE' in code:
                else_section = code.split('ELSE')[1].split('END_IF')[0]
                if ':= 0.0' in else_section or ':= FALSE' in else_section:
                    # Check if Error variable is reset
                    if '#Error' in code and '#Error := 0.0' not in else_section:
                        self.errors.append(LogicError(
                            file_name=file_name,
                            location="ELSE block",
                            error_type=ErrorType.INCOMPLETE_STATE,
                            severity=ErrorSeverity.FUNCTIONAL,
                            description="Missing error reset in ELSE block - output not fully reset",
                            fix="Add #Error := 0.0; to the ELSE block"
                        ))
    
    def _detect_data_block_errors(self, parser: PLCXMLParser, file_name: str):
        """Detect errors in data blocks"""
        variables = parser.extract_data_block_variables()
        
        for var in variables:
            # Check for unsafe speed setpoints
            if 'speed' in var['name'].lower() and 'setpoint' in var['name'].lower():
                if var['start_value'] and var['datatype'] == 'Real':
                    try:
                        speed = float(var['start_value'])
                        # Typical motor max speed is 3000 RPM
                        if speed > 3000.0:
                            self.errors.append(LogicError(
                                file_name=file_name,
                                location=f"Variable {var['name']}",
                                error_type=ErrorType.INVALID_DEFAULT,
                                severity=ErrorSeverity.CRITICAL,
                                description=f"Unsafe default speed setpoint ({speed} RPM) exceeds typical motor rating",
                                fix="Change to safe default value (e.g., 1500.0 RPM)"
                            ))
                    except ValueError:
                        pass


class ErrorCorrector:
    """Generate fixes for detected errors"""
    
    @staticmethod
    def generate_fix(error: LogicError, parser: PLCXMLParser) -> Optional[str]:
        """Generate a code fix for the detected error"""
        
        if error.error_type == ErrorType.MISSING_LOGIC:
            if 'stop button' in error.description.lower():
                return '''Add a negated contact for the stop button:
<Part Name="Contact" UId="XX">
  <Negated Name="operand" />
</Part>'''
        
        elif error.error_type == ErrorType.LOGIC_INVERSION:
            if 'emergency stop' in error.description.lower():
                return '''Add negation to the emergency stop contact:
<Part Name="Contact" UId="XX">
  <Negated Name="operand" />
</Part>'''
            elif 'anti-windup' in error.description.lower():
                return 'IF #ControlOutput > 100.0 THEN'
        
        elif error.error_type == ErrorType.BOOLEAN_LOGIC:
            return 'IF #Start AND NOT #Stop THEN'
        
        elif error.error_type == ErrorType.SIGN_ERROR:
            return '#Error := #Setpoint - #ProcessValue;'
        
        elif error.error_type == ErrorType.INCOMPLETE_STATE:
            return '#Error := 0.0;'
        
        elif error.error_type == ErrorType.INVALID_DEFAULT:
            return '<StartValue>1500.0</StartValue>'
        
        return error.fix


def analyze_file(file_path: str) -> Tuple[List[LogicError], PLCXMLParser]:
    """Analyze a single PLC file for errors"""
    try:
        parser = PLCXMLParser(file_path)
        detector = ErrorDetector()
        errors = detector.detect_all_errors(parser, file_path.split('\\')[-1])
        return errors, parser
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return [], None


def main():
    """Main analysis function"""
    import os
    
    # Analyze buggy files
    buggy_dir = r"C:\Users\hbui11\Desktop\PLC_RL\Export\Buggy"
    
    if not os.path.exists(buggy_dir):
        print(f"Directory not found: {buggy_dir}")
        return
    
    print("=" * 80)
    print("PLC Logic Error Detector")
    print("=" * 80)
    print()
    
    total_errors = 0
    
    for filename in os.listdir(buggy_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(buggy_dir, filename)
            print(f"\nAnalyzing: {filename}")
            print("-" * 80)
            
            errors, parser = analyze_file(file_path)
            
            if parser:
                print(f"Block Type: {parser.get_block_type()}")
                print(f"Block Name: {parser.get_block_name()}")
                print(f"Language: {parser.get_programming_language()}")
                print()
            
            if errors:
                print(f"Found {len(errors)} error(s):\n")
                for i, error in enumerate(errors, 1):
                    print(f"{i}. [{error.severity.value.upper()}] {error.error_type.value}")
                    print(f"   Location: {error.location}")
                    print(f"   Issue: {error.description}")
                    print(f"   Fix: {error.fix}")
                    print()
                total_errors += len(errors)
            else:
                print("No errors detected (or unable to parse file)")
            
            print("-" * 80)
    
    print()
    print("=" * 80)
    print(f"Total Errors Detected: {total_errors}")
    print("=" * 80)


if __name__ == "__main__":
    main()
