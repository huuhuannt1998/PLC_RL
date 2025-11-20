"""
XML to NuSMV Automatic Converter
Converts PLC XML files to NuSMV formal verification models
"""
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Set, Tuple, Optional


class XMLToNuSMVConverter:
    """Converts PLC XML to NuSMV model for formal verification"""
    
    def __init__(self):
        self.variables: Set[str] = set()
        self.inputs: Set[str] = set()
        self.outputs: Set[str] = set()
        self.logic_statements: List[str] = []
        self.properties: List[Tuple[str, str]] = []  # (name, ltl_formula)
        
    def extract_code_from_xml(self, xml_path: str) -> str:
        """Extract SCL/STL code from XML"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find StructuredText element
            for structured_text in root.iter('{http://www.siemens.com/automation/Openness/SW/NetworkSource/StructuredText/v3}StructuredText'):
                code = structured_text.text
                if code:
                    return code.strip()
            
            return ""
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return ""
    
    def parse_scl_code(self, code: str):
        """Parse SCL code and extract variables and logic"""
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip comments
            if line.startswith('//') or not line:
                continue
            
            # Extract variables from assignments
            if ':=' in line:
                var_match = re.search(r'#(\w+)\s*:=', line)
                if var_match:
                    var_name = var_match.group(1)
                    self.outputs.add(var_name)
                    self.variables.add(var_name)
            
            # Extract variables from conditions
            var_matches = re.findall(r'#(\w+)', line)
            for var in var_matches:
                self.variables.add(var)
                # Assume variables in conditions are inputs (heuristic)
                if ':=' not in line or line.index('#' + var) < line.index(':='):
                    self.inputs.add(var)
            
            # Store logic statements
            if 'IF' in line or 'ELSIF' in line or 'ELSE' in line or 'END_IF' in line:
                self.logic_statements.append(line)
            elif ':=' in line:
                self.logic_statements.append(line)
    
    def convert_condition_to_nusmv(self, condition: str) -> str:
        """Convert SCL condition to NuSMV syntax"""
        # Remove # prefix
        condition = condition.replace('#', '')
        
        # Convert operators
        condition = condition.replace(' AND ', ' & ')
        condition = condition.replace(' OR ', ' | ')
        condition = condition.replace(' NOT ', ' !')
        condition = condition.replace('NOT ', '!')
        
        # Handle comparison operators (keep as is)
        # <, >, <=, >=, =, <> are valid in NuSMV
        condition = condition.replace('<>', '!=')
        
        return condition.strip()
    
    def generate_nusmv_model(self, xml_path: str, model_name: str = None) -> str:
        """Generate complete NuSMV model from XML"""
        # Extract and parse code
        code = self.extract_code_from_xml(xml_path)
        if not code:
            return "-- Error: No code found in XML"
        
        self.parse_scl_code(code)
        
        if not model_name:
            model_name = "PLC_Model"
        
        # Build NuSMV model
        nusmv_code = f"MODULE main\n\n"
        
        # VAR declarations
        nusmv_code += "VAR\n"
        for var in sorted(self.variables):
            nusmv_code += f"  {var} : boolean;\n"
        nusmv_code += "\n"
        
        # ASSIGN section
        nusmv_code += "ASSIGN\n"
        
        # Initial values
        for var in sorted(self.variables):
            nusmv_code += f"  init({var}) := FALSE;\n"
        nusmv_code += "\n"
        
        # Next state logic
        nusmv_code += self.generate_next_state_logic()
        
        # Generate safety properties
        nusmv_code += self.generate_safety_properties()
        
        return nusmv_code
    
    def generate_next_state_logic(self) -> str:
        """Generate next() statements from logic"""
        logic = ""
        
        # Group assignments by variable
        variable_conditions = {}
        
        current_if_block = []
        in_if_block = False
        
        for stmt in self.logic_statements:
            if 'IF' in stmt and 'END_IF' not in stmt:
                in_if_block = True
                current_if_block = [stmt]
            elif in_if_block:
                current_if_block.append(stmt)
                if 'END_IF' in stmt:
                    in_if_block = False
                    # Extract variable and conditions from IF block
                    conditions = self.extract_conditions_from_if_block(current_if_block)
                    for var, cond_list in conditions.items():
                        if var not in variable_conditions:
                            variable_conditions[var] = []
                        variable_conditions[var].extend(cond_list)
                    current_if_block = []
            elif ':=' in stmt:
                # Simple assignment
                match = re.search(r'(\w+)\s*:=\s*(.+);', stmt)
                if match:
                    var = match.group(1)
                    value = self.convert_condition_to_nusmv(match.group(2))
                    if var not in variable_conditions:
                        variable_conditions[var] = []
                    variable_conditions[var].append(('TRUE', value))
        
        # Generate NuSMV case statements for each variable
        for var, conditions in variable_conditions.items():
            logic += f"  next({var}) := case\n"
            for condition, value in conditions:
                logic += f"    {condition} : {value};\n"
            logic += f"    TRUE : {var};\n"
            logic += "  esac;\n\n"
        
        return logic
    
    def extract_conditions_from_if_block(self, if_block: List[str]) -> Dict[str, List]:
        """Extract variable assignments and their conditions from IF block"""
        conditions = {}
        current_condition = None
        
        for line in if_block:
            line_stripped = line.strip()
            
            if line_stripped.startswith('IF') or line_stripped.startswith('ELSIF'):
                # Extract condition
                condition_match = re.search(r'(?:IF|ELSIF)\s+(.+?)\s+THEN', line_stripped)
                if condition_match:
                    current_condition = self.convert_condition_to_nusmv(condition_match.group(1))
            
            elif line_stripped.startswith('ELSE') and not line_stripped.startswith('ELSIF'):
                current_condition = 'TRUE'
            
            elif ':=' in line_stripped and current_condition:
                # Extract variable assignment
                var_match = re.search(r'(\w+)\s*:=\s*(.+?);', line_stripped)
                if var_match:
                    var = var_match.group(1)
                    value = var_match.group(2).upper()
                    if var not in conditions:
                        conditions[var] = []
                    conditions[var].append((current_condition, value))
        
        return conditions
    
    def convert_if_block_to_case(self, if_block: List[str]) -> str:
        """Convert IF/ELSIF/ELSE block to NuSMV case statement"""
        result = ""
        
        # Find output variable
        output_var = None
        for line in if_block:
            if ':=' in line:
                match = re.search(r'(\w+)\s*:=', line)
                if match:
                    output_var = match.group(1)
                    break
        
        if not output_var:
            return "  -- Error: Could not find output variable\n"
        
        result += f"  next({output_var}) := case\n"
        
        for line in if_block:
            if line.strip().startswith('IF') or line.strip().startswith('ELSIF'):
                # Extract condition
                condition_match = re.search(r'(?:IF|ELSIF)\s+(.+?)\s+THEN', line)
                if condition_match:
                    condition = self.convert_condition_to_nusmv(condition_match.group(1))
                    
                    # Find assignment value in next lines
                    value = "TRUE"  # Default
                    assignment_line = None
                    idx = if_block.index(line) + 1
                    if idx < len(if_block):
                        assignment_line = if_block[idx]
                        value_match = re.search(r':=\s*(\w+)', assignment_line)
                        if value_match:
                            value = value_match.group(1).upper()
                    
                    result += f"    {condition} : {value};\n"
            
            elif line.strip().startswith('ELSE') and not line.strip().startswith('ELSIF'):
                # ELSE clause (default case)
                idx = if_block.index(line) + 1
                if idx < len(if_block):
                    assignment_line = if_block[idx]
                    value_match = re.search(r':=\s*(\w+)', assignment_line)
                    if value_match:
                        value = value_match.group(1).upper()
                        result += f"    TRUE : {value};\n"
        
        # Default to keep current value
        result += f"    TRUE : {output_var};\n"
        result += "  esac;\n\n"
        
        return result
    
    def convert_assignment(self, assignment: str) -> str:
        """Convert simple assignment to NuSMV next() statement"""
        match = re.search(r'(\w+)\s*:=\s*(.+);', assignment)
        if not match:
            return ""
        
        var = match.group(1)
        value = self.convert_condition_to_nusmv(match.group(2))
        
        return f"  next({var}) := {value};\n\n"
    
    def generate_safety_properties(self) -> str:
        """Generate common safety properties to verify"""
        properties = "\n-- Safety Properties\n"
        
        # Detect common patterns
        safety_vars = [v for v in self.variables if any(
            keyword in v.lower() for keyword in ['emergency', 'stop', 'safety', 'alarm']
        )]
        
        critical_vars = [v for v in self.outputs if any(
            keyword in v.lower() for keyword in ['motor', 'pump', 'valve', 'machine', 'heater']
        )]
        
        # Generate properties
        if safety_vars and critical_vars:
            for safety_var in safety_vars:
                for critical_var in critical_vars:
                    prop_name = f"safety_{safety_var}_stops_{critical_var}"
                    properties += f"LTLSPEC G ({safety_var} -> F !{critical_var})\n"
                    properties += f"  -- Property: If {safety_var} is active, {critical_var} should eventually stop\n\n"
        
        # Mutual exclusion properties
        if len(critical_vars) >= 2:
            properties += f"LTLSPEC G !({critical_vars[0]} & {critical_vars[1]})\n"
            properties += f"  -- Property: {critical_vars[0]} and {critical_vars[1]} should not be active simultaneously\n\n"
        
        # Liveness properties
        for var in self.inputs:
            if 'start' in var.lower():
                for output_var in self.outputs:
                    if 'motor' in output_var.lower() or 'machine' in output_var.lower():
                        properties += f"LTLSPEC G ({var} -> F {output_var})\n"
                        properties += f"  -- Property: If {var} is pressed, {output_var} should eventually activate\n\n"
        
        return properties
    
    def convert_file(self, xml_path: str, output_path: str = None) -> str:
        """Convert XML file to NuSMV and save"""
        model = self.generate_nusmv_model(xml_path)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(model)
            print(f"✓ Generated NuSMV model: {output_path}")
        
        return model


def batch_convert(input_dir: str, output_dir: str):
    """Convert all XML files in directory to NuSMV models"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    print(f"Converting {len(xml_files)} XML files to NuSMV...")
    print("=" * 70)
    
    for xml_file in xml_files:
        converter = XMLToNuSMVConverter()
        xml_path = os.path.join(input_dir, xml_file)
        smv_file = xml_file.replace('.xml', '.smv')
        smv_path = os.path.join(output_dir, smv_file)
        
        try:
            converter.convert_file(xml_path, smv_path)
        except Exception as e:
            print(f"✗ Error converting {xml_file}: {e}")
    
    print("=" * 70)
    print(f"✓ Conversion complete. Models saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single file conversion
        xml_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else xml_path.replace('.xml', '.smv')
        
        converter = XMLToNuSMVConverter()
        converter.convert_file(xml_path, output_path)
    else:
        # Batch convert training data
        batch_convert('training_data/buggy', 'verification/auto_generated/buggy')
        batch_convert('training_data/correct', 'verification/auto_generated/correct')
