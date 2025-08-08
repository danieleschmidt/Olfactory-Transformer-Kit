#!/usr/bin/env python3
"""Basic code validation script without external dependencies."""

import ast
import sys
import os
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        ast.parse(content, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_imports(file_path):
    """Check for potentially dangerous imports."""
    dangerous_imports = [
        'os.system',
        'subprocess.call',
        'eval',
        'exec',
        '__import__',
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        for line_num, line in enumerate(content.split('\n'), 1):
            for dangerous in dangerous_imports:
                if dangerous in line and not line.strip().startswith('#'):
                    issues.append(f"Line {line_num}: Potentially dangerous import/call: {dangerous}")
        
        return issues
    except Exception as e:
        return [f"Error checking imports: {e}"]


def validate_docstrings(file_path):
    """Check for basic docstring presence."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(f"Missing docstring: {node.name} at line {node.lineno}")
        
        return issues
    except Exception as e:
        return [f"Error checking docstrings: {e}"]


def check_code_quality(file_path):
    """Basic code quality checks."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line.rstrip()) > 120:
                issues.append(f"Line {line_num}: Line too long ({len(line.rstrip())} > 120)")
            
            # Check for trailing whitespace
            if line.rstrip() != line.rstrip('\n').rstrip('\r'):
                issues.append(f"Line {line_num}: Trailing whitespace")
            
            # Check for tabs (prefer spaces)
            if '\t' in line:
                issues.append(f"Line {line_num}: Contains tabs (prefer 4 spaces)")
                
            # Check for print statements (should use logging)
            if line.strip().startswith('print(') and 'test' not in file_path.name.lower():
                issues.append(f"Line {line_num}: Use logging instead of print")
    
    except Exception as e:
        issues.append(f"Error checking code quality: {e}")
    
    return issues


def validate_file(file_path):
    """Validate a single Python file."""
    print(f"Validating {file_path}...")
    
    # Check syntax
    syntax_ok, syntax_error = validate_python_syntax(file_path)
    if not syntax_ok:
        print(f"  ❌ SYNTAX ERROR: {syntax_error}")
        return False
    
    print(f"  ✅ Syntax OK")
    
    # Check imports
    import_issues = check_imports(file_path)
    if import_issues:
        print(f"  ⚠️  Import issues:")
        for issue in import_issues[:5]:  # Show first 5
            print(f"    - {issue}")
    else:
        print(f"  ✅ Imports OK")
    
    # Check docstrings
    docstring_issues = validate_docstrings(file_path)
    if docstring_issues:
        print(f"  ⚠️  Docstring issues: {len(docstring_issues)} missing")
    else:
        print(f"  ✅ Docstrings OK")
    
    # Check code quality
    quality_issues = check_code_quality(file_path)
    if quality_issues:
        print(f"  ⚠️  Code quality issues: {len(quality_issues)} found")
        for issue in quality_issues[:3]:  # Show first 3
            print(f"    - {issue}")
    else:
        print(f"  ✅ Code quality OK")
    
    return syntax_ok


def main():
    """Main validation function."""
    repo_root = Path(__file__).parent
    python_files = []
    
    # Find all Python files
    for pattern in ['**/*.py']:
        python_files.extend(repo_root.glob(pattern))
    
    # Filter out some directories
    exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'build', 'dist']
    python_files = [
        f for f in python_files 
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]
    
    print(f"Found {len(python_files)} Python files to validate")
    print("=" * 50)
    
    failed_files = []
    total_files = len(python_files)
    
    for file_path in python_files:
        if not validate_file(file_path):
            failed_files.append(file_path)
        print()
    
    # Summary
    print("=" * 50)
    print("VALIDATION SUMMARY:")
    print(f"Total files: {total_files}")
    print(f"Passed: {total_files - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✅ All files passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()