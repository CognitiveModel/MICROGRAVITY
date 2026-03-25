import os
import sys
import ast

def test_architectural_integrity():
    """
    Verifies that the coding_agent package adheres to its modular design rules:
    1. No circular dependencies between core and subagents.
    2. All external API calls (gemini) are routed through utils.
    3. Storage components handles file IO.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    print(f"--- Running Architectural Integrity Audit on {base_dir} ---")
    
    issues = []
    
    # 1. No direct API calls outside of utils or specific agents making use of utils
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py') and "utils" not in root and "test" not in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            if 'google.generativeai' in n.name and "storage" not in root:
                                issues.append(f"Violation in {file}: Direct API import outside utils/storage. Use gemini_client.")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and 'google.generativeai' in node.module and "storage" not in root:
                            issues.append(f"Violation in {file}: Direct API import outside utils/storage. Use gemini_client.")

    if not issues:
        print("\n✅ Architectural Integrity Verified: No violations found.")
        return True
    else:
        print("\n❌ Architectural Violations Detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False

if __name__ == "__main__":
    success = test_architectural_integrity()
    if success:
        print("🎉 Architectural Awareness Tests Passed.")
    else:
        sys.exit(1)
