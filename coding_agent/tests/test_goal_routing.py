import sys
import os
import re

# Add project root to path
sys.path.append(r'c:\Users\HP\coding agent final')

def test_goal_extraction():
    # Mocking regex and indicators from agent.py
    meta_indicators = ["initial plan", "relies heavily", "delegating to", "requires ui automation"]
    original_task = "create excel sheet containg the report of the daily oil prices now a days"
    
    test_cases = [
        # Case 1: Standard delegation logic with preamble (The failure case)
        {
            "response": "FINAL_RESULT: The initial plan relies heavily on UI automation... I am delegating the task to the swarm. DELEGATE_TO_SWARM.",
            "expected_revert": True
        },
        # Case 2: Refined delegation
        {
            "response": "FINAL_RESULT: DELEGATE_TO_SWARM: Search for oil prices and save to Excel.",
            "expected": "Search for oil prices and save to Excel.",
            "expected_revert": False
        },
        # Case 3: Empty delegation
        {
            "response": "FINAL_RESULT: DELEGATE_TO_SWARM",
            "expected_revert": True
        }
    ]

    print("Running Goal Extraction Verification...")
    
    for i, case in enumerate(test_cases):
        response = case["response"]
        result = response.split("FINAL_RESULT:")[1].strip()
        task_to_delegate = result
        
        # Simulated logic from agent.py
        task_to_delegate = re.sub(r'^FINAL_RESULT:\s*', '', task_to_delegate, flags=re.IGNORECASE)
        if "DELEGATE_TO_SWARM:" in task_to_delegate:
            task_to_delegate = task_to_delegate.split("DELEGATE_TO_SWARM:")[1].strip()
        elif "DELEGATE_TO_SWARM" in task_to_delegate:
            task_to_delegate = task_to_delegate.replace("DELEGATE_TO_SWARM", "").strip()
            
        task_to_delegate = re.sub(r'```[\s\S]*?```', '', task_to_delegate).strip()
        task_match = re.search(r'task\s*=\s*["\'](.*?)["\']', task_to_delegate)
        if task_match:
            task_to_delegate = task_match.group(1)
            
        is_meta = any(ind in task_to_delegate.lower() for ind in meta_indicators)
        final_task = original_task if (not task_to_delegate or len(task_to_delegate) < 10 or is_meta) else task_to_delegate
        
        reverted = (final_task == original_task)
        pass_status = (reverted == case["expected_revert"])
        
        if not case["expected_revert"] and "expected" in case:
            pass_status = pass_status and (final_task == case["expected"])

        print(f"Test {i+1}: {'PASS' if pass_status else 'FAIL'}")
        print(f"  Input: {response[:50]}...")
        print(f"  Extracted: '{task_to_delegate[:50]}...'")
        print(f"  Final: '{final_task[:50]}...'")
        print(f"  Reverted to Original: {reverted}")
        
if __name__ == "__main__":
    test_goal_extraction()
