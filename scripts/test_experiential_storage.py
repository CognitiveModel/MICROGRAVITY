import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(os.getcwd())
sys.path.append(str(project_root))

from coding_agent.ui_agent.planning.task_schema import ParameterizedTaskSchema, ErrorRecord, TaskPath, ActionNode
from coding_agent.core.experiential_storage import ExperientialStorage

def test_task_storage():
    print("--- Starting Task Storage Test ---")
    
    # 1. Initialize Storage
    storage = ExperientialStorage(base_dir="test_experiential_data")
    
    # 2. Create a mock schema
    task_id = "test_task_789"
    schema = ParameterizedTaskSchema(
        task_id=task_id,
        task_name="Verify Storage Logic",
        app_class="UTILITY"
    )
    
    # 3. Add a path
    path_id = "path_alpha"
    nodes = [ActionNode(node_id="n1", action_type="test", target_identifier="null")]
    schema.valid_paths[path_id] = TaskPath(path_id=path_id, primary_nodes=nodes)
    
    # 4. Add an error
    error_id = "err_beta"
    schema.known_errors[error_id] = ErrorRecord(
        error_id=error_id,
        description="Test error",
        recovery_summary="Test recovery",
        failed_steps=["step 1"]
    )
    
    # 5. Save
    print(f"Saving task {task_id}...")
    schema.save(storage)
    
    # 6. Verify disk structure
    base = project_root / "test_experiential_data" / "tasks" / task_id
    expected_files = [
        base / "schema.json",
        base / "paths" / path_id / "path_data.json",
        base / "errors" / error_id / "error_data.json"
    ]
    
    all_exist = True
    for p in expected_files:
        if p.exists():
            print(f"[SUCCESS] Found: {p.relative_to(project_root)}")
        else:
            print(f"[FAILURE] Missing: {p.relative_to(project_root)}")
            all_exist = False
            
    if all_exist:
        print("\n--- TEST PASSED ---")
    else:
        print("\n--- TEST FAILED ---")

if __name__ == "__main__":
    test_task_storage()
