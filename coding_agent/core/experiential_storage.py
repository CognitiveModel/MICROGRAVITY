import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ExperientialStorage:
    """
    Handles persistent storage for the Experiential Learning Architecture.
    Organizes data into folders by TaskID, PathID, and ErrorID.
    """
    def __init__(self, base_dir: str = "experiential_data"):
        # Use absolute path relative to the current working directory
        self.base_dir = Path(os.getcwd()) / base_dir
        self.tasks_dir = self.base_dir / "tasks"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensures the base directories exist."""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def get_task_dir(self, task_id: str) -> Path:
        """Returns the base directory for a specific task."""
        task_dir = self.tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def get_path_dir(self, task_id: str, path_id: str) -> Path:
        """Returns the directory for a specific execution path within a task."""
        path_dir = self.get_task_dir(task_id) / "paths" / path_id
        path_dir.mkdir(parents=True, exist_ok=True)
        return path_dir

    def get_error_dir(self, task_id: str, error_id: str) -> Path:
        """Returns the directory for a specific error record within a task."""
        error_dir = self.get_task_dir(task_id) / "errors" / error_id
        error_dir.mkdir(parents=True, exist_ok=True)
        return error_dir

    def save_task_schema(self, schema_dict: Dict[str, Any]):
        """Saves a ParameterizedTaskSchema to its structured folder."""
        task_id = schema_dict.get("task_id")
        if not task_id:
            raise ValueError("Schema must contain a task_id")
        
        task_dir = self.get_task_dir(task_id)
        with open(task_dir / "schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)
        print(f"[STORAGE] Saved task schema for {task_id}")

    def save_path(self, task_id: str, path_data: Dict[str, Any]):
        """Saves a TaskPath to its structured folder."""
        path_id = path_data.get("path_id")
        if not path_id:
            raise ValueError("Path data must contain a path_id")
            
        path_dir = self.get_path_dir(task_id, path_id)
        with open(path_dir / "path_data.json", "w", encoding="utf-8") as f:
            json.dump(path_data, f, indent=2)
        print(f"[STORAGE] Saved path {path_id} for task {task_id}")

    def save_error(self, task_id: str, error_data: Dict[str, Any]):
        """Saves an ErrorRecord to its structured folder."""
        error_id = error_data.get("error_id")
        if not error_id:
            raise ValueError("Error data must contain an error_id")
            
        error_dir = self.get_error_dir(task_id, error_id)
        with open(error_dir / "error_data.json", "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        print(f"[STORAGE] Saved error {error_id} for task {task_id}")

if __name__ == "__main__":
    # Test internal logic
    storage = ExperientialStorage()
    mock_task_id = "task_test_123"
    storage.save_task_schema({"task_id": mock_task_id, "name": "Test Task"})
    storage.save_path(mock_task_id, {"path_id": "path_01", "steps": []})
    storage.save_error(mock_task_id, {"error_id": "err_01", "description": "Fail"})
