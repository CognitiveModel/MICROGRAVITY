"""
SwarmScriptManager — Dynamic CV observation script authoring, testing, and execution.

Enables the swarm agent to write custom OpenCV-based observation scripts at runtime,
test them against sample frames, and register them as reusable tool-like artifacts.

Scripts are stored in agent_memory/scripts/ with metadata for discoverability.
Execution is sandboxed to only allow cv2, numpy, and math globals.

Usage in the agentic loop:
  1. Agent identifies a repeated UI pattern that needs a custom observable
  2. Agent (via planner) creates a script: {"action": "create_cv_script", "name": "...", "code": "..."}
  3. Script is syntax-validated, stored, and available for future use
  4. Agent uses the script: {"action": "run_cv_script", "name": "...", ...}
  5. Result is consumed by the anticipatory observer for fast feedback
"""

import os
import json
import time
import ast
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except ImportError:
    HAS_CV = False


class SwarmScriptManager:
    """Registry for agent-authored custom CV observation scripts."""

    # Allowed globals in script execution sandbox
    SAFE_MODULES = {"cv2", "np", "numpy", "math"}
    # Forbidden AST nodes for safety
    FORBIDDEN_NODES = {ast.Import, ast.ImportFrom, ast.Delete, ast.Global, ast.Nonlocal}

    def __init__(self, storage_dir: Optional[str] = None, anticipatory_observer=None):
        self.storage_dir = Path(storage_dir or "agent_memory/scripts")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.anticipatory_observer = anticipatory_observer
        self._scripts: Dict[str, Dict[str, Any]] = {}
        self._load_existing_scripts()

    def _load_existing_scripts(self):
        """Loads previously saved scripts from disk."""
        metadata_path = self.storage_dir / "script_registry.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._scripts = json.load(f)
                print(f"[SwarmScriptManager] Loaded {len(self._scripts)} scripts from registry")
            except Exception as e:
                print(f"[SwarmScriptManager] Failed to load registry: {e}")
                self._scripts = {}

    def _save_registry(self):
        """Persists the script registry to disk."""
        metadata_path = self.storage_dir / "script_registry.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._scripts, f, indent=2)
        except Exception as e:
            print(f"[SwarmScriptManager] Save error: {e}")

    def create_script(
        self,
        name: str,
        code: str,
        description: str = "",
        expected_inputs: str = "before, after, action",
        expected_output: str = "result (bool)",
    ) -> Dict[str, Any]:
        """
        Creates and registers a new CV observation script.

        Args:
            name: Unique script name (e.g., "check_button_highlight")
            code: Python code string. Must set `result = True/False`.
                  Available variables: cv2, np, before, after, action, result
            description: Human-readable description of what the script checks
            expected_inputs: Description of expected input variables
            expected_output: Description of expected output

        Returns:
            {"success": bool, "error": str or None, "validation": str}
        """
        # Validate syntax
        validation = self._validate_script(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Script validation failed: {validation['error']}",
                "validation": validation,
            }

        # Store the script
        script_meta = {
            "name": name,
            "code": code,
            "description": description,
            "expected_inputs": expected_inputs,
            "expected_output": expected_output,
            "created_at": time.time(),
            "execution_count": 0,
            "success_count": 0,
            "last_used": None,
        }
        self._scripts[name] = script_meta

        # Save code to file
        script_path = self.storage_dir / f"{name}.py"
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f"# Script: {name}\n# {description}\n# Created: {time.ctime()}\n\n")
                f.write(code)
        except Exception as e:
            return {"success": False, "error": f"Failed to save script file: {e}"}

        self._save_registry()
        print(f"[SwarmScriptManager] Created script '{name}': {description}")

        # Register as custom observable in AnticipatoryObserver if available
        if self.anticipatory_observer and hasattr(self.anticipatory_observer, 'register_custom_observable'):
            self.anticipatory_observer.register_custom_observable(name, code, script_meta)

        return {
            "success": True,
            "error": None,
            "validation": validation,
            "script_path": str(script_path),
        }

    def test_script(
        self,
        name: str,
        test_before_path: Optional[str] = None,
        test_after_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dry-runs a script against sample frames (or blank frames if none provided).
        Returns the script result and any errors.
        """
        if name not in self._scripts:
            return {"success": False, "error": f"Script '{name}' not found"}

        if not HAS_CV:
            return {"success": False, "error": "cv2 not available"}

        code = self._scripts[name]["code"]

        # Load or create test frames
        if test_before_path and os.path.exists(test_before_path):
            before = cv2.imread(test_before_path)
        else:
            before = np.zeros((480, 640, 3), dtype=np.uint8)

        if test_after_path and os.path.exists(test_after_path):
            after = cv2.imread(test_after_path)
        else:
            after = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray

        action = {"action": "test", "target": "test_element", "x": 320, "y": 240}

        try:
            result = self._execute_sandboxed(code, before, after, action)
            return {
                "success": True,
                "script_result": result,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "script_result": None,
                "error": str(e),
            }

    def apply_script(
        self,
        name: str,
        before_frame,
        after_frame,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes a stored script against real frames.
        Returns the script's boolean result.
        """
        if name not in self._scripts:
            return {"success": False, "error": f"Script '{name}' not found", "result": False}

        script_meta = self._scripts[name]
        code = script_meta["code"]

        try:
            result = self._execute_sandboxed(code, before_frame, after_frame, action)
            # Update stats
            script_meta["execution_count"] += 1
            script_meta["last_used"] = time.time()
            if result:
                script_meta["success_count"] += 1
            self._save_registry()

            return {
                "success": True,
                "result": bool(result),
                "method": f"CUSTOM_SCRIPT:{name}",
            }

        except Exception as e:
            print(f"[SwarmScriptManager] Execution error in '{name}': {e}")
            return {"success": False, "error": str(e), "result": False}

    def list_scripts(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered scripts."""
        return [
            {
                "name": meta["name"],
                "description": meta["description"],
                "execution_count": meta["execution_count"],
                "success_rate": (
                    meta["success_count"] / max(1, meta["execution_count"])
                ),
                "last_used": meta.get("last_used"),
            }
            for meta in self._scripts.values()
        ]

    def get_script(self, name: str) -> Optional[Dict[str, Any]]:
        """Returns full metadata for a specific script."""
        return self._scripts.get(name)

    def delete_script(self, name: str) -> bool:
        """Removes a script from the registry."""
        if name in self._scripts:
            del self._scripts[name]
            script_path = self.storage_dir / f"{name}.py"
            if script_path.exists():
                script_path.unlink()
            self._save_registry()
            print(f"[SwarmScriptManager] Deleted script '{name}'")
            return True
        return False

    # ═══════════════════════  Sandboxed Execution  ═══════════════════════

    def _execute_sandboxed(self, code: str, before, after, action: Dict) -> bool:
        """Executes script code in a restricted sandbox."""
        import math
        safe_globals = {
            "__builtins__": {
                "range": range, "len": len, "int": int, "float": float,
                "bool": bool, "str": str, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "min": min, "max": max,
                "abs": abs, "sum": sum, "round": round, "print": print,
                "True": True, "False": False, "None": None,
                "enumerate": enumerate, "zip": zip, "map": map,
            },
            "cv2": cv2 if HAS_CV else None,
            "np": np if HAS_CV else None,
            "math": math,
            "before": before,
            "after": after,
            "action": action,
            "result": False,
        }
        exec(code, safe_globals)  # noqa: S102
        return bool(safe_globals.get("result", False))

    # ═══════════════════════  Script Validation  ═══════════════════════

    def _validate_script(self, code: str) -> Dict[str, Any]:
        """Validates script syntax and safety."""
        # Step 1: Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax error: {e}"}

        # Step 2: Safety check — no imports, no deletes, no global/nonlocal
        for node in ast.walk(tree):
            if type(node) in self.FORBIDDEN_NODES:
                return {
                    "valid": False,
                    "error": f"Forbidden operation: {type(node).__name__}. "
                             f"Scripts can only use cv2, np, math, and builtins.",
                }

        # Step 3: Check that 'result' is assigned somewhere
        assigns_result = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "result":
                        assigns_result = True

        if not assigns_result:
            return {
                "valid": False,
                "error": "Script must assign to 'result' variable (e.g., result = True/False)",
            }

        return {"valid": True, "error": None}

    # ═══════════════════════  Artifact Integration  ═══════════════════════

    def is_script_action(self, action_name: str) -> bool:
        """Returns True if the action is a script management action."""
        return action_name in ("create_cv_script", "run_cv_script", "test_cv_script", "list_cv_scripts")

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches a script-related action."""
        action_name = action.get("action", "")

        if action_name == "create_cv_script":
            return self.create_script(
                name=action.get("name", f"script_{int(time.time())}"),
                code=action.get("code", ""),
                description=action.get("description", ""),
            )
        elif action_name == "run_cv_script":
            # This needs before/after frames from the caller
            return {"success": False, "error": "run_cv_script must be called via apply_script()"}
        elif action_name == "test_cv_script":
            return self.test_script(
                name=action.get("name", ""),
                test_before_path=action.get("test_before_path"),
                test_after_path=action.get("test_after_path"),
            )
        elif action_name == "list_cv_scripts":
            return {"success": True, "scripts": self.list_scripts()}

        return {"success": False, "error": f"Unknown script action: {action_name}"}
