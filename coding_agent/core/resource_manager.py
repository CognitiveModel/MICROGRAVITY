import os
import json
import threading
from typing import Dict, List, Any

class ResourceManager:
    def __init__(self):
        self.api_quotas: Dict[str, Dict[str, int]] = {
            "gemini": {"total": 1000, "used": 0},
            "telegram": {"total": 30, "used": 0}
        }
        self.gui_instances: List[Dict[str, Any]] = [] # List of indexed GUI instances
        self._locks: Dict[str, threading.Lock] = {}

    def acquire_lock(self, resource_name: str, task_id: str = "sys", blocking: bool = True) -> bool:
        if resource_name not in self._locks:
            self._locks[resource_name] = threading.Lock()
        print(f"[MUTEX] Task {task_id} waiting for {resource_name}...")
        acquired = self._locks[resource_name].acquire(blocking=blocking)
        if acquired:
            print(f"[MUTEX] Task {task_id} ACQUIRED {resource_name}")
        return acquired

    def release_lock(self, resource_name: str, task_id: str = "sys"):
        lock = self._locks.get(resource_name)
        if lock:
            try:
                lock.release()
                print(f"[MUTEX] Task {task_id} RELEASED {resource_name}")
            except RuntimeError:
                pass # Already unlocked

    def update_quota_usage(self, service: str, amount: int = 1):
        if service in self.api_quotas:
            self.api_quotas[service]["used"] += amount

    def index_gui_instance(self, instance_type: str, metadata: Dict[str, Any]) -> int:
        """Indices a new GUI instance (e.g., a Chrome window)."""
        instance_id = len(self.gui_instances) + 1
        instance = {
            "id": instance_id,
            "type": instance_type,
            "metadata": metadata,
            "status": "active"
        }
        self.gui_instances.append(instance)
        print(f"Indexed {instance_type} instance #{instance_id}: {metadata}")
        return instance_id

    def introspect_gui_state(self, instance_id: int) -> Dict[str, Any]:
        """Returns the state of a specific GUI instance."""
        for instance in self.gui_instances:
            if instance["id"] == instance_id:
                return instance
        return {}

    def get_resource_report(self) -> str:
        """Returns a summary of resource usage and indexed instances."""
        report = {
            "api_quotas": self.api_quotas,
            "gui_instances": self.gui_instances
        }
        return json.dumps(report, indent=2)

    def analyze_sharing_conflict(self, task_description: str) -> List[str]:
        """Simple analysis of potential resource conflicts based on task description."""
        conflicts = []
        task_lower = task_description.lower()
        if "chrome" in task_lower or "browser" in task_lower:
            active_browsers = [i for i in self.gui_instances if i.get("type") == "chrome"]
            if len(active_browsers) > 5:
                conflicts.append("High number of concurrent browser windows (potential RAM pressure)")
        
        # Check quotas
        for service, quota in self.api_quotas.items():
            used = quota["used"]
            total = quota["total"]
            if total > 0 and used / total > 0.9:
                conflicts.append(f"API Quota for {service} is almost exhausted (>90%)")
                
        return conflicts

if __name__ == "__main__":
    rm = ResourceManager()
    rm.update_quota_usage("gemini", 500)
    rm.index_gui_instance("chrome", {"url": "https://gmail.com", "logged_in": True})
    rm.index_gui_instance("chrome", {"url": "https://github.com", "logged_in": False})
    
    print(rm.get_resource_report())
    print(f"Conflicts: {rm.analyze_sharing_conflict('Open more chrome tabs for research')}")
