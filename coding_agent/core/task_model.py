import enum
import uuid
from typing import List, Dict, Optional, Set

class ResponsibilityType(enum.Enum):
    SELF = "Self-Responsible"
    OTHER = "Other-Responsible"
    GREATER = "Greater Responsibility"

class ActionType(enum.Enum):
    RECURSIVE = "Recursive"
    REPETITIVE = "Repetitive"

class TaskStatus(enum.Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"

class TaskNode:
    def __init__(self, name: str, description: str, responsibility: ResponsibilityType = ResponsibilityType.SELF):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.responsibility = responsibility
        self.status = TaskStatus.PENDING
        self.children: List['TaskNode'] = []
        self.dependencies: Set[str] = set() # Set of IDs this task depends on
        self.action_type = ActionType.REPETITIVE # Default

    def add_child(self, child: 'TaskNode'):
        self.children.append(child)

    def add_dependency(self, task_id: str):
        self.dependencies.add(task_id)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "responsibility": self.responsibility.value,
            "dependencies": list(self.dependencies),
            "children": [child.to_dict() for child in self.children]
        }

class TaskModel:
    def __init__(self):
        self.root_goals: List[TaskNode] = []
        self.task_registry: Dict[str, TaskNode] = {}

    def create_goal(self, name: str, description: str) -> TaskNode:
        goal = TaskNode(name, description)
        self.root_goals.append(goal)
        self.task_registry[goal.id] = goal
        return goal

    def add_task(self, parent_id: str, name: str, description: str, 
                 responsibility: ResponsibilityType = ResponsibilityType.SELF) -> TaskNode:
        parent = self.task_registry.get(parent_id)
        if not parent:
            raise ValueError(f"Parent task {parent_id} not found.")
        
        task = TaskNode(name, description, responsibility)
        parent.add_child(task)
        self.task_registry[task.id] = task
        return task

    def get_task_summary(self) -> str:
        """Returns a string representation of the task hierarchy and dependencies."""
        import json
        return json.dumps([goal.to_dict() for goal in self.root_goals], indent=2)

    def resolve_dependencies(self, task_id: str) -> bool:
        """Checks if all dependencies for a task are completed."""
        task = self.task_registry.get(task_id)
        if not task:
            return False
        for dep_id in task.dependencies:
            dep = self.task_registry.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True

if __name__ == "__main__":
    # Internal test
    tm = TaskModel()
    goal = tm.create_goal("Build Introspection Suite", "Core system enhancement")
    t1 = tm.add_task(goal.id, "Design Schema", "Database structure")
    t2 = tm.add_task(goal.id, "Implement Logic", "Core algorithms")
    t2.add_dependency(t1.id)
    
    print(tm.get_task_summary())
    print(f"Is t2 ready? {tm.resolve_dependencies(t2.id)}")
    t1.status = TaskStatus.COMPLETED
    print(f"Is t2 ready after t1 completes? {tm.resolve_dependencies(t2.id)}")
