from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import time
from coding_agent.core.experiential_storage import ExperientialStorage

class NodeType(Enum):
    ACTION = "ACTION"
    BRANCH = "BRANCH"
    ITERABLE = "ITERABLE"

@dataclass
class SchemaDifferentiation:
    """Metadata tracking why a schema was forked/differentiated."""
    justification: str
    is_human_guided: bool
    trigger_condition: str

@dataclass
class ErrorRecord:
    """Concise distillation of a past mistake so the agent avoids it."""
    error_id: str
    description: str
    recovery_summary: str
    failed_steps: List[str] = field(default_factory=list)

@dataclass
class TaskPath:
    """A valid execution path that successfully completed the task."""
    path_id: str
    primary_nodes: List[Union[ActionNode, BranchNode, IterableNode]] = field(default_factory=list)
    efficiency_score: float = 1.0  # Optional score based on step count or speed

@dataclass
class ProcessConstant:
    """A value that remains identical across multiple executions of the same task."""
    name: str
    value: Any
    confidence: float = 1.0

@dataclass
class ProcessVariable:
    """A value that changes across different executions of the same task."""
    name: str
    description: str
    examples: set = field(default_factory=set)

@dataclass
class ActionNode:
    """A concrete UI action step in a parameterized task."""
    node_id: str
    action_type: str  # 'click', 'type', 'move', etc.
    target_identifier: str  # Element ID or coordinate reference
    method_ref: str = "" # Associated variable or constant name (e.g. text to type)
    node_type: NodeType = NodeType.ACTION
    expected_outcome: str = ""

@dataclass
class BranchNode:
    """A conditional divergence in the task flow."""
    node_id: str
    condition_description: str
    if_true_path: List[Union[ActionNode, 'BranchNode', 'IterableNode']] = field(default_factory=list)
    if_false_path: List[Union[ActionNode, 'BranchNode', 'IterableNode']] = field(default_factory=list)
    node_type: NodeType = NodeType.BRANCH

@dataclass
class IterableNode:
    """A sequence of actions that repeats dynamically."""
    node_id: str
    iterable_description: str
    body: List[Union[ActionNode, BranchNode, 'IterableNode']] = field(default_factory=list)
    average_iterations: float = 1.0
    node_type: NodeType = NodeType.ITERABLE

@dataclass
class ParameterizedTaskSchema:
    """
    A deep-labeled, classified schema for a task.
    Accommodates process constants, variables, iterables, and conditional branches.
    Generated via experiential learning across multiple episodes.
    """
    task_id: str
    task_name: str
    app_class: str
    
    constants: Dict[str, ProcessConstant] = field(default_factory=dict)
    variables: Dict[str, ProcessVariable] = field(default_factory=dict)
    
    # The execution graph consisting of Actions, Branches, and Iterables
    execution_graph: List[Union[ActionNode, BranchNode, IterableNode]] = field(default_factory=list)
    
    # Path & Error Knowledge
    known_errors: Dict[str, ErrorRecord] = field(default_factory=dict)
    valid_paths: Dict[str, TaskPath] = field(default_factory=dict)
    
    # Metadata
    success_count: int = 1
    total_episodes_analyzed: int = 1
    estimated_total_steps: int = 1 # Order of magnitude estimate 
    is_planned_only: bool = False  # True if created during speculative planning (not yet experientially verified)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Differentiation Tracking
    parent_schema_id: Optional[str] = None
    differentiation: Optional[SchemaDifferentiation] = None
    
    def serialize(self) -> Dict[str, Any]:
        """Convert structure to dict for JSON serialization."""
        diff = self.differentiation
        diff_dict = None
        if diff is not None:
            diff_dict = {
                "justification": diff.justification,
                "is_human_guided": diff.is_human_guided,
                "trigger_condition": diff.trigger_condition
            }
            
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "app_class": self.app_class,
            "constants": {k: {"name": v.name, "value": v.value, "confidence": v.confidence} for k, v in self.constants.items()},
            "variables": {k: {"name": v.name, "description": v.description, "examples": list(v.examples)} for k, v in self.variables.items()},
            "known_errors": {
                k: {
                    "error_id": err.error_id, "description": err.description, 
                    "recovery_summary": err.recovery_summary, "failed_steps": err.failed_steps
                } for k, err in self.known_errors.items()
            },
            "valid_paths": {
                k: {
                    "path_id": p.path_id, "efficiency_score": p.efficiency_score,
                    "primary_nodes": [self._serialize_node(n) for n in p.primary_nodes]
                } for k, p in self.valid_paths.items()
            },
            "execution_graph": [self._serialize_node(node) for node in self.execution_graph],
            "success_count": self.success_count,
            "total_episodes_analyzed": self.total_episodes_analyzed,
            "estimated_total_steps": self.estimated_total_steps,
            "is_planned_only": self.is_planned_only,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "parent_schema_id": self.parent_schema_id,
            "differentiation": diff_dict
        }

    def save(self, storage: Optional[ExperientialStorage] = None):
        """Persists the schema and its components to disk."""
        if storage is None:
            storage = ExperientialStorage()
        
        # Save the main schema
        schema_dict = self.serialize()
        storage.save_task_schema(schema_dict)
        
        # Save nested items into their subdirectories
        for path_id, path in self.valid_paths.items():
            path_data = {
                "path_id": path.path_id,
                "efficiency_score": path.efficiency_score,
                "primary_nodes": [self._serialize_node(n) for n in path.primary_nodes]
            }
            storage.save_path(self.task_id, path_data)
            
        for error_id, error in self.known_errors.items():
            error_data = {
                "error_id": error.error_id,
                "description": error.description,
                "recovery_summary": error.recovery_summary,
                "failed_steps": error.failed_steps
            }
            storage.save_error(self.task_id, error_data)
            
        print(f"[SCHEMA] Fully persisted task {self.task_id} to disk.")

    def _serialize_node(self, node: Union[ActionNode, BranchNode, IterableNode]) -> Dict[str, Any]:
        if isinstance(node, ActionNode):
            return {
                "node_id": node.node_id,
                "node_type": "ACTION",
                "action_type": node.action_type,
                "target_identifier": node.target_identifier,
                "method_ref": node.method_ref,
                "expected_outcome": node.expected_outcome
            }
        elif isinstance(node, BranchNode):
            return {
                "node_id": node.node_id,
                "node_type": "BRANCH",
                "condition_description": node.condition_description,
                "if_true_path": [self._serialize_node(n) for n in node.if_true_path],
                "if_false_path": [self._serialize_node(n) for n in node.if_false_path]
            }
        elif isinstance(node, IterableNode):
            return {
                "node_id": node.node_id,
                "node_type": "ITERABLE",
                "iterable_description": node.iterable_description,
                "body": [self._serialize_node(n) for n in node.body],
                "average_iterations": node.average_iterations
            }
        return {}
