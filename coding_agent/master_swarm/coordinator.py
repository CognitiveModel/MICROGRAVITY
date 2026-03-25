import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class NodeStatus:
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class TaskNode:
    """Represents a single node in a Swarm Execution DAG."""
    def __init__(self, description: str, node_id: Optional[str] = None):
        self.node_id = node_id or uuid.uuid4().hex[:8]
        self.description = description
        self.status = NodeStatus.PENDING
        self.parents: List['TaskNode'] = []
        self.children: List['TaskNode'] = []
        self.result: Any = None
        self.completion_event = asyncio.Event()

    def add_child(self, child: 'TaskNode'):
        self.children.append(child)
        child.parents.append(self)

class SwarmCoordinator:
    """Centralizes parallel DAG execution and Task lifecycle across the physical swarm."""
    
    def __init__(self):
        self._active_graphs: Dict[str, List[TaskNode]] = {}
        # Global dictionary of all tasks mapped to their cancellation events/tasks
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._resource_locks: Dict[str, asyncio.Lock] = {}

    def register_graph(self, task_id: str, nodes: List[TaskNode]):
        """Registers a full DAG graph under a specific Master Task ID."""
        self._active_graphs[task_id] = nodes
        logger.info(f"[COORDINATOR] Registered DAG for Master Task {task_id} with {len(nodes)} nodes.")

    async def wait_for_dependencies(self, node: TaskNode) -> bool:
        """Waits asynchronously until all parent nodes are COMPLETE."""
        for parent in node.parents:
            # Wait for the parent to signal completion
            await parent.completion_event.wait()
            if parent.status in [NodeStatus.FAILED, NodeStatus.CANCELLED]:
                logger.warning(f"[COORDINATOR] Node {node.node_id} aborting. Parent {parent.node_id} failed.")
                node.status = NodeStatus.CANCELLED
                node.completion_event.set() # Propagate cancellation downstream
                return False
        return True

    def mark_node_completed(self, node: TaskNode, result: Any = None):
        """Marks a node complete and unblocks children."""
        node.status = NodeStatus.COMPLETED
        node.result = result
        node.completion_event.set()
        
    def mark_node_failed(self, node: TaskNode, error: str):
        """Marks a node as failed, cascading abort to children."""
        node.status = NodeStatus.FAILED
        node.result = f"Error: {error}"
        node.completion_event.set()
        
    async def acquire_resource(self, resource_name: str, task_id: str = "sys"):
        """Active Mutex Lock for OS/Desktop resources."""
        if resource_name not in self._resource_locks:
            self._resource_locks[resource_name] = asyncio.Lock()
        
        logger.info(f"[LOCKED] Task {task_id} waiting for {resource_name}...")
        await self._resource_locks[resource_name].acquire()
        logger.info(f"[LOCKED] Task {task_id} ACQUIRED {resource_name}")

    def release_resource(self, resource_name: str, task_id: str = "sys"):
        """Releases the active resource lock."""
        lock = self._resource_locks.get(resource_name)
        if lock and lock.locked():
            lock.release()
            logger.info(f"[UNLOCKED] Task {task_id} released {resource_name}")

    def register_system_task(self, task_id: str, task_obj: asyncio.Task):
        self._running_tasks[task_id] = task_obj

    def remove_system_task(self, task_id: str):
        self._running_tasks.pop(task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        """Triggers asyncio cancellation down a specific subagent/task tree."""
        if task_id in self._running_tasks:
            logger.info(f"[COORDINATOR] Sending kill signal to task {task_id}")
            self._running_tasks[task_id].cancel()
            return True
        logger.warning(f"[COORDINATOR] Task {task_id} not found to cancel.")
        return False
