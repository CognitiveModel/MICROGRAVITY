import queue
import time
from typing import Callable, Any

class SwarmInterface:
    """
    A communication stub to integrate this UI Agent into a larger swarm architecture.
    """
    def __init__(self, hitl_contextualizer=None):
        self.task_queue = queue.Queue()
        self.result_callbacks = []
        self.hitl = hitl_contextualizer
        self.interrupt_callbacks = []
        
    def submit_task(self, task_description: str, priority: int = 1, current_screen: str = ""):
        """
        Called by other agents or users in the swarm to request UI interactions or give feedback.
        """
        print(f"[SwarmInterface] Message received: {task_description}")
        
        # Intercept via HITL if available
        if self.hitl:
            analysis = self.hitl.process_message(task_description, current_screen)
            category = analysis.get("category")
            
            print(f"[SwarmInterface] HITL Categorized as: {category} (Urgency: {analysis.get('urgency')})")
            
            if category in ("FEEDBACK_CORRECTION", "INTERRUPT_STOP"):
                # Broadcast immediately to active agent, bypassing the regular task queue
                for callback in self.interrupt_callbacks:
                    callback(analysis)
                return
            elif category == "NOISE_SPECULATIVE":
                # Ignore chitchat
                print("[SwarmInterface] Ignoring noise.")
                return
                
            # If NEW_TASK, modify description to extracted intent and fall through to queue
            task_description = analysis.get("extracted_intent", task_description)
            task_id = f"task_{int(time.time())}"
            self.hitl.register_task(task_id, task_description)

        self.task_queue.put((priority, task_description))
        
    def get_next_task(self) -> str:
        """
        Called by the UIAgent to get the next task from the swarm.
        """
        if not self.task_queue.empty():
             # In real implementation, handle tuple unpacking and PriorityQueue
             return self.task_queue.get()[1]
        return None
        
    def register_interrupt_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Registers a callback for the UI Agent to receive immediate mid-task feedback/interrupts.
        """
        self.interrupt_callbacks.append(callback)
        
    def register_callback(self, callback: Callable[[str, Any], None]):
        """
        Registers a callback so the UI Agent can report task completion or derived info back to the swarm.
        """
        self.result_callbacks.append(callback)
        
    def report_result(self, task_id: str, result_data: Any):
        """
        Broadcasts the result of a UI action back to the swarm.
        """
        print(f"[SwarmInterface] Broadcasting result for {task_id}")
        for callback in self.result_callbacks:
             callback(task_id, result_data)
