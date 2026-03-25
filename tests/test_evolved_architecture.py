import asyncio
import threading
import time
import sys
import os

# Ensure the root pkg is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coding_agent.master_swarm.coordinator import SwarmCoordinator, TaskNode
from coding_agent.core.resource_manager import ResourceManager
from coding_agent.core.bus import MessageBus, Message

print("====================================")
print("VERIFYING EVOLVED ARCHITECTURE")
print("====================================")

async def test_dag_coordinator():
    print("\n[TEST 1] SwarmCoordinator DAG Execution")
    coord = SwarmCoordinator()
    
    node_a = TaskNode("Task_A", node_id="A")
    node_b = TaskNode("Task_B", node_id="B")
    node_c = TaskNode("Task_C", node_id="C")
    
    # A and B must complete before C
    node_a.add_child(node_c)
    node_b.add_child(node_c)
    
    coord.register_graph("MASTER_1", [node_a, node_b, node_c])
    
    assert "MASTER_1" in coord._active_graphs
    print(" - Dependencies registered correctly.")
    
    # Simulate completion of A and B
    coord.mark_node_completed(node_a)
    coord.mark_node_completed(node_b)
    
    # Wait for dependencies should succeed immediately
    await asyncio.wait_for(coord.wait_for_dependencies(node_c), timeout=1.0)
    print(" - Node C acquired execution graph successfully after A and B.")
    print("[PASS] SwarmCoordinator tests out.")

def test_resource_manager():
    print("\n[TEST 2] Active Resource Mutex Locks")
    rm = ResourceManager()
    
    def thread_1():
        rm.acquire_lock("PORT_3000", "t1")
        time.sleep(1)
        rm.release_lock("PORT_3000", "t1")

    def thread_2(results):
        start = time.time()
        rm.acquire_lock("PORT_3000", "t2")
        elapsed = time.time() - start
        rm.release_lock("PORT_3000", "t2")
        results.append(elapsed)

    res = []
    t1 = threading.Thread(target=thread_1)
    t2 = threading.Thread(target=thread_2, args=(res,))
    
    t1.start()
    time.sleep(0.1)
    t2.start()
    
    t1.join()
    t2.join()
    
    assert res[0] >= 0.8, f"Thread 2 didn't block! It took {res[0]}s"
    print(f" - Mutex locked successfully. Thread 2 waited {res[0]:.2f}s for Thread 1.")
    print("[PASS] ResourceManager Mutex locks test out.")

async def test_bus_interrupt():
    print("\n[TEST 3] MessageBus Preemption/Interrupt")
    bus = MessageBus()
    
    async def dummy_listener():
        async for msg in bus.subscribe("inbound"):
            if msg.message_type == "INTERRUPT":
                assert msg.metadata["target_task_id"] == "TASK_123"
                print(" - INTERRUPT propagated through bus correctly.")
                break

    task = asyncio.create_task(dummy_listener())
    
    # Yield control so the listener can attach to the bus queue
    await asyncio.sleep(0.1)
    
    await bus.publish(Message(
        content="Cancel Pls",
        message_type="INTERRUPT",
        metadata={"target_task_id": "TASK_123"}
    ))
    
    await asyncio.wait_for(task, timeout=1.0)
    print("[PASS] MessageBus Preemption test out.")

async def main():
    await test_dag_coordinator()
    test_resource_manager()
    await test_bus_interrupt()
    print("\n====================================")
    print("ALL TESTS PASSED SUCCESSFULLY.")
    print("====================================")

if __name__ == "__main__":
    asyncio.run(main())
