import os
import sys

# Ensure local imports resolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from coding_agent.core.agent import IntrospectionAgent
from coding_agent.storage.world_model import WorldModel

def test():
    print("Initializing Multi-Layered Introspection Agent...")
    agent = IntrospectionAgent(max_retries=1)
    
    # We will trigger the Wisdom Pipeline using the keyword "architect"
    task = "architect a clean function to parse JSON dynamically."
    
    print("\n[TEST] Running Agent...")
    try:
        # Expected output: Semantic Drift WARNING, Structural Analogies loaded (if any), 
        # and World Model saving the DOMAIN -> TASK -> STRATEGY -> OUTCOME relation.
        agent.run(task)
    except Exception as e:
        print(f"Agent failed gracefully: {e}")
        
    print("\n[TEST] Verifying Graph Traces in WorldModel...")
    wm = WorldModel()
    print(f"Nodes recorded: {len(wm.nodes)}")
    print(f"Edges recorded: {len(wm.edges)}")

if __name__ == "__main__":
    test()
