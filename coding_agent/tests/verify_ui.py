import sys
import traceback

def verify():
    print("Testing UIAgent initialization...")
    try:
        from coding_agent.ui_agent.core.ui_agent import UIAgent
        print("[SUCCESS] Successfully imported UIAgent.")
    except Exception as e:
        print("[FAILURE] Could not import UIAgent.")
        traceback.print_exc()

if __name__ == "__main__":
    verify()
