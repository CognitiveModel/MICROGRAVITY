import sys
import os
import time

sys.path.append(os.getcwd())

from coding_agent.core.agent import IntrospectionAgent

def main():
    print("--- [STARTING MICROGRAVITY TELEGRAM AUTOMATION] ---")
    
    # Initialize the agent (this starts the gateway dispatcher and telegram polling)
    agent = IntrospectionAgent()
    
    print("Microgravity and Telegram Channel initialized.")
    
    task = "open the telegram from the web and then send the message hi to dikshant present in the chat list using the ui agen"
    
    print(f"Task: {task}", flush=True)
    result = agent.run(task)
    
    print("\n--- [FINAL RESULT] ---", flush=True)
    # Safely print result by ignoring non-supported characters (like emojis) if needed
    try:
        print(result, flush=True)
    except UnicodeEncodeError:
        print(result.encode('ascii', 'ignore').decode('ascii'), flush=True)
    
    # Wait for the background swarm thread to complete (up to 5 minutes)
    print("\n[INFO] Waiting for background swarm automation to complete...", flush=True)
    time.sleep(300) 


if __name__ == "__main__":
    main()


