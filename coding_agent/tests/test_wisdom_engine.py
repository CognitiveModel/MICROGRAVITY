import asyncio
from coding_agent.core.wisdom_engine import WisdomEngine, SophisticationLevel

class MockGeminiClient:
    async def generate_content(self, prompt: str) -> str:
        if "CHARACTERIZATION" in prompt:
            return "This task has 3 dimensions: 1. Code complexity. 2. UI flow. 3. Data consistency. It requires deep analysis."
        if "EXCLUSIONS" in prompt:
            return "We are explicitly excluding the backend database optimization because this is a frontend-only refactor."
        if "CONSTRUCTION" in prompt:
            return "The final structure uses a hook-based state machine. DISCERNMENT: State coupling was an illusion created by props drilling."
        return "Generic wisdom insight."

async def test_wisdom_engine():
    print("Initializing Wisdom Engine...")
    engine = WisdomEngine(llm_client=MockGeminiClient())
    
    print("\nExecuting Pipeline on Mock Task: 'Refactor the React Native login flow for better speed'")
    report = await engine.execute_pipeline("Refactor the React Native login flow for better speed", SophisticationLevel.SYSTEMIC)
    
    print("\n--- Final Wisdom Report ---")
    for key, value in report.items():
        print(f"{key.upper()}: {value}")
        
    print(f"\nTotal Active Interrupts Handled: {report['interrupts_handled']}")
    if report['interrupts_handled'] == 0:
        print("[SUCCESS] Pipeline completed without unresolved simplification interrupts.")

if __name__ == "__main__":
    asyncio.run(test_wisdom_engine())
