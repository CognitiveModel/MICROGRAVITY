"""
Verification Script for Microgravity Evolution & Self-Awareness

Tests the system's ability to:
1. Log and retrieve User Nuances.
2. Track Architectural Debt.
3. Trigger the InternalAuditor for 'Fundamental Flaws'.
"""

import logging
import json
import asyncio
from coding_agent.storage.microgravity_tracker import MicrogravityTracker, Nuance, RoadmapItem
from coding_agent.subagents.internal_auditor import InternalAuditor
from coding_agent.core.agent import IntrospectionAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_microgravity_evolution():
    print("\n--- [TEST] VERIFYING MICROGRAVITY EVOLUTION & SELF-AWARENESS ---\n")
    
    # 1. Test Nuance & Debt Persistence
    print("[1/3] Testing Nuance & Debt Logging...")
    tracker = MicrogravityTracker()
    
    # Clear old test data for clean run
    tracker.nuances = []
    tracker.architectural_debt = []
    
    tracker.nuances.append(Nuance(
        description="User prefers 'Vibrant Blue' themes over 'Plain Gray'.",
        source="user",
        severity="info"
    ))
    tracker.architectural_debt.append("Synchronous execution in agent.py blocks concurrency.")
    tracker._save()
    
    # Re-load to check persistence
    new_tracker = MicrogravityTracker()
    print(f"Logged Nuances: {len(new_tracker.nuances)}")
    print(f"Logged Debt: {len(new_tracker.architectural_debt)}")
    assert len(new_tracker.nuances) > 0
    assert "Synchronous" in new_tracker.architectural_debt[0]

    # 2. Test Internal Auditor (Flaw Introspection)
    print("\n[2/3] Testing Internal Auditor Flaw Introspection...")
    auditor = InternalAuditor()
    feedback = "The sync/async bridge in agent.py is a fundamental flaw that will cause deadlocks!"
    report = auditor.introspect_flaw(feedback, "Current architecture uses asyncio.run inside sync methods.")
    
    print(f"Auditor Flaw Identified: {report['flaw_identified']}")
    print(f"Evolution Report Snippet: {report['evolution_report'][:100]}...")
    assert report["flaw_identified"] == True

    # 3. Test Agent Integration (Contextual Awareness)
    print("\n[3/3] Testing Agent Contextual Awareness of Nuances...")
    agent = IntrospectionAgent()
    # Check if nuances appear in context
    ctx = agent.microgravity.get_microgravity_context()
    print(f"Microgravity context provided to LLM: {ctx}")
    assert "Vibrant Blue" in ctx
    assert "Synchronous execution" in ctx

    print("\n--- [SUCCESS] MICROGRAVITY EVOLUTION SYSTEM OPERATIONAL ---\n")

if __name__ == "__main__":
    asyncio.run(test_microgravity_evolution())
