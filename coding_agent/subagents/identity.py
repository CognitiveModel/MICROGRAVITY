import json
from typing import Dict, List, Any

class IdentitySubagent:
    def __init__(self):
        self.capabilities = {
            "technical": ["Python", "Gemini API", "File System", "Browser Interaction", "JSON Parsing"],
            "metacognitive": [
                "Introspection", "Recursive Retrying", "Knowledge Retrieval", "Task Modeling",
                "Deep-Labeled Task ID Extraction", "Multi-Path Optimization (PathID)", 
                "Autonomous Error/Mistake Distillation"
            ],
            "experiential": [
                "UI Action Parameterization", "Process Constant Deduction", 
                "Process Variable Identification", "Experiential Schema Forking"
            ],
            "communication": ["Artifact Creation", "User Notification", "Markdown Formatting"]
        }
        self.ambition_level = 0.8 # 0 to 1 scale
        self.current_role = "Generalist Coding Assistant"
        self.growth_goals = ["Deep Methodical Learning", "Ethical Alignment", "Cross-Domain Expertise"]

    def assess_capability(self, task_requirement: str) -> bool:
        """Checks if the agent has the technical capability for a requirement."""
        for category in self.capabilities.values():
            if any(cap.lower() in task_requirement.lower() for cap in category):
                return True
        return False

    def assume_role(self, role_name: str):
        """Adjusts the agent's identity and 'attitude' based on a role."""
        prev_role = self.current_role
        self.current_role = role_name
        print(f"[IDENTITY] Role switch: {prev_role} -> {self.current_role}")

    def get_identity_context(self) -> str:
        """Returns a string description of the agent's current identity and ambition."""
        return (f"Role: {self.current_role}\n"
                f"Ambition Level: {self.ambition_level}\n"
                f"Growth Goals: {', '.join(self.growth_goals)}\n"
                f"Known Capabilities: {json.dumps(self.capabilities)}")

if __name__ == "__main__":
    ids = IdentitySubagent()
    print(ids.get_identity_context())
    ids.assume_role("Lead Systems Architect")
    print(f"Can handle SQL? {ids.assess_capability('Design a complex SQL database')}")
