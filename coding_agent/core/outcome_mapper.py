import json
from typing import Dict, List, Any

class OutcomeMapper:
    def __init__(self):
        # A map of perceptions -> behaviors -> predicted outcomes
        self.experience_map = {
            "Perception: Error encountered": {
                "Behavior: Recursive Retry": "Outcome: Resolution (85% success)",
                "Behavior: Skip": "Outcome: Failure (100%)"
            }
        }

    def map_behavior_to_outcome(self, perception: str, behavior: str) -> str:
        """Attempts to predict the outcome of a behavior based on perception."""
        if perception in self.experience_map:
            return self.experience_map[perception].get(behavior, "Outcome: Unknown")
        return "Outcome: Unknown (Initial Perception Case)"

    def log_actual_outcome(self, perception: str, behavior: str, outcome: str):
        """Logs the actual outcome to refine the experience map."""
        if perception not in self.experience_map:
            self.experience_map[perception] = {}
        self.experience_map[perception][behavior] = outcome
        print(f"[MAPPER] Logged Experience: {perception} -> {behavior} -> {outcome}")

    def get_mapper_summary(self) -> str:
        """Returns a summary of the experience map."""
        return json.dumps(self.experience_map, indent=2)

if __name__ == "__main__":
    om = OutcomeMapper()
    print(om.map_behavior_to_outcome("Perception: Error encountered", "Behavior: Recursive Retry"))
    om.log_actual_outcome("Perception: Ambiguous user request", "Behavior: Ask Clarification", "Outcome: Goal Alignment (95%)")
    print(om.get_mapper_summary())
