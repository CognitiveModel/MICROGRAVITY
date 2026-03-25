import json
from typing import Dict, List, Any

class MethodLearner:
    def __init__(self):
        # Dictionary of domains and their learned methods
        self.learned_methods = {
            "coding": {
                "name": "Recursive Introspection",
                "attributes": ["Efficient", "Scalable", "Robust", "Creative"]
            },
            "research": {
                "name": "Recursive Search depth=3",
                "attributes": ["Thorough", "Scalable"]
            }
        }
        self.domains = ["Medical", "Legal", "Financial", "Scientific", "Business", "Government"]

    def learn_method(self, domain: str, method_name: str, attributes: List[str]):
        """Records a new learned method for a specific domain."""
        if domain not in self.learned_methods:
            self.learned_methods[domain] = []
        
        self.learned_methods[domain].append({
            "name": method_name,
            "attributes": attributes
        })
        print(f"[METHOD] Learned new {domain} method: {method_name}")

    def get_domain_methods(self, domain: str) -> List[Dict[str, Any]]:
        """Retrieves learned methods for a specific domain."""
        return self.learned_methods.get(domain, [])

    def get_method_summary(self) -> str:
        """Returns a summary of all learned methods across domains."""
        return json.dumps(self.learned_methods, indent=2)

if __name__ == "__main__":
    ml = MethodLearner()
    ml.learn_method("Legal", "Contract Analysis v2", ["Reliable", "Security", "Interoperability"])
    print(ml.get_method_summary())
