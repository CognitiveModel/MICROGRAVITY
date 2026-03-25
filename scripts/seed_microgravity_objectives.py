"""
Microgravity Objectives Seeder

Populates the Case Study DB with initial high-ambition objectives, 
extracting essence keywords for hybrid search optimization.
"""

import json
from coding_agent.storage.case_study_db import CaseStudyDB
from coding_agent.subagents.keyword_extractor import KeywordExtractorSubagent

def seed():
    db = CaseStudyDB()
    extractor = KeywordExtractorSubagent()
    
    objectives = [
        {
            "domain": "cybersecurity",
            "title": "Autonomous Penetration Testing Swarm",
            "description": "A distributed subagent network that identifies and patches kernel-level vulnerabilities in real-time.",
            "monetization": "SaaS Subscription for high-sec enterprise.",
            "resources": "50x H100 GPU cluster for inference scaling.",
            "calculation": "Estimated ROI 400% after 12 months."
        },
        {
            "domain": "networking",
            "title": "Quantum-Resistant Mesh Protocol",
            "description": "A decentralised networking protocol using lattice-based cryptography for post-quantum safety.",
            "monetization": "Infrastructure licensing and hardware integration.",
            "resources": "Geographically distributed edge nodes.",
            "calculation": "Complexity O(N log N) for routing convergence."
        },
        {
            "domain": "commerce",
            "title": "Predictive Supply Chain Oracle",
            "description": "Algorithmic forecasting of global shipping trends to optimize port allocation.",
            "monetization": "0.1% transaction fee on optimized shipments.",
            "resources": "Direct API access to port logistics telemetry.",
            "calculation": "High-complexity Bayesian inference models."
        }
    ]

    for obj in objectives:
        print(f"Propagating {obj['title']}...")
        # Extract keywords for the 'Case Constitution'
        essence_raw = f"{obj['title']} {obj['description']} {obj['domain']}"
        keywords = extractor.extract_keywords(essence_raw)
        
        obj["keywords"] = keywords
        db.save_case(obj["domain"], obj)

if __name__ == "__main__":
    seed()
