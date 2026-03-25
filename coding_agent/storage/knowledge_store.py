import os
import json
import time
import numpy as np
import google.generativeai as genai
from coding_agent.utils.config import GEMINI_API_KEY

class KnowledgeStore:
    def __init__(self, storage_file="knowledge.json"):
        if not os.path.isabs(storage_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_file = os.path.join(base_dir, storage_file)
        else:
            self.storage_file = storage_file
            
        self.knowledge_base = []
        self.load_knowledge()
        genai.configure(api_key=GEMINI_API_KEY)

    def load_knowledge(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)

    def save_knowledge(self):
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)

    def record_success(self, task, parameters):
        embedding = self.get_embedding(task, is_query=False)
        self.knowledge_base.append({
            "task": task,
            "parameters": parameters,
            "embedding": embedding,
            "timestamp": time.time()
        })
        self.save_knowledge()

    def get_embedding(self, text, is_query=False):
        task_type = "retrieval_query" if is_query else "retrieval_document"
        if is_query:
            result = genai.embed_content(model="models/text-embedding-004", content=text, task_type=task_type)
        else:
            result = genai.embed_content(model="models/text-embedding-004", content=text, task_type=task_type, title="Knowledge base entry")
        return result['embedding']

    def retrieve_relevant_knowledge(self, task, threshold=0.5):
        if not self.knowledge_base: return None
        query_embedding = self.get_embedding(task, is_query=True)
        best_match = None
        best_score = -1
        for entry in self.knowledge_base:
            score = np.dot(query_embedding, entry['embedding'])
            if score > best_score:
                best_score = score
                best_match = entry
        return best_match["parameters"] if best_score > threshold else None

    def retrieve(self, query, top_k=10):
        if not self.knowledge_base: return []
        query_embedding = self.get_embedding(query, is_query=True)
        scored_entries = []
        for entry in self.knowledge_base:
            score = np.dot(query_embedding, entry['embedding'])
            scored_entries.append((score, entry))
        
        # Sort by best score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Return the top_k entries directly (ExploratoryPatternSearcher expects object with 'content' or primitive string)
        # We wrapper dictionary so 'content' works
        class ResultWrapper:
            def __init__(self, data):
                self.content = data.get("parameters", str(data))
        
        return [ResultWrapper(item[1]) for item in scored_entries[:top_k]]

