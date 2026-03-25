import os
import json
import numpy as np
import google.generativeai as genai
try:
    from coding_agent.utils.config import GEMINI_API_KEY
except ImportError:
    from utils.config import GEMINI_API_KEY

class IntentStore:
    def __init__(self, storage_file="intents.json"):
        if not os.path.isabs(storage_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_file = os.path.join(base_dir, storage_file)
        else:
            self.storage_file = storage_file
        self.intents = []
        self.load_intents()
        genai.configure(api_key=GEMINI_API_KEY)

    def load_intents(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.intents = json.load(f)

    def add_intent(self, name, description, action_meta=None):
        embedding = self.get_embedding(f"{name}: {description}")
        self.intents.append({
            "name": name,
            "description": description,
            "embedding": embedding,
            "action_meta": action_meta or {}
        })
        self.save_intents()

    def get_embedding(self, text):
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return result['embedding']

    def save_intents(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.intents, f, indent=2)

    def find_similar_intent(self, query):
        if not self.intents: return None, 0
        query_embedding = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")['embedding']
        best_match = None
        best_score = -1
        for intent in self.intents:
            score = np.dot(query_embedding, intent['embedding'])
            if score > best_score:
                best_score = score
                best_match = intent
        return best_match, best_score
