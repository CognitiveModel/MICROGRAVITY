import json
import sys
import os

try:
    from coding_agent.storage.intent_store import IntentStore
    from coding_agent.utils.gemini_client import init_gemini, get_gemini_response
    from coding_agent.core.researcher import WebResearcher
    from coding_agent.core.crawler import WebCrawler
    from coding_agent.subagents.context import ContextSubagent
    from coding_agent.subagents.due_diligence import DueDiligenceSubagent
except ImportError:
    # Fallback
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from storage.intent_store import IntentStore
    from utils.gemini_client import init_gemini, get_gemini_response
    from core.researcher import WebResearcher
    from core.crawler import WebCrawler
    from subagents.context import ContextSubagent
    from subagents.due_diligence import DueDiligenceSubagent

class IntentSubagent:
    def __init__(self, intents_file="intents.json"):
        self.store = IntentStore(intents_file)
        self.gemini_model = init_gemini()
        self.researcher = WebResearcher()
        self.crawler = WebCrawler()
        self.context_engineer = ContextSubagent()
        self.due_diligence = DueDiligenceSubagent()
        self.actions = {
            "send_email": self.action_send_email,
            "get_weather": self.action_get_weather,
            "log_task": self.action_log_task,
            "web_research": self.action_web_research
        }

    def process_request(self, user_input):
        print(f"Processing: '{user_input}'")
        
        # 1. Context Engineering
        context_summary = self.context_engineer.engineer_context(user_input)
        print(context_summary)
        
        # 2. Due Diligence / Optimization Audit (New)
        if any(word in user_input.lower() for word in ["build", "design", "architecture", "develop", "implementation"]):
            print("Triggering Due Diligence Audit...")
            audit_report = self.due_diligence.analyze_plan(user_input, "Generic development request based on codebase current state.")
            return f"### 🛡️ Due Diligence Report\n\n{audit_report}"

        # 3. Search for existing intent
        match, score = self.store.find_similar_intent(user_input)
        
        if match and score > 0.7:
            print(f"Matched intent: {match['name']} (Score: {score:.2f})")
            return self.execute_intent(match, user_input)
        
        # 4. If no match, use LLM
        prompt = f"""
        User input: "{user_input}"
        Context: {context_summary}
        Intents: {", ".join([i['name'] for i in self.store.intents])}
        Return JSON format: {{"type": "new_intent"|"general", "name": "...", "description": "..."}}
        """
        
        llm_decision = get_gemini_response(self.gemini_model, prompt)
        try:
            clean_json = llm_decision.strip().replace("```json", "").replace("```", "").strip()
            decision = json.loads(clean_json)
            if decision.get("type") == "new_intent":
                self.store.add_intent(decision['name'], decision['description'])
                return f"I've learned a new intent: **{decision['name']}**.\nDescription: {decision['description']}"
            else:
                return get_gemini_response(self.gemini_model, f"Context:\n{context_summary}\n\nTask:\n{user_input}")
        except Exception:
            return get_gemini_response(self.gemini_model, f"Context:\n{context_summary}\n\nTask:\n{user_input}")

    def execute_intent(self, intent, original_input):
        intent_name = intent['name']
        action_meta = intent.get('action_meta', {})
        action_name = action_meta.get('action_name')
        if action_name in self.actions:
            return self.actions[action_name](intent, original_input)
        return get_gemini_response(self.gemini_model, f"Goal: {intent['description']}\nUser said: {original_input}")

    def action_send_email(self, intent, input_text):
        return f"Executing 'Send Email' action based on intent: {intent['name']}"

    def action_get_weather(self, intent, input_text):
        return f"Executing 'Get Weather' action based on intent: {intent['name']}"

    def action_log_task(self, intent, input_text):
        return f"Executing 'Log Task' action based on intent: {intent['name']}"

    def action_web_research(self, intent, input_text):
        print(f"Executing web research for: {input_text}")
        search_results = self.researcher.search(input_text, max_results=3)
        if not search_results:
            return "No information found."
        research_context = ""
        for r in search_results:
            content = self.researcher.scrape_url(r['href'])
            research_context += f"\nSource: {r['title']} ({r['href']})\nContent: {content[:2000]}\n"
        return get_gemini_response(self.gemini_model, f"Query: {input_text}\nData: {research_context}")

if __name__ == "__main__":
    agent = IntentSubagent("test_intents.json")
    print(agent.process_request("How is the researcher integrated?"))
