import json
import traceback
import sys
import os
from pathlib import Path

from coding_agent.utils.gemini_client import init_gemini, get_gemini_response
from coding_agent.subagents.context import ContextSubagent
from coding_agent.subagents.due_diligence import Due_diligence_subagent
from coding_agent.storage.knowledge_store import KnowledgeStore
from coding_agent.core.task_model import TaskModel, TaskStatus
from coding_agent.core.resource_manager import ResourceManager
from coding_agent.subagents.identity import IdentitySubagent
from coding_agent.subagents.ethics import EthicsSubagent
from coding_agent.subagents.method_learner import MethodLearner
from coding_agent.core.outcome_mapper import OutcomeMapper
from coding_agent.core.exploratory_search import ExploratoryPatternSearcher
from coding_agent.core.wisdom_engine import WisdomEngine, SophisticationLevel
from coding_agent.core.soul_identity import SoulIdentity
from coding_agent.storage.world_model import WorldModel
from coding_agent.storage.microgravity_tracker import MicrogravityTracker
from coding_agent.core.bus import MessageBus, Message
from coding_agent.core.gateway import ChannelManager
from coding_agent.core.channels.telegram import TelegramChannel
from coding_agent.subagents.internal_auditor import InternalAuditor
from coding_agent.master_swarm.swarm_manager import SubagentManager
from coding_agent.subagents.hitl_agent import HumanInTheLoopAgent
from coding_agent.core.db_router import DatabaseRouter
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)



class IntrospectionAgent:
    def __init__(self, max_retries=5):
        self.gemini_model = init_gemini()
        self.context_engineer = ContextSubagent()
        self.due_diligence = Due_diligence_subagent()
        self.knowledge_store = KnowledgeStore()
        self.task_model = TaskModel()
        self.resource_manager = ResourceManager()
        self.identity = IdentitySubagent()
        self.ethics = EthicsSubagent()
        self.method_learner = MethodLearner()
        self.outcome_mapper = OutcomeMapper()
        self.db_router = DatabaseRouter()
        self.max_retries = max_retries
        
        # Initialize Exploratory Search Skill
        self.exploratory_search = ExploratoryPatternSearcher(
            search_engine=self.knowledge_store,
            embedding_calculator=lambda text: self.knowledge_store.get_embedding(text, is_query=True)
        )
        
        # Initialize Metacognitive Wisdom Engine
        self.wisdom_engine = WisdomEngine()
        
        # Initialize Soul Identity, World Model, and Microgravity System
        self.soul = SoulIdentity()
        self.world_model = WorldModel()
        self.microgravity = MicrogravityTracker()
        self.internal_auditor = InternalAuditor()
        
        # Swarm Orchestration
        self.swarm = SubagentManager(self.gemini_model, Path(os.getcwd()))
        
        # Communication Layer
        self.bus = MessageBus()
        self.gateway = ChannelManager(self.bus)
        
        # Initialize Telegram Channel
        self.telegram = TelegramChannel(self.bus)
        self.gateway.add_channel("telegram", self.telegram)
        
        # Initialize HITL Agent
        self.hitl = HumanInTheLoopAgent(self.bus)
        
        self.gateway_task: Optional[asyncio.Task] = None
        self._start_gateway()

    def _start_gateway(self):
        """Starts the gateway dispatcher in the background."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.gateway_task = loop.create_task(self.gateway.start_all())
                loop.create_task(self.hitl.start_listening())
                # Start listening for inbound messages to process
                loop.create_task(self._listen_inbound())
            else:
                # If no loop is running, we might be in a sync context that calls run() later
                pass
        except RuntimeError:
            pass

    def restart_gateway(self):
        """Restarts the gateway dispatcher."""
        print("\n[GATEWAY] Restarting Gateway...")
        if self.gateway_task:
            self.gateway_task.cancel()
        
        self.gateway = ChannelManager(self.bus)
        self._start_gateway()
        print("[GATEWAY] Gateway Restarted.")

    def run(self, task_description, role="Generalist Coding Assistant"):
        print(f"\n[START] Starting Multi-Layered Introspection Agent")
        self.identity.assume_role(role)
        conflicts = self.resource_manager.analyze_sharing_conflict(task_description)
        conflict_context = f"\nResource Conflicts: {', '.join(conflicts)}" if conflicts else ""
        goal = self.task_model.create_goal(f"Resolve: {task_description[:50]}", task_description)
        ethics_audit = self.ethics.analyze_ethics(task_description)
        domain = self.detect_domain(task_description)
        learned_methods = self.method_learner.get_domain_methods(domain)
        method_context = f"\nMethods for {domain}: {json.dumps(learned_methods)}" if learned_methods else ""
        
        # Get files for context engineering
        try:
            files_available = os.listdir(os.getcwd())
        except:
            files_available = []
            
        context_summary = self.context_engineer.engineer_context(task_description, files_available)
        raw_learned_params = self.knowledge_store.retrieve_relevant_knowledge(task_description)
        learned_params_context = self.db_router.build_context_payload(domain, raw_learned_params)
        
        # --- WORLD MODEL STRUCTURAL CONGRUENCE & NOVELTY ---
        analogous_strategies = self.world_model.find_analogous_strategies(task_description.split())
        
        task_uid = goal.id
        is_novel = False
        if not analogous_strategies:
            import uuid
            task_uid = f"NOVEL_{uuid.uuid4().hex[:8]}"
            is_novel = True
            print(f"\n[DISCOVERY] No historical analogies found. Generated NOVEL Task ID: {task_uid}")
            analogy_context = "\n[DISCOVERY MODE ON] No analogous strategies exist in the World Model. Synthesize a first-principles approach."
        else:
            analogy_context = f"\nStructural Analogies: {', '.join(analogous_strategies)}"

        
        # --- EXPLORATORY SKILL BRIDGE ---
        exploratory_context = ""
        task_lower = task_description.lower()
        if any(kw in task_lower for kw in ["explore", "pattern", "architecture", "hypothesis"]):
            print("\n[SKILL] Exploratory MMR Search Triggered!")
            diverse_findings = self.exploratory_search.test_design_hypothesis(
                pattern_description=task_description,
                limit=10,
                return_k=3
            )
            if diverse_findings:
                # Format the diverse findings for the prompt
                findings_str = "\\n".join([f"- {str(f)}" for f in diverse_findings])
                exploratory_context = f"\n--- Exploratory Search Findings (Dynamically Diversified) ---\n{findings_str}"
        # --------------------------------
        
        # --- WISDOM METACOGNITIVE PASS ---
        wisdom_context = ""
        if any(kw in task_lower for kw in ["architect", "strategy", "complex", "refactor", "design"]):
            print("\n[METACOGNITION] High-stakes task detected. Running Wisdom Pipeline...")
            # In a real sync/async bridged environment we'd use asyncio.run but assuming the loop allows this or we mock it for now.
            # To protect the synchronous flow, we mock the async call if we have no valid loop, but for now we expect async awareness at the top level or we wrap it properly.
            try:
                loop = asyncio.get_event_loop()
                wisdom_report = loop.run_until_complete(
                    self.wisdom_engine.execute_pipeline(task_description, SophisticationLevel.SYSTEMIC)
                )
            except RuntimeError:
                wisdom_report = asyncio.run(self.wisdom_engine.execute_pipeline(task_description, SophisticationLevel.SYSTEMIC))
                
            wisdom_context = (
                f"\n--- 7R Wisdom Pipeline Insights ---\n"
                f"R1 Entity Map: {wisdom_report.get('r1_entity_map')}\n"
                f"R3 Named Intent: {wisdom_report.get('r3_named_intent')}\n"
                f"R4 Constraints: {wisdom_report.get('r4_constraints')}\n"
                f"R5 Operator Plan: {wisdom_report.get('r5_operator_plan')}\n"
                f"R6 Conviction Grade: {wisdom_report.get('hardness_grade')}/10\n"
            )
            
            # TRIGGER ENGINE: Semantic Drift 
            if wisdom_report.get('hardness_grade', 10) < 7:
                 print("\n[TRIGGER ENGINE] Anomaly: Semantic Drift Detected! Wisdom Conviction low.")
                 wisdom_context += "\nWARNING: R6 Conviction is too tentative. Apply heightened scrutiny."
        # ---------------------------------
        
        # Publish start event to bus
        asyncio.run(self.bus.publish(Message(
            content=f"Starting task: {task_description[:50]}...",
            metadata={"is_progress": True}
        )))
        
        microgravity_ctx = self.microgravity.get_microgravity_context()
        initial_plan_prompt = (
            f"Identity: {self.identity.get_identity_context()}\n"
            f"Soul Traits: {self.soul.traits}\n"
            f"Hardened Features: {self.soul.traits.hardened_wisdom_features}\n"
            f"Task: {task_description}\n"
            f"Context: {context_summary}\n"
            f"{learned_params_context}\n"
            f"{microgravity_ctx}\n{conflict_context}\n{method_context}\n{analogy_context}\n"
            f"{exploratory_context}\n{wisdom_context}\n"
            "SYSTEM NOTE: Refer to FEATURE_MANIFEST.md for details on Task ID, Path ID, and Error Distillation.\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the situational context and exploratory findings.\n"
            "2. If this task requires UI automation or real-time desktop interaction, prepare a Swarm delegation request.\n"
            "   Example: 'DELEGATE_TO_SWARM: Navigate to web.telegram.org and send a message...'\n"
            "3. Generate a ROBUST step-by-step plan starting with 'PLAN:'\n"
        )
        initial_plan = get_gemini_response(self.gemini_model, initial_plan_prompt)
        audit_report = self.due_diligence.analyze_plan(task_description, initial_plan)
        return self.recursive_execute(task_uid, task_description, context_summary, learned_params_context, audit_report, domain)

    def detect_domain(self, task):
        t = task.lower()
        if "medical" in t: return "Medical"
        if "legal" in t: return "Legal"
        if "finance" in t: return "Financial"
        return "Coding"

    def recursive_execute(self, task_id, task, context, initial_knowledge, audit, domain, attempt=1, last_response=None):
        if attempt > self.max_retries: 
            return f"[FAILURE] Max retries reached.\nLast Raw Output: {last_response}"
        asyncio.run(self.bus.publish(Message(
            content=f"Attempt {attempt}/{self.max_retries}",
            metadata={"is_progress": True}
        )))
        predicted_outcome = self.outcome_mapper.map_behavior_to_outcome(f"Perception: Attempt {attempt}", "Behavior: Standard Execution")
        print(f"[MAPPER] Predicted Outcome: {predicted_outcome}")
        
        exec_prompt = (
            f"Identity: {self.identity.get_identity_context()}\n"
            f"Domain: {domain}\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Audit/Plan: {audit}\n"
            f"Attempt: {attempt}/{self.max_retries}\n\n"
            "ROLE: You are the recursive executor. Your goal is to move the task toward completion.\n\n"
            "INSTRUCTIONS:\n"
            "- If the task requires UI automation (e.g. Browser, Telegram, Screen), move immediately to FINAL_RESULT with DELEGATE_TO_SWARM.\n"
            "- If you encounter an error you can't fix, output 'ERROR: <specific technical reason>'.\n"
            "- If the task is finished, output 'FINAL_RESULT: <clear summary of success>'.\n"
            "\nOutput STRICTLY starting with 'ERROR:' or 'FINAL_RESULT:'"
        )
        response = get_gemini_response(self.gemini_model, exec_prompt, retries=5)
        print(f"\n[AGENT] Raw Gemini Response:\n{response}\n")
        
        if "ERROR:" in response:
            error = response.split("ERROR:")[1].strip()
            
            # --- FLAW INTROSPECTION TRIGGER ---
            if "flaw" in error.lower() or "fundamental" in error.lower():
                print("\n[AUDIT] Potential Fundamental Flaw detected! Pivoing iteration...")
                pivot_report = self.internal_auditor.introspect_flaw(error, self.microgravity.get_microgravity_context())
                error = f"{error}\nPIVOT STRATEGY: {pivot_report.get('evolution_report')}"
            # ----------------------------------
            
            self.outcome_mapper.log_actual_outcome(f"Perception: Attempt {attempt}", "Behavior: Standard Execution", "Outcome: Error")
            self.world_model.record_run(domain, task, "Standard Task Execution", False)
            correction = get_gemini_response(self.gemini_model, f"Previous Error: {error}\nFix it.")
            return self.recursive_execute(task_id, task, context, initial_knowledge, f"Error: {error}\nCorrection: {correction}", domain, attempt + 1, last_response=response)
        elif "FINAL_RESULT:" in response:
            result = response.split("FINAL_RESULT:")[1].strip()
            
            # --- SWARM DELEGATION TRIGGER ---
            # Permissive check: if DELEGATE_TO_SWARM is mentioned anywhere in the result
            if "DELEGATE_TO_SWARM" in result:
                # Extract task: remove FINAL_RESULT, DELEGATE_TO_SWARM, and any code blocks
                import re
                task_to_delegate = result
                # Remove FINAL_RESULT prefix
                task_to_delegate = re.sub(r'^FINAL_RESULT:\s*', '', task_to_delegate, flags=re.IGNORECASE)
                # Handle DELEGATE_TO_SWARM format
                if "DELEGATE_TO_SWARM:" in task_to_delegate:
                    task_to_delegate = task_to_delegate.split("DELEGATE_TO_SWARM:")[1].strip()
                elif "DELEGATE_TO_SWARM" in task_to_delegate:
                    # If it's just the keyword, the description is likely meta-commentary
                    # e.g. "I am delegating... DELEGATE_TO_SWARM."
                    task_to_delegate = task_to_delegate.replace("DELEGATE_TO_SWARM", "").strip()
                
                # Strip markdown code blocks if any
                task_to_delegate = re.sub(r'```[\s\S]*?```', '', task_to_delegate).strip()
                
                # If there's a (task="...") pattern, extract the inner string
                task_match = re.search(r'task\s*=\s*["\'](.*?)["\']', task_to_delegate)
                if task_match:
                    task_to_delegate = task_match.group(1)
                
                # CRITICAL: If the extracted string contains "initial plan", "heavily relies", 
                # or is basically just the LLM talking about delegation, use the ORIGINAL task.
                meta_indicators = ["initial plan", "relies heavily", "delegating to", "requires ui automation"]
                is_meta = any(ind in task_to_delegate.lower() for ind in meta_indicators)

                if not task_to_delegate or len(task_to_delegate) < 10 or is_meta:
                    print(f"[AGENT] Extracted task looks like meta-commentary ('{task_to_delegate[:50]}...'). Reverting to original task.")
                    task_to_delegate = task
                
                print(f"\n[AGENT] Triggering Swarm for: {task_to_delegate}")
                swarm_msg = self.swarm.spawn_sync(task_to_delegate)
                return f"{result}\n\n🚀 {swarm_msg}"
                result = f"{result}\n\n🚀 {swarm_msg}"
            # --------------------------------
            
            self.outcome_mapper.log_actual_outcome(f"Perception: Attempt {attempt}", "Behavior: Standard Execution", "Outcome: Success")
            self.learn_from_success(task, response)
            self.world_model.record_run(domain, task, "Standard Task Execution", True)
            print(f"[END] Introspection Complete.\n")
            return result
        return self.recursive_execute(task_id, task, context, initial_knowledge, "Ambiguous result.", domain, attempt + 1)

    async def _listen_inbound(self):
        """Listens for and processes inbound messages from the bus."""
        print("[AGENT] Listener: Online and waiting for inbound messages...", flush=True)
        async for msg in self.bus.subscribe(direction="inbound"):
            print(f"[AGENT] Received Inbound via {msg.channel}: {msg.content}", flush=True)
            
            if msg.message_type == "INTERRUPT":
                target_task_id = msg.metadata.get("target_task_id")
                if target_task_id:
                    print(f"[AGENT] PREEMPTION: Received INTERRUPT for Task {target_task_id}")
                    self.swarm.coordinator.cancel_task(target_task_id)
                    await self.bus.publish(Message(
                        content=f"⚠️ Preemption: Task {target_task_id} cancellation signal dispatched.",
                        channel=msg.channel,
                        metadata={"chat_id": msg.metadata.get("chat_id")}
                    ))
                continue
                
            asyncio.create_task(self.process_message(msg))

    async def process_message(self, msg: Message):
        """Processes a single inbound message by running the full recursive iteration."""
        print(f"[AGENT] Processing Inbound Autonomous Task: {msg.content}", flush=True)
        
        # 1. Provide initial progress feedback
        await self.bus.publish(Message(
            content=f"🧠 Microgravity is engaging: {msg.content[:60]}...",
            channel=msg.channel,
            metadata={"chat_id": msg.metadata.get("chat_id"), "_progress": True}
        ))
        
        # 2. Run the full introspection pipeline in an executor to avoid blocking the bus
        # Since run() is synchronous, we use run_in_executor
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.run, msg.content)
            
            # 3. Publish the final result
            await self.bus.publish(Message(
                content=str(result),
                channel=msg.channel,
                metadata={"chat_id": msg.metadata.get("chat_id")}
            ))
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            traceback.print_exc()
            await self.bus.publish(Message(
                content=f"❌ Task Execution Failed: {str(e)}",
                channel=msg.channel,
                metadata={"chat_id": msg.metadata.get("chat_id")}
            ))

    def learn_from_success(self, task, successful_response):
        pass
