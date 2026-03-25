"""
ExperientialMemory — 4-tier hierarchical learning system persisted across sessions.

Hierarchy:
  global/               → Cross-app universal knowledge
  app_classes/           → Per-app-type knowledge (BROWSER, EDITOR, CHAT, etc.)
  app_instances/         → Per-app knowledge (Chrome, VSCode, Notepad)
    └─ site_specific/    → Per-website / per-project knowledge (auto-expanding)

Tiers (at each hierarchy level):
  1. EpisodicMemory: Records full execution episodes (what happened)
  2. HypothesisMemory: Builds testable if-then hypotheses (why things happen)
  3. ProcessMemory: Stores proven reusable action sequences
  4. NuanceLedger: Records subtle edge cases and quirks
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from coding_agent.core.experiential_storage import ExperientialStorage
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

from coding_agent.ui_agent.planning.task_schema import (
    ParameterizedTaskSchema, ProcessConstant, ProcessVariable, 
    ActionNode, BranchNode, IterableNode, NodeType, SchemaDifferentiation,
    ErrorRecord, TaskPath
)
from coding_agent.subagents.path_analyzer_agent import PathAnalyzerAgent


# ──────────────────────────  Data Structures  ──────────────────────────

class ComplexityTier(str, Enum):
    SIMPLE = "SIMPLE"      # 1-2 steps, linear
    COMPOUND = "COMPOUND"  # 3-5 steps, deterministic
    COMPLEX = "COMPLEX"    # 5+ steps, branches, or nuances
    NAVIGATION = "NAVIGATION" # Intent is state/URL shift

class PatternTier(str, Enum):
    STATIC = "STATIC"          # Fixed structure and content (e.g. system dialog)
    DYNAMIC = "DYNAMIC"        # Fixed structure, changing content (e.g. search results)
    ADAPTIVE = "ADAPTIVE"      # Layout itself adapts/evolves (e.g. collapsible panels)

@dataclass
class PatternElement:
    """A member of a UI pattern."""
    label: str
    element_type: str
    rel_x: float               # Relative to pattern anchor (top-left)
    rel_y: float
    optional: bool = False

@dataclass
class UIPattern:
    """A recurring structural UI bundle (modal, sidebar, etc.)."""
    pattern_id: str
    label: str
    tier: PatternTier
    app_class: str             # Empty for global patterns
    elements: List[PatternElement]
    signature: List[str]       # Sorted unique element types/labels for fast matching
    
    success_count: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=time.time)
    
    def matches_elements(self, detected_labels: List[str]) -> float:
        """Returns match confidence [0-1] based on signature."""
        target = set(self.signature)
        found = set(detected_labels)
        intersection = target.intersection(found)
        if not target: return 0.0
        return len(intersection) / len(target)

@dataclass
class Episode:
    """A single recorded execution episode."""
    episode_id: str
    task: str
    app_name: str
    app_class: str
    steps: List[Dict[str, Any]]     # [{action, target, result, timestamp, screenshot_hash}]
    success: bool
    failure_reason: str = ""
    duration_s: float = 0.0
    timestamp: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    complexity_tier: str = "SIMPLE"  # ComplexityTier value


@dataclass
class Hypothesis:
    """A testable if-then hypothesis about UI behavior."""
    hypothesis_id: str
    condition: str                  # "Clicking 'Save' when file is new"
    prediction: str                 # "Opens 'Save As' dialog"
    app_class: str
    evidence_for: int = 0
    evidence_against: int = 0
    confidence: float = 0.0        # evidence_for / (evidence_for + evidence_against + 1)
    is_fact: bool = False           # Promoted when confidence > 0.9 after 5+ tests
    created: float = 0.0
    last_tested: float = 0.0


@dataclass
class ProcessIP:
    """A reusable action sequence (Interaction Pattern)."""
    process_id: str
    task_pattern: str               # "open_file_in_browser", "save_document", etc.
    app_class: str
    steps: List[Dict[str, Any]]     # [{action, target_desc, method}]
    run_count: int = 0
    success_count: int = 0
    category: str = "RARE"          # TYPICAL (>5 runs), SPECIAL (2-5), GENERAL (cross-app), RARE (1)
    complexity_tier: str = "SIMPLE" # ComplexityTier value
    is_navigation: bool = False
    created: float = 0.0
    last_used: float = 0.0


@dataclass
class Nuance:
    """A subtle non-obvious behavior record."""
    nuance_id: str
    app_class: str
    element_id: str
    nuance_type: str                # TIMING | MODAL | STATE_DEPENDENT | MULTI_STEP | PLATFORM_QUIRK | RESOURCE
    severity: str                   # CRITICAL | IMPORTANT | INFORMATIONAL
    description: str
    workaround: str = ""
    trigger_condition: str = ""
    created: float = 0.0
    occurrence_count: int = 1


@dataclass
class NavigationRoute:
    """A learned route between application states or URLs."""
    route_id: str
    app_class: str
    start_state: str                # e.g., "google.com/search"
    end_state: str                  # e.g., "google.com/finance"
    steps: List[Dict[str, Any]]     # Steps taken for this specific transition
    success_count: int = 1
    total_count: int = 1
    avg_duration_s: float = 0.0
    last_used: float = 0.0

@dataclass
class UserProfile:
    """Tracks known accounts, behavioral patterns, and agent assumptions about the user."""
    profile_id: str = "default_user"
    known_accounts: Dict[str, Dict[str, str]] = field(default_factory=dict) # e.g. "google": {"email": "test@gmail.com"}
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

@dataclass
class DeviceProfile:
    """Tracks physical and OS environment facts."""
    device_id: str = "default_device"
    os_info: str = "Windows"
    display_resolution: str = "Unknown"
    file_system_anchors: Dict[str, str] = field(default_factory=dict) # e.g. "Downloads": "C:\\Users\\...\\Downloads"
    last_updated: float = field(default_factory=time.time)

@dataclass
class AdaptiveAnchor:
    """Tracks frequently used UI elements with a confidence decay mechanism."""
    element_id: str
    last_known_coords: List[int] # [x, y]
    confidence: float = 1.0 # 0.0 to 1.0
    variations: List[List[int]] = field(default_factory=list)
    app_context: str = ""
    last_used: float = field(default_factory=time.time)
    
    def decay(self, days_elapsed: float) -> None:
        """Decrease confidence slightly if not used recently."""
        decay_rate = 0.05 * days_elapsed
        self.confidence = max(0.0, self.confidence - decay_rate)

# ──────────────────────────  ExperientialMemory  ──────────────────────────


class ExperientialMemory:
    """4-tier hierarchical learning system persisted to disk.
    
    Hierarchy: global → app_class → app_instance → site/context.
    Every app, website, or context auto-creates its own branch.
    """

    def __init__(self, storage_dir: str = "experiential_memory"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.exp_storage = ExperientialStorage()

        # ── Flat stores (backward-compatible) ──
        self.episodes: List[Episode] = []
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.processes: Dict[str, ProcessIP] = {}
        self.nuances: Dict[str, Nuance] = {}
        self.routes: Dict[str, NavigationRoute] = {}
        self.patterns: Dict[str, UIPattern] = {}
        self.task_schemas: Dict[str, ParameterizedTaskSchema] = {}
        self.adaptive_anchors: Dict[str, AdaptiveAnchor] = {}
        self.strategy_stats: Dict[str, Dict[str, Any]] = {}  # Persisted StrategySelector state
        
        # ── Context Profiles ──
        self.user_profile: UserProfile = UserProfile()
        self.device_profile: DeviceProfile = DeviceProfile()

        # ── Hierarchical tree ──
        # Structure: {level: {scope_key: {tier: [items]}}}
        # level = "global" | "app_class" | "app_instance" | "site"
        # scope_key = "*" for global, "BROWSER" for app_class, "Chrome" for instance, "reddit.com" for site
        self._hierarchy: Dict[str, Dict[str, Dict[str, list]]] = {
            "global": {"*": {"hypotheses": [], "processes": [], "nuances": [], "routes": []}},
            "app_class": {},
            "app_instance": {},
            "site": {},
        }

        # ── Indices ──
        self._app_episodes: Dict[str, List[int]] = defaultdict(list)
        self._task_processes: Dict[str, List[str]] = defaultdict(list)

        # ── Cross-app promotion tracking ──
        # hypothesis_id → set of app_classes where it was confirmed
        self._cross_app_confirmations: Dict[str, set] = defaultdict(set)
        self._promotion_threshold = 3  # Promote to global after confirmed in N apps

        # ── Loop incident records (for future avoidance) ──
        self.loop_incidents: List[Dict[str, Any]] = []

        # Load persisted data
        self._load()
        print(f"[ExperientialMemory] Loaded: {len(self.episodes)} episodes, {len(self.hypotheses)} hypotheses, "
              f"{len(self.processes)} processes, {len(self.nuances)} nuances")
        print(f"[ExperientialMemory] Hierarchy: "
              f"{len(self._hierarchy['app_class'])} app classes, "
              f"{len(self._hierarchy['app_instance'])} app instances, "
              f"{len(self._hierarchy['site'])} sites")

    # ═══════════════════════  Tier 1: Episodic Memory  ═══════════════════════

    def record_episode(self, task: str, app_name: str, app_class: str,
                       steps: List[Dict[str, Any]], success: bool, failure_reason: str = "",
                       context: Optional[Dict[str, Any]] = None) -> str:
        """Records a full execution episode."""
        episode_id = f"ep_{int(time.time())}_{len(self.episodes)}"
        ep = Episode(
            episode_id=episode_id,
            task=task,
            app_name=app_name,
            app_class=app_class,
            steps=steps,
            success=success,
            failure_reason=failure_reason,
            duration_s=sum(s.get("duration", 0) for s in steps),
            timestamp=time.time(),
            context=context or {},
        )
        # Phase 12: Autonomous Complexity Tagging
        step_count = len(steps)
        if step_count >= 6:
            ep.complexity_tier = ComplexityTier.COMPLEX.value
        elif step_count >= 3:
            ep.complexity_tier = ComplexityTier.COMPOUND.value
        else:
            ep.complexity_tier = ComplexityTier.SIMPLE.value

        self.episodes.append(ep)
        idx = max(0, len(self.episodes) - 1)
        self._app_episodes[app_class].append(idx)

        # Auto-extract process if successful
        if success and len(steps) >= 2:
            self._maybe_extract_process(ep)
        
        # Phase 12: Auto-detect navigation intent
        if success:
             self._maybe_extract_navigation_route(ep)

        # Auto-extract nuances from failures
        if not success:
            self._extract_failure_nuance(ep)

        self._save()
        return episode_id

    def recall_similar(self, task: str, app_class: str = "", top_k: int = 3) -> List[Episode]:
        """Retrieves past episodes with similar tasks using semantic string matching."""
        import difflib
        task_lower = task.lower()
        scored = []

        for ep in self.episodes:
            # Semantic string similarity using SequenceMatcher
            ep_task_lower = ep.task.lower()
            # ratio() returns a float in [0, 1] measuring the sequence similarity
            score = difflib.SequenceMatcher(None, ep_task_lower, task_lower).ratio()
            
            # Boost for same app
            if app_class and ep.app_class == app_class:
                score += 0.3

            # Boost for success
            if ep.success:
                score += 0.2

            if score > 0.4: # Require a decent baseline similarity before returning
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Type-safe slice using list comprehension
        res = [scored[i][1] for i in range(len(scored)) if i < top_k]  # type: ignore
        return res

    def get_app_episodes(self, app_class: str) -> List[Episode]:
        """Gets all episodes for a specific app."""
        indices = self._app_episodes.get(app_class, [])
        return [self.episodes[i] for i in indices if i < len(self.episodes)]

    # ═══════════════════════  Tier 2: Hypothesis Memory  ═══════════════════════

    def generate_hypothesis(self, condition: str, prediction: str, app_class: str) -> str:
        """Creates a new hypothesis."""
        hyp_id = f"hyp_{int(time.time())}_{len(self.hypotheses)}"
        self.hypotheses[hyp_id] = Hypothesis(
            hypothesis_id=hyp_id,
            condition=condition,
            prediction=prediction,
            app_class=app_class,
            created=time.time(),
        )
        self._save()
        return hyp_id

    def test_hypothesis(self, hypothesis_id: str, outcome_matches: bool) -> Optional[Hypothesis]:
        """Updates hypothesis confidence based on test result."""
        hyp = self.hypotheses.get(hypothesis_id)
        if not hyp:
            return None

        if outcome_matches:
            hyp.evidence_for += 1
        else:
            hyp.evidence_against += 1

        hyp.confidence = hyp.evidence_for / (hyp.evidence_for + hyp.evidence_against + 1)
        hyp.last_tested = time.time()

        # Promote to fact if confidence is high enough
        total_tests = hyp.evidence_for + hyp.evidence_against
        if hyp.confidence > 0.9 and total_tests >= 5:
            hyp.is_fact = True

        self._save()
        return hyp

    def get_relevant_hypotheses(self, app_class: str, context: str = "") -> List[Hypothesis]:
        """Gets hypotheses relevant to the current app and context."""
        relevant = []
        context_lower = context.lower()

        for hyp in self.hypotheses.values():
            if hyp.app_class == app_class or hyp.app_class == "*":
                # Check if context matches condition
                if not context_lower or any(w in hyp.condition.lower() for w in context_lower.split()):
                    relevant.append(hyp)

        relevant.sort(key=lambda x: x.confidence, reverse=True)
        return [relevant[i] for i in range(len(relevant)) if i < 10]  # type: ignore

    # ═══════════════════════  Tier 3: Process Memory  ═══════════════════════

    def find_matching_process(self, task: str, app_class: str = "") -> Optional[ProcessIP]:
        """Finds a stored process matching the current task."""
        task_lower = task.lower()
        best_match = None
        best_score = 0

        import difflib
        for proc in self.processes.values():
            # Keyword overlap -> upgraded to Semantic
            # Normalize old saved processes (e.g. "open_chrome_browser") by replacing underscores
            proc_pattern_clean = proc.task_pattern.lower().replace("_", " ")
            
            # Use SequenceMatcher for phrase-level semantic similarity
            score = difflib.SequenceMatcher(None, proc_pattern_clean, task_lower).ratio()

            # Boost same app
            if app_class and proc.app_class == app_class:
                score += 0.4

            # Boost proven processes
            if proc.category == "TYPICAL":
                score += 0.3
            elif proc.category == "GENERAL":
                score += 0.2

            if score > best_score and score > 0.3:
                best_score = score
                best_match = proc

        return best_match

    def record_process(self, task_pattern: str, app_class: str, steps: List[Dict]) -> str:
        """Records a new reusable process."""
        proc_id = f"proc_{int(time.time())}_{len(self.processes)}"
        proc = ProcessIP(
            process_id=proc_id,
            task_pattern=task_pattern,
            app_class=app_class,
            steps=steps,
            run_count=1,
            success_count=1,
            created=time.time(),
            last_used=time.time(),
        )
        self.processes[proc_id] = proc
        self._task_processes[task_pattern].append(proc_id)
        self._save()
        return proc_id

    def update_process_usage(self, process_id: str, success: bool):
        """Updates process usage statistics."""
        proc = self.processes.get(process_id)
        if not proc:
            return

        proc.run_count += 1
        if success:
            proc.success_count += 1
        proc.last_used = time.time()

        # Update category
        if proc.run_count > 5:
            proc.category = "TYPICAL"
        elif proc.run_count >= 2:
            proc.category = "SPECIAL"
        else:
            proc.category = "RARE"

        self._save()

    def replay_process(self, process_id: str) -> Optional[List[Dict]]:
        """Returns the steps of a stored process for replay."""
        proc = self.processes.get(process_id)
        return proc.steps if proc else None

    # ═══════════════════════  Tier 4: Nuance Ledger  ═══════════════════════

    def record_nuance(self, app_class: str, element_id: str, nuance_type: str,
                      severity: str, description: str, workaround: str = "",
                      trigger_condition: str = "") -> str:
        """Records a subtle non-obvious behavior."""
        nuance_id = f"nua_{int(time.time())}_{len(self.nuances)}"
        self.nuances[nuance_id] = Nuance(
            nuance_id=nuance_id,
            app_class=app_class,
            element_id=element_id,
            nuance_type=nuance_type,
            severity=severity,
            description=description,
            workaround=workaround,
            trigger_condition=trigger_condition,
            created=time.time(),
        )
        self._save()
        return nuance_id

    def get_nuances(self, app_class: str, element_id: str = "") -> List[Nuance]:
        """Gets nuances for a specific app and optionally a specific element."""
        results = []
        for n in self.nuances.values():
            if n.app_class == app_class:
                if not element_id or n.element_id == element_id:
                    results.append(n)

        # Sort: CRITICAL first, then IMPORTANT, then INFORMATIONAL
        severity_order = {"CRITICAL": 0, "IMPORTANT": 1, "INFORMATIONAL": 2}
        results.sort(key=lambda n: severity_order.get(n.severity, 3))
        return results

    # ═══════════════════════  Auto-extraction  ═══════════════════════

    def _maybe_extract_process(self, episode: Episode):
        """Auto-extracts a reusable process from a successful episode."""
        # Use the FULL task as the pattern, replacing spaces with underscores
        # instead of aggressively truncating to 4 words.
        pattern = episode.task.lower().replace(" ", "_")

        # Check if similar process already exists
        existing = self.find_matching_process(episode.task, episode.app_class)
        if existing:
            self.update_process_usage(existing.process_id, True)
        else:
            # Extract step summaries
            step_summaries = []
            for s in episode.steps:
                step_summaries.append({
                    "action": s.get("action", ""),
                    "target_desc": s.get("target", ""),
                    "method": s.get("method", ""),
                })
            if step_summaries:
                # Calculate complexity tier
                num_steps = len(step_summaries)
                tier = ComplexityTier.SIMPLE.value
                if num_steps >= 5:
                    tier = ComplexityTier.COMPLEX.value
                elif num_steps >= 3:
                    tier = ComplexityTier.COMPOUND.value
                
                # Check for navigation intent if task involves certain keywords
                nav_keywords = ["open", "go to", "navigate", "browse", "switch"]
                is_nav = any(kw in episode.task.lower() for kw in nav_keywords)
                if is_nav:
                    tier = ComplexityTier.NAVIGATION.value

                proc_id = self.record_process(pattern, episode.app_class, step_summaries)
                proc = self.processes[proc_id]
                proc.complexity_tier = tier
                proc.is_navigation = is_nav

        self._maybe_deduce_task_schema(episode)

    def _maybe_deduce_task_schema(self, episode: Episode):
        """Creates or updates a deep-labeled ParameterizedTaskSchema from episodes."""
        pattern = episode.task.lower().strip()
        task_id = f"schema_{pattern.replace(' ', '_')}"
        
        if task_id not in self.task_schemas:
            self.task_schemas[task_id] = ParameterizedTaskSchema(
                task_id=task_id,
                task_name=episode.task,
                app_class=episode.app_class
            )
        schema = self.task_schemas[task_id]
        
        # --- PHASE 1: PATH AND ERROR TRACKING (Individual Episode Learning) ---
        # 1. Did the episode contain failures but succeed overall? Subagent distillation!
        has_failures = any(not s.get("success", False) for s in episode.steps)
        if has_failures:
            print("[ExperientialMemory] Episode involves recoveries; initializing PathAnalyzerAgent...")
            analyzer = PathAnalyzerAgent()
            err_record = analyzer.analyze_episode(episode.steps)
            if err_record:
                schema.known_errors[err_record.error_id] = err_record
                print(f"[ExperientialMemory] Logged new error path: {err_record.description}")
        
        # 2. Extract successful sequence as a valid path
        successful_steps = [s for s in episode.steps if s.get("success", True)]
        current_path_nodes = []
        for i, s in enumerate(successful_steps):
            current_path_nodes.append(ActionNode(
                node_id=f"{task_id}_pstep_{i}",
                action_type=s.get("action", ""),
                target_identifier=s.get("target", "")
            ))
            
        path_id = f"path_{int(time.time())}"
        schema.valid_paths[path_id] = TaskPath(
            path_id=path_id,
            primary_nodes=current_path_nodes,
            efficiency_score=1.0 / max(len(current_path_nodes), 1)
        )
        
        # 3. Mark as verified
        schema.is_planned_only = False

        # --- PHASE 2: COMPARATIVE DEDUCTION (Multi-Episode Learning) ---
        similar_eps = [e for e in self.episodes if e.task.lower().strip() == pattern and e.success]
        if len(similar_eps) < 2:
            # Even with one episode, we can have a basic schema
            if not schema.execution_graph:
                schema.execution_graph = current_path_nodes
            self._save()
            return 
            
        schema.total_episodes_analyzed = len(similar_eps)
        
        # Simple deduction algorithm: compare the latest episode with the first episode
        ref_ep = similar_eps[0]
        curr_ep = episode
        
        # Compare step by step (assuming linear for now)
        min_steps = min(len(ref_ep.steps), len(curr_ep.steps))
        max_steps = max(len(ref_ep.steps), len(curr_ep.steps))
        
        # Self-Realization Differentiation Check
        # If the number of steps differs by > 50% or if critical targets differ completely, we fork.
        if min_steps > 0 and (max_steps / min_steps > 1.5):
            new_task_id = f"{task_id}_variant_{int(time.time())}"
            justification = f"Self-realization: Episode step count ({len(curr_ep.steps)}) diverged significantly from base schema ({len(ref_ep.steps)})."
            
            self.task_schemas[new_task_id] = ParameterizedTaskSchema(
                task_id=new_task_id,
                task_name=episode.task,
                app_class=episode.app_class,
                parent_schema_id=task_id,
                differentiation=SchemaDifferentiation(
                    justification=justification,
                    is_human_guided=False,
                    trigger_condition="Structural divergence in step count"
                )
            )
            schema = self.task_schemas[new_task_id]
            # Since it's a new schema, we treat the current episode as its baseline
            ref_ep = curr_ep
            min_steps = len(curr_ep.steps)
        
        new_graph = []
        
        for i in range(min_steps):
            ref_step = ref_ep.steps[i]
            curr_step = curr_ep.steps[i]
            
            action_type = curr_step.get("action", "")
            target = curr_step.get("target", "")
            method = curr_step.get("method", "")
            
            ref_target = ref_step.get("target", "")
            ref_method = ref_step.get("method", "")
            
            # Action Node
            node = ActionNode(
                node_id=f"{task_id}_step_{i}",
                action_type=action_type,
                target_identifier=target
            )
            
            # Deduce Variable vs Constant
            if method != ref_method and method:
                var_name = f"var_{i}_{action_type}"
                if var_name not in schema.variables:
                    schema.variables[var_name] = ProcessVariable(name=var_name, description=f"Variable for {action_type}")
                schema.variables[var_name].examples.add(method)
                schema.variables[var_name].examples.add(ref_method)
                node.method_ref = f"${var_name}"
            elif method:
                const_name = f"const_{i}_{action_type}"
                if const_name not in schema.constants:
                    schema.constants[const_name] = ProcessConstant(name=const_name, value=method)
                node.method_ref = f"#{const_name}"
                
            new_graph.append(node)
            
        schema.execution_graph = new_graph
        # Simple Order of Magnitude calculation (e.g. 5 -> 5, 23 -> 20, 105 -> 100)
        magnitude = int(10 ** (len(str(len(new_graph))) - 1))
        schema.estimated_total_steps = max(len(new_graph) // magnitude * magnitude, 1) if len(new_graph) > 5 else max(1, len(new_graph))
        schema.last_updated = time.time()
        schema.execution_graph = new_graph
        # Simple Order of Magnitude calculation (e.g. 5 -> 5, 23 -> 20, 105 -> 100)
        magnitude = int(10 ** (len(str(len(new_graph))) - 1))
        schema.estimated_total_steps = max(len(new_graph) // magnitude * magnitude, 1) if len(new_graph) > 5 else max(1, len(new_graph))
        schema.last_updated = time.time()
        self._save()

    def propose_planned_schema(self, schema: ParameterizedTaskSchema) -> None:
        """Stores a proactively planned schema for later execution/verification."""
        schema.is_planned_only = True
        self.task_schemas[schema.task_id] = schema
        self._save()
        print(f"[ExperientialMemory] Proposed planned schema for: {schema.task_id}")

    def discard_planned_schema(self, task_id: str) -> bool:
        """Discards a planned schema if it proved useless before or after execution."""
        if task_id in self.task_schemas:
            # We can optionally restrict discarding to ONLY 'is_planned_only' schemas
            # but usually we want the flexibility to discard bad plans anytime.
            del self.task_schemas[task_id]
            self._save()
            print(f"[ExperientialMemory] Discarded schema: {task_id}")
            return True
        return False

    def differentiate_schema_human_loop(self, base_task_id: str, new_task_name: str, justification: str, trigger_condition: str) -> str:
        """Explicitly forks a schema via human-in-the-loop feedback."""
        if base_task_id not in self.task_schemas:
            raise KeyError(f"Base schema {base_task_id} not found.")
            
        base_schema = self.task_schemas[base_task_id]
        new_task_id = f"{base_task_id}_human_variant_{int(time.time())}"
        
        import copy
        new_schema = copy.deepcopy(base_schema)
        new_schema.task_id = new_task_id
        new_schema.task_name = new_task_name
        new_schema.parent_schema_id = base_task_id
        new_schema.differentiation = SchemaDifferentiation(
            justification=justification,
            is_human_guided=True,
            trigger_condition=trigger_condition
        )
        new_schema.last_updated = time.time()
        
        self.task_schemas[new_task_id] = new_schema
        self._save()
        return new_task_id

    def _extract_failure_nuance(self, episode: Episode):
        """Auto-extracts nuances from a failed episode."""
        if not episode.failure_reason:
            return

        reason = episode.failure_reason.lower()

        # Classify nuance type
        if "timeout" in reason or "slow" in reason or "wait" in reason:
            ntype = "TIMING"
        elif "dialog" in reason or "popup" in reason or "modal" in reason:
            ntype = "MODAL"
        elif "state" in reason or "condition" in reason:
            ntype = "STATE_DEPENDENT"
        elif "step" in reason or "sequence" in reason:
            ntype = "MULTI_STEP"
        else:
            ntype = "PLATFORM_QUIRK"

        self.record_nuance(
            app_class=episode.app_class,
            element_id=episode.steps[-1].get("target", "") if episode.steps else "",
            nuance_type=ntype,
            severity="IMPORTANT",
            description=episode.failure_reason,
            trigger_condition=episode.task,
        )

    def _maybe_extract_navigation_route(self, episode: Episode):
        """Auto-extracts navigation routes from a successful episode."""
        if not episode.success or len(episode.steps) < 1:
            return

        # Start state: initial URL or app title
        start_state = episode.context.get("initial_url", episode.context.get("initial_title", "start"))
        # Specific slice for linter
        ep_list = list(episode.steps)
        final_context = ep_list[-1].get("context", {})  # type: ignore
        end_state = final_context.get("url", final_context.get("title", "end"))

        if start_state == end_state:
            return

        route_id = f"route_{int(time.time())}_{len(self.routes)}"
        route = NavigationRoute(
            route_id=route_id,
            app_class=episode.app_class,
            start_state=str(start_state),
            end_state=str(end_state),
            steps=episode.steps,
            success_count=1,
            total_count=1,
            avg_duration_s=episode.duration_s,
            last_used=time.time(),
        )
        self.routes[route_id] = route
        
        # Categorize into hierarchy
        self._ensure_hierarchy_branch("app_class", episode.app_class)
        self._hierarchy["app_class"][episode.app_class]["routes"].append(route_id)

    # ═══════════════════════  Persistence  ═══════════════════════
    
    def _to_dict(self, obj: Any) -> Any:
        """Safely converts dataclass or dict to dictionary for serialization."""
        if isinstance(obj, (Episode, Hypothesis, ProcessIP, Nuance, NavigationRoute, UIPattern, PatternElement)):
            return asdict(obj) # type: ignore
        if isinstance(obj, Enum):
            return obj.value
        try:
            return dict(obj)
        except Exception:
            return str(obj)

    def _save(self):
        """Persists all memory to disk (flat + hierarchical)."""
        episode_list = [self.episodes[i] for i in range(len(self.episodes)) if i >= max(0, len(self.episodes)-100)]
        data = {
            "episodes": [self._to_dict(ep) for ep in episode_list],
            "hypotheses": {k: self._to_dict(v) for k, v in self.hypotheses.items()},
            "processes": {k: self._to_dict(v) for k, v in self.processes.items()},
            "nuances": {nid: self._to_dict(n) for nid, n in self.nuances.items()},
            "routes": {rid: self._to_dict(r) for rid, r in self.routes.items()},
            "task_schemas": {tid: ts.serialize() for tid, ts in self.task_schemas.items()},
            "adaptive_anchors": {k: self._to_dict(v) for k, v in self.adaptive_anchors.items()},
            "strategy_stats": self.strategy_stats,  # Dynamic ActionPredictor learning data
            "user_profile": self._to_dict(self.user_profile),
            "device_profile": self._to_dict(self.device_profile),
            "hierarchy": self._hierarchy,
            "cross_app_confirmations": {k: list(v) for k, v in self._cross_app_confirmations.items()},
            "loop_incidents": self.loop_incidents,
        }

        path = os.path.join(self.storage_dir, "memory.json")
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            # Structured Experiential Saving (Task/Path/Error Folders)
            for schema in self.task_schemas.values():
                schema.save(self.exp_storage)
                
        except Exception as e:
            print(f"[ExperientialMemory] Save failed: {e}")

    def _load(self):
        """Loads persisted memory from disk (flat + hierarchical)."""
        path = os.path.join(self.storage_dir, "memory.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Restore episodes
            for ep_data in data.get("episodes", []):
                # Ensure types for Episode
                if "duration_s" in ep_data: ep_data["duration_s"] = float(ep_data["duration_s"])
                if "timestamp" in ep_data: ep_data["timestamp"] = float(ep_data["timestamp"])
                if "steps" in ep_data: ep_data["steps"] = list(ep_data["steps"])
                if "context" in ep_data: ep_data["context"] = dict(ep_data["context"])
                self.episodes.append(Episode(**ep_data))  # type: ignore
            
            for i in range(len(self.episodes)):
                ep = self.episodes[i]
                self._app_episodes[ep.app_class].append(i)

            # Restore hypotheses
            for k, v in data.get("hypotheses", {}).items():
                if "confidence" in v: v["confidence"] = float(v["confidence"])
                if "evidence_for" in v: v["evidence_for"] = int(v["evidence_for"])
                if "evidence_against" in v: v["evidence_against"] = int(v["evidence_against"])
                if "created" in v: v["created"] = float(v["created"])
                if "last_tested" in v: v["last_tested"] = float(v["last_tested"])
                self.hypotheses[k] = Hypothesis(**v)  # type: ignore

            # Restore processes
            for k, v in data.get("processes", {}).items():
                if "run_count" in v: v["run_count"] = int(v["run_count"])
                if "success_count" in v: v["success_count"] = int(v["success_count"])
                if "created" in v: v["created"] = float(v["created"])
                if "last_used" in v: v["last_used"] = float(v["last_used"])
                if "steps" in v: v["steps"] = list(v["steps"])
                self.processes[k] = ProcessIP(**v)  # type: ignore
                self._task_processes[v.get("task_pattern", "")].append(k)

            # Restore nuances
            for k, v in data.get("nuances", {}).items():
                if "occurrence_count" in v: v["occurrence_count"] = int(v["occurrence_count"])
                if "created" in v: v["created"] = float(v["created"])
                self.nuances[k] = Nuance(**v)  # type: ignore

            # Restore routes
            for rid, rd in data.get("routes", {}).items():
                if "success_count" in rd: rd["success_count"] = int(rd["success_count"])
                if "total_count" in rd: rd["total_count"] = int(rd["total_count"])
                if "avg_duration_s" in rd: rd["avg_duration_s"] = float(rd["avg_duration_s"])
                if "last_used" in rd: rd["last_used"] = float(rd["last_used"])
                if "steps" in rd: rd["steps"] = list(rd["steps"])
                self.routes[rid] = NavigationRoute(**rd)  # type: ignore

            for pid, pd in data.get("patterns", {}).items():
                if "elements" in pd:
                    elements = []
                    for ed in pd["elements"]:
                        if "rel_x" in ed: ed["rel_x"] = float(ed["rel_x"])
                        if "rel_y" in ed: ed["rel_y"] = float(ed["rel_y"])
                        if "optional" in ed: ed["optional"] = bool(ed["optional"])
                        elements.append(PatternElement(**ed))  # type: ignore
                    pd["elements"] = elements
                if "success_count" in pd: pd["success_count"] = int(pd["success_count"])
                if "last_used" in pd: pd["last_used"] = float(pd["last_used"])
                if "created" in pd: pd["created"] = float(pd["created"])
                if "signature" in pd: pd["signature"] = list(pd["signature"])
                self.patterns[pid] = UIPattern(**pd)  # type: ignore

            for k, v in data.get("task_schemas", {}).items():
                v_copy = dict(v)
                v_copy.pop("execution_graph", None)
                v_copy.pop("constants", None)
                v_copy.pop("variables", None)
                diff_data = v_copy.pop("differentiation", None)
                errors_data = v_copy.pop("known_errors", None) or {}
                paths_data = v_copy.pop("valid_paths", None) or {}
                
                # Default for existing files
                if "estimated_total_steps" not in v_copy:
                    v_copy["estimated_total_steps"] = 1 # type: ignore
                if "is_planned_only" not in v_copy:
                    v_copy["is_planned_only"] = False # type: ignore
                
                schema = ParameterizedTaskSchema(**v_copy)
                if diff_data:
                    schema.differentiation = SchemaDifferentiation(**diff_data)
                    
                for ck, cv in v.get("constants", {}).items():
                    schema.constants[ck] = ProcessConstant(**cv)
                for vk, vv in v.get("variables", {}).items():
                    schema.variables[vk] = ProcessVariable(**vv)
                for ek, ev in errors_data.items():
                    schema.known_errors[ek] = ErrorRecord(**ev)
                for pk, pv in paths_data.items():
                    # Parse internal nodes
                    raw_nodes = pv.pop("primary_nodes", [])
                    constructed_nodes = []
                    for rn in raw_nodes:
                        # minimal reconstruct for paths
                        if rn.get("node_type") == "ACTION":
                            constructed_nodes.append(ActionNode(**{k: v for k, v in rn.items() if k != "node_type"}))
                    pv["primary_nodes"] = constructed_nodes
                    schema.valid_paths[pk] = TaskPath(**pv)
                    
                for nd in v.get("execution_graph", []):
                    if nd.get("node_type") == "ACTION":
                        schema.execution_graph.append(ActionNode(node_id=nd["node_id"], action_type=nd["action_type"], target_identifier=nd["target_identifier"], method_ref=nd.get("method_ref", ""), expected_outcome=nd.get("expected_outcome", "")))
                
                self.task_schemas[k] = schema

            self._hierarchy = data.get("hierarchy", self._hierarchy)

            # Restore cross-app confirmations
            for k, v in data.get("cross_app_confirmations", {}).items():
                self._cross_app_confirmations[k] = set(v)

            # Restore loop incidents
            self.loop_incidents = data.get("loop_incidents", [])

            # Restore Context Profiles
            if "user_profile" in data:
                up_data = data["user_profile"]
                if "known_accounts" in up_data: up_data["known_accounts"] = dict(up_data["known_accounts"])
                if "behavior_patterns" in up_data: up_data["behavior_patterns"] = dict(up_data["behavior_patterns"])
                if "assumptions" in up_data: up_data["assumptions"] = list(up_data["assumptions"])
                if "last_updated" in up_data: up_data["last_updated"] = float(up_data["last_updated"])
                self.user_profile = UserProfile(**up_data) # type: ignore
                
            if "device_profile" in data:
                dp_data = data["device_profile"]
                if "file_system_anchors" in dp_data: dp_data["file_system_anchors"] = dict(dp_data["file_system_anchors"])
                if "last_updated" in dp_data: dp_data["last_updated"] = float(dp_data["last_updated"])
                self.device_profile = DeviceProfile(**dp_data) # type: ignore
                
            for k, v in data.get("adaptive_anchors", {}).items():
                if "last_known_coords" in v: v["last_known_coords"] = list(v["last_known_coords"])
                if "confidence" in v: v["confidence"] = float(v["confidence"])
                if "variations" in v: v["variations"] = list(v["variations"])
                if "last_used" in v: v["last_used"] = float(v["last_used"])
                self.adaptive_anchors[k] = AdaptiveAnchor(**v) # type: ignore

            # Restore strategy stats for dynamic ActionPredictor
            self.strategy_stats = data.get("strategy_stats", {})

        except Exception as e:
            print(f"[ExperientialMemory] Load failed: {e}")

    def _migrate_flat_to_hierarchy(self):
        """Auto-migrates existing flat data into the hierarchy on first load."""
        print("[ExperientialMemory] Migrating flat data to hierarchical structure...")

        for hyp_id, hyp in self.hypotheses.items():
            ac = hyp.app_class if hyp.app_class != "*" else "*"
            if ac == "*":
                self._hierarchy["global"]["*"]["hypotheses"].append(hyp_id)
            else:
                self._ensure_hierarchy_branch("app_class", ac)
                self._hierarchy["app_class"][ac]["hypotheses"].append(hyp_id)

        for proc_id, proc in self.processes.items():
            ac = proc.app_class
            self._ensure_hierarchy_branch("app_class", ac)
            self._hierarchy["app_class"][ac]["processes"].append(proc_id)

        for nua_id, nua in self.nuances.items():
            ac = nua.app_class
            self._ensure_hierarchy_branch("app_class", ac)
            self._hierarchy["app_class"][ac]["nuances"].append(nua_id)

        # Categorize episodes
        for ep in self.episodes:
            self._categorize_episode(ep)

        self._save()
        print("[ExperientialMemory] Migration complete.")

    # ═══════════════════════  Loop Incident Recording  ═══════════════════════

    def record_loop_incident(self, app_class: str, loop_type: str,
                              offending_target: str, repeat_count: int,
                              root_cause: str, recovery_outcome: str,
                              context_summary: str = ""):
        """Records a loop incident for future avoidance learning.

        Args:
            app_class: The application class where the loop occurred.
            loop_type: REPETITION, OSCILLATION, STALL, or COORD_DRIFT.
            offending_target: The target element causing the loop.
            repeat_count: How many times the sub-objective was repeated.
            root_cause: Diagnosed root cause (e.g., UNRESOLVED_LOOP, WRONG_TARGET).
            recovery_outcome: What recovered the agent (or FORCED_FAIL).
            context_summary: Brief description of the loop context.
        """
        incident = {
            "timestamp": time.time(),
            "app_class": app_class,
            "loop_type": loop_type,
            "offending_target": offending_target,
            "repeat_count": repeat_count,
            "root_cause": root_cause,
            "recovery_outcome": recovery_outcome,
            "context_summary": context_summary,
        }
        self.loop_incidents.append(incident)

        # Keep only last 50 incidents to avoid bloat
        if len(self.loop_incidents) > 50:
            self.loop_incidents = self.loop_incidents[-50:]  # type: ignore

        print(f"[ExperientialMemory] Recorded loop incident: "
              f"{loop_type} on '{offending_target}' (x{repeat_count}), "
              f"cause={root_cause}, outcome={recovery_outcome}")
        self._save()

    # ═══════════════════════  Context Generation  ═══════════════════════

    def get_context_for_planner(self, app_class: str, current_task: str = "",
                                app_instance: str = "", site: str = "") -> str:
        """Generates a concise context string for the planner using hierarchical recall."""
        lines = []

        # Hierarchical hypotheses (specific → general)
        hyps = self.recall_hypotheses_hierarchical(app_class, app_instance, site)
        if hyps:
            lines.append("Known behaviors:")
            hyps_list = list(hyps)
            for h, source in hyps_list[0:5]:  # type: ignore
                status = "FACT" if h.is_fact else f"conf={h.confidence:.1f}"
                lines.append(f"  - IF {h.condition} THEN {h.prediction} [{status}] (from:{source})")

        # Hierarchical process match
        proc, proc_source = self.find_process_hierarchical(current_task, app_class, app_instance)
        if proc:
            lines.append(f"Reusable process: '{proc.task_pattern}' ({proc.complexity_tier}, {proc.category}, {proc.run_count} runs, from:{proc_source})")
            if proc.is_navigation:
                lines.append(f"  - Note: This is an efficient NAVIGATION route.")
                
        # Phase 9: Inject Relevant Past Episodes directly
        if current_task:
            similar_episodes = self.recall_similar(current_task, app_class, top_k=2)
            if similar_episodes:
                lines.append("\n## RELEVANT PAST EPISODES (Ground Truth)")
                lines.append("Here are exact steps you took in similar successful past tasks. Follow these exactly if they match the current goal.")
                for i, ep in enumerate(similar_episodes, 1):
                    lines.append(f"\n[Episode {i}] Task: '{ep.task}'")
                    lines.append(f"Success: {ep.success}")
                    lines.append("Steps taken:")
                    for j, s in enumerate(ep.steps, 1):
                         lines.append(f"  {j}. {s.get('action', '')} on '{s.get('target', '')}'")
                         if s.get('method'):
                             lines.append(f"     Method: {s.get('method')}")

        # Hierarchical routes (specific → general)
        routes_list = self.recall_routes_hierarchical(app_class, app_instance)
        if routes_list:
            lines.append("Recurring Navigation Routes (Specific):")
            routes_chunk = [routes_list[i] for i in range(len(routes_list)) if i < 3]  # type: ignore
            for r in routes_chunk:
                lines.append(f"  - Route: {r.start_state} -> {r.end_state} (OK in {r.avg_duration_s:.1f}s)")

        patterns_list = self.recall_patterns_hierarchical(app_class)
        if patterns_list:
            lines.append("Recurring UI Patterns (Modals/Grids):")
            patterns_chunk = [patterns_list[i] for i in range(len(patterns_list)) if i < 5]  # type: ignore
            for p in patterns_chunk:
                lines.append(f"  - Pattern: {p.tier.name} {p.label} (Conf: N/A)")
                sig = ", ".join(p.signature[:5])  # type: ignore
                lines.append(f"  - Pattern: {p.label} [{p.tier}] Signature: {sig}")

        # Nuances (include critical from any level)
        nuances = self.get_nuances(app_class)
        critical_nuances = [n for n in nuances if n.severity == "CRITICAL"]
        if critical_nuances:
            lines.append("⚠ Critical nuances:")
            for n in critical_nuances[:3]:  # type: ignore
                lines.append(f"  - [{n.nuance_type}] {n.description[:80]}")

        # ── Loop Avoidance Warnings ──
        if self.loop_incidents:
            matching_incidents = [
                inc for inc in self.loop_incidents
                if inc.get("app_class", "") == app_class or inc.get("app_class", "") == "*"
            ]
            if matching_incidents:
                lines.append("")
                lines.append("⚠ LOOP AVOIDANCE (from past incidents):")
                for inc in matching_incidents[-3:]:  # type: ignore
                    lines.append(
                        f"  - AVOID: '{inc.get('offending_target', '?')}' via "
                        f"{inc.get('loop_type', '?')} loop (repeated {inc.get('repeat_count', '?')}x). "
                        f"Recovery: {inc.get('recovery_outcome', 'N/A')}"
                    )

        # ── Environmental Context Profiles ──
        has_profile_data = False
        profile_lines = ["", "## ENVIRONMENTAL CONTEXT (Learned Profiles)"]
        
        if hasattr(self, 'user_profile') and self.user_profile:
            if self.user_profile.known_accounts:
                has_profile_data = True
                profile_lines.append("User Accounts Detected:")
                for acct in self.user_profile.known_accounts.values():
                    profile_lines.append(f"  - {acct.get('detail', 'Unknown')}")
            
            if self.user_profile.assumptions:
                has_profile_data = True
                profile_lines.append("User Preferences/Assumptions:")
                for assumption in self.user_profile.assumptions:
                    profile_lines.append(f"  - {assumption}")

        if hasattr(self, 'device_profile') and self.device_profile and self.device_profile.os_info:
            has_profile_data = True
            profile_lines.append(f"Device OS: {self.device_profile.os_info}")
            
        if has_profile_data:
            lines.extend(profile_lines)

        return "\n".join(lines) if lines else ""

    # ═══════════════════════  Hierarchical Methods  ═══════════════════════

    def _ensure_hierarchy_branch(self, level: str, scope_key: str):
        """Auto-creates a hierarchy branch if it doesn't exist."""
        if scope_key not in self._hierarchy[level]:
            self._hierarchy[level][scope_key] = {
                "hypotheses": [], "processes": [], "nuances": [], "routes": [], "patterns": []
            }
            print(f"[ExperientialMemory] Auto-created hierarchy: {level}/{scope_key}")

    def _extract_site(self, context: Dict) -> str:
        """Extracts a site/context identifier from episode context."""
        # Try URL first
        url = context.get("url", context.get("current_url", ""))
        if url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                return parsed.netloc or parsed.path.split("/")[0]
            except Exception:
                pass
        # Try window title
        title = context.get("window_title", context.get("title", ""))
        if title:
            # Extract domain-like patterns from titles
            for pattern in ["- Google Chrome", "- Firefox", "- Microsoft Edge"]:
                if pattern in title:
                    page_title = title.replace(pattern, "").strip()
                    return page_title[:40]  # First 40 chars as context key
        # Try project/file context
        project = context.get("project", context.get("file_path", ""))
        if project:
            return os.path.basename(project)
        return ""

    def _categorize_episode(self, episode: Episode):
        """Sorts episode knowledge into the correct hierarchy branches."""
        app_class = episode.app_class
        app_name = episode.app_name
        site = self._extract_site(episode.context)

        # Ensure branches exist
        self._ensure_hierarchy_branch("app_class", app_class)
        self._ensure_hierarchy_branch("app_instance", app_name)
        if site:
            site_key = f"{app_name}/{site}"
            self._ensure_hierarchy_branch("site", site_key)

    def recall_hypotheses_hierarchical(self, app_class: str, app_instance: str = "",
                                        site: str = "") -> List[Tuple[Hypothesis, str]]:
        """Retrieves hypotheses from specific → general.
        Returns [(hypothesis, source_level)] with specificity weighting."""
        results = []

        # 1. Site-specific
        if site and app_instance:
            site_key = f"{app_instance}/{site}"
            for hyp_id in self._hierarchy.get("site", {}).get(site_key, {}).get("hypotheses", []):
                if hyp_id in self.hypotheses:
                    results.append((self.hypotheses[hyp_id], f"site:{site}"))

        # 2. App-instance
        if app_instance:
            for hyp_id in self._hierarchy.get("app_instance", {}).get(app_instance, {}).get("hypotheses", []):
                if hyp_id in self.hypotheses and (self.hypotheses[hyp_id], f"instance:{app_instance}") not in results:
                    results.append((self.hypotheses[hyp_id], f"instance:{app_instance}"))

        # 3. App-class
        for hyp_id in self._hierarchy.get("app_class", {}).get(app_class, {}).get("hypotheses", []):
            if hyp_id in self.hypotheses:
                results.append((self.hypotheses[hyp_id], f"class:{app_class}"))

        # 4. Global
        for hyp_id in self._hierarchy.get("global", {}).get("*", {}).get("hypotheses", []):
            if hyp_id in self.hypotheses:
                results.append((self.hypotheses[hyp_id], "global"))

        # Deduplicate, sort by confidence
        seen = set()
        deduped = []
        for h, src in results:
            if h.hypothesis_id not in seen:
                seen.add(h.hypothesis_id)
                deduped.append((h, src))
        deduped.sort(key=lambda x: x[0].confidence, reverse=True)
        return deduped

    def find_process_hierarchical(self, task: str, app_class: str,
                                   app_instance: str = "") -> Tuple[Optional[ProcessIP], str]:
        """Searches: site → instance → class → global.
        Returns (process, source_level) or (None, "")."""

        # 1. App-instance processes
        if app_instance:
            for proc_id in self._hierarchy.get("app_instance", {}).get(app_instance, {}).get("processes", []):
                proc = self.processes.get(proc_id)
                if proc and self._task_matches_process(task, proc):
                    return proc, f"instance:{app_instance}"

        # 2. App-class processes
        for proc_id in self._hierarchy.get("app_class", {}).get(app_class, {}).get("processes", []):
            proc = self.processes.get(proc_id)
            if proc and self._task_matches_process(task, proc):
                return proc, f"class:{app_class}"

        # 3. Global processes
        for proc_id in self._hierarchy.get("global", {}).get("*", {}).get("processes", []):
            proc = self.processes.get(proc_id)
            if proc and self._task_matches_process(task, proc):
                return proc, "global"

        # 4. Fallback to flat search
        proc = self.find_matching_process(task, app_class)
        return (proc, "flat") if proc else (None, "")

    def recall_routes_hierarchical(self, app_class: str, app_instance: str = "") -> List[NavigationRoute]:
        """Retrieves navigation routes from specific → general."""
        results = []
        # 1. App-instance
        if app_instance:
            for route_id in self._hierarchy.get("app_instance", {}).get(app_instance, {}).get("routes", []):
                if route_id in self.routes:
                    results.append(self.routes[route_id])

        # 2. App-class
        for route_id in self._hierarchy.get("app_class", {}).get(app_class, {}).get("routes", []):
            if route_id in self.routes:
                results.append(self.routes[route_id])

        # Deduplicate
        seen = set()
        deduped = []
        for r in results:
            key = f"{r.start_state}->{r.end_state}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    def recall_patterns_hierarchical(self, app_class: str) -> List[UIPattern]:
        """Retrieves structural UI patterns from specific → global."""
        results = []
        # 1. App-class
        for pid in self._hierarchy.get("app_class", {}).get(app_class, {}).get("patterns", []):
            if pid in self.patterns:
                results.append(self.patterns[pid])

        # 2. Global
        for pid in self._hierarchy.get("global", {}).get("*", {}).get("patterns", []):
            if pid in self.patterns:
                results.append(self.patterns[pid])

        return results

    def _task_matches_process(self, task: str, proc: ProcessIP) -> bool:
        """Checks if a task description matches a stored process."""
        proc_words = set(proc.task_pattern.lower().replace("_", " ").split())
        task_words = set(task.lower().split())
        overlap = len(proc_words & task_words)
        return overlap / max(len(proc_words), 1) > 0.3

    # ═══════════════════════  Cross-App Generalization  ═══════════════════════

    def promote_to_global(self, hypothesis_id: str):
        """Promotes an app-specific hypothesis to global if confirmed across 3+ apps."""
        hyp = self.hypotheses.get(hypothesis_id)
        if not hyp:
            return

        # Add to global hypotheses
        global_hyps = self._hierarchy["global"]["*"]["hypotheses"]
        if hypothesis_id not in global_hyps:
            global_hyps.append(hypothesis_id)
            hyp.app_class = "*"  # Mark as universal
            print(f"[ExperientialMemory] PROMOTED to global: '{hyp.condition}' -> '{hyp.prediction}'")
            self._save()

    def check_cross_app_promotion(self, hypothesis_id: str, app_class: str):
        """Records that a hypothesis was confirmed in an app and promotes if threshold met."""
        self._cross_app_confirmations[hypothesis_id].add(app_class)
        if len(self._cross_app_confirmations[hypothesis_id]) >= self._promotion_threshold:
            self.promote_to_global(hypothesis_id)

    def extract_cross_app_patterns(self):
        """Scans all app-class knowledge for common patterns and auto-promotes."""
        # Find hypotheses that appear in multiple app classes
        condition_map: Dict[str, List[str]] = defaultdict(list)  # condition → [hyp_ids]
        for hyp_id, hyp in self.hypotheses.items():
            if hyp.app_class != "*":  # Skip already-global
                key = hyp.condition.lower().strip()
                condition_map[key].append(hyp_id)

        promoted_count: int = 0
        for condition, hyp_ids in condition_map.items():
            apps = set(self.hypotheses[hid].app_class for hid in hyp_ids if hid in self.hypotheses)
            if len(apps) >= self._promotion_threshold:
                # Promote the highest-confidence one
                best_id = max(hyp_ids, key=lambda hid: self.hypotheses.get(hid, Hypothesis("","","","")).confidence)
                self.promote_to_global(best_id)
                promoted_count = int(promoted_count) + 1  # type: ignore

        if int(promoted_count) > 0:  # type: ignore
            print(f"[ExperientialMemory] Cross-app scan: promoted {promoted_count} hypotheses to global")

    # ═══════════════════════  Hierarchical Episode Recording  ═══════════════════════

    def record_episode_hierarchical(self, task: str, app_name: str, app_class: str,
                                     steps: List[Dict[str, Any]], success: bool, failure_reason: str = "",
                                     context: Optional[Dict[str, Any]] = None) -> str:
        """Records an episode with automatic hierarchical categorization."""
        # Use the existing flat recorder
        ep_id = self.record_episode(task, app_name, app_class, steps, success,
                                     failure_reason, context)

        # Categorize into hierarchy
        ep = self.episodes[-1]
        self._categorize_episode(ep)

        # If episode generated hypotheses, file them hierarchically
        site = self._extract_site(context or {})
        if site:
            site_key = f"{app_name}/{site}"
            self._ensure_hierarchy_branch("site", site_key)

        return ep_id

    # ═══════════════════════  Learning Stats  ═══════════════════════

    def get_learning_stats(self, app_class: Optional[str] = None) -> Dict[str, Any]:
        """Returns comprehensive learning statistics."""
        # Hypothesis tiers
        if app_class:
            hyps_list = list(self.hypotheses.values())
            class_hyps = [h for h in hyps_list if h.app_class == app_class]
            facts = [h for h in class_hyps if h.is_fact]
            strong = [h for h in class_hyps if h.confidence > 0.7 and not h.is_fact]
            weak = [h for h in class_hyps if h.confidence <= 0.7]
        else:
            facts = [h for h in self.hypotheses.values() if h.is_fact]
            strong = [h for h in self.hypotheses.values() if h.confidence > 0.7 and not h.is_fact]
            weak = [h for h in self.hypotheses.values() if h.confidence <= 0.7]

        # Process categories
        proc_cats: Dict[str, int] = {}
        for proc in self.processes.values():
            cat = proc.complexity_tier.value if hasattr(proc.complexity_tier, "value") else str(proc.complexity_tier)  # type: ignore
            count = proc_cats.get(cat, 0)
            proc_cats[cat] = int(count) + 1

        # Nuance severities
        nua_sev: Dict[str, int] = {}
        for nua in self.nuances.values():
            sev = nua.severity.value if hasattr(nua.severity, "value") else str(nua.severity)  # type: ignore
            nua_sev[sev] = int(nua_sev.get(sev, 0)) + 1

        return {
            "total_episodes": len(self.episodes),
            "successful_episodes": sum(1 for e in self.episodes if e.success),
            "hypotheses": {
                "total": len(self.hypotheses),
                "facts": len(facts),
                "strong": len(strong),
                "weak": len(weak),
            },
            "processes": dict(proc_cats),
            "nuances": dict(nua_sev),
            "hierarchy": {
                "app_classes": list(self._hierarchy["app_class"].keys()),
                "app_instances": list(self._hierarchy["app_instance"].keys()),
                "sites": list(self._hierarchy["site"].keys()),
                "global_hypotheses": len(self._hierarchy["global"]["*"]["hypotheses"]),
                "global_processes": len(self._hierarchy["global"]["*"]["processes"]),
            },
            "cross_app_promotions": sum(
                1 for hyps in self._cross_app_confirmations.values()
                if len(hyps) >= self._promotion_threshold
            ),
        }
