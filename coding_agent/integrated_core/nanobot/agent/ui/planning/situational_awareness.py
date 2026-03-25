"""
SituationalAwareness — Cognitive integration layer merging L1+L2+L3 into a World Model.

Builds and maintains a real-time World Model for informed decision-making:
  - Integrates OS state, app profiles, CV analysis
  - Generates navigation strategies for window switching
  - Detects/resolves window overlaps
  - Assesses tab situations
  - Produces concise context summaries for the AgenticPlanner
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ──────────────────────────  World Model  ──────────────────────────

@dataclass
class WorldModel:
    """Real-time snapshot of the agent's entire environment."""
    timestamp: float = 0.0
    frame_id: str = ""

    # Layer 1: OS state
    foreground_title: str = ""
    foreground_hwnd: int = 0
    foreground_state: Dict[str, Any] = field(default_factory=dict)    # Full WindowState as dict
    background: List[Dict[str, Any]] = field(default_factory=list)    # Background window summaries
    minimized_windows: List[Dict] = field(default_factory=list)
    maximized_windows: List[Dict] = field(default_factory=list)
    snapped_windows: List[Dict] = field(default_factory=list)
    anchored_windows: List[Dict] = field(default_factory=list)
    overlap_map: List[Dict] = field(default_factory=list)

    # Layer 2: Perception & App Profile
    foreground_profile: Dict[str, Any] = field(default_factory=dict)  # AppProfile as dict
    element_map: List[Dict[str, Any]] = field(default_factory=list)   # Detected elements
    chrome_boundaries: Dict[str, Any] = field(default_factory=dict)   # Learned element boundaries
    stability_map: Dict[str, str] = field(default_factory=dict)       # STATIC/DYNAMIC/TRANSIENT
    text_regions: List[Dict] = field(default_factory=list)
    detected_structural_patterns: List[Dict[str, Any]] = field(default_factory=list)
    experiential_insights: str = ""

    # Layer 3: System Context
    resource_state: Dict[str, Any] = field(default_factory=dict)
    tab_situation: Dict[str, Any] = field(default_factory=dict)
    navigation_recommendations: List[str] = field(default_factory=list)


# ──────────────────────────  Engine  ──────────────────────────

class SituationalAwareness:
    """Builds and maintains a real-time World Model for decision-making."""

    def __init__(self, os_awareness=None, app_characterizer=None, cv_pipeline=None,
                 boundary_learner=None, stability_classifier=None, memory=None,
                 experiential_memory=None):
        self.os = os_awareness
        self.app = app_characterizer
        self.cv = cv_pipeline
        self.boundary_learner = boundary_learner
        self.stability = stability_classifier
        self.memory = memory  # Should be UIMemoryAgent
        self.experiential_memory = experiential_memory # Should be ExperientialMemory
        self.world_model = WorldModel()

    # ═══════════════════════  Build World Model  ═══════════════════════

    def build_world_model(self, current_frame=None) -> WorldModel:
        """
        Full environment scan:
        1. OS scan -> windows, processes, resources
        2. Foreground app -> full characterization + boundary learning
        3. Background apps -> minimal profiles
        4. CV analysis -> elements, stability, text regions
        """
        model = WorldModel(timestamp=time.time())

        # ── Layer 1: OS State ──
        if self.os:
            self.os.scan_all_windows()

            fg = self.os.get_foreground_window()
            if fg:
                model.foreground_title = fg.title
                model.foreground_hwnd = fg.hwnd
                model.foreground_state = {
                    "hwnd": fg.hwnd, "title": fg.title, "state": fg.state,
                    "outer_rect": fg.outer_rect, "width": fg.width, "height": fg.height,
                    "snap_position": fg.snap_position, "screen_coverage_pct": fg.screen_coverage_pct,
                    "size_category": fg.size_category, "class_name": fg.class_name,
                    "process_name": fg.process_name,
                }

            # Background windows
            z_order = self.os.get_z_order()
            for ws in z_order:
                if ws.hwnd != model.foreground_hwnd and ws.state != "MINIMIZED":
                    model.background.append({
                        "hwnd": ws.hwnd, "title": ws.title[:50], "state": ws.state, # type: ignore
                        "process_name": ws.process_name, "snap_position": ws.snap_position,
                        "z_order": ws.z_order,
                    })

            # Categorized lists
            model.minimized_windows = [
                {"hwnd": w.hwnd, "title": w.title[:40], "process": w.process_name} # type: ignore
                for w in self.os.get_minimized_windows()
            ]
            model.maximized_windows = [
                {"hwnd": w.hwnd, "title": w.title[:40]} # type: ignore
                for w in self.os.get_maximized_windows()
            ]
            model.snapped_windows = [
                {"hwnd": w.hwnd, "title": w.title[:40], "snap": w.snap_position} # type: ignore
                for w in self.os.get_snapped_windows()
            ]
            model.anchored_windows = [
                {"hwnd": w.hwnd, "title": w.title[:40], "anchor": w.anchor_coords} # type: ignore
                for w in self.os.get_anchored_windows()
            ]

            # Overlaps
            model.overlap_map = self.os.detect_overlapping_windows()

            # Resources
            self.os.scan_processes()
            model.resource_state = self.os.get_resource_summary()

        # ── Layer 2: App Profile ──
        if self.app and model.foreground_hwnd:
            profile = self.app.characterize(
                model.foreground_hwnd,
                screenshot_frame=current_frame,
                process_name=model.foreground_state.get("process_name", ""),
                window_class=model.foreground_state.get("class_name", ""),
            )
            model.foreground_profile = profile.to_dict()
            model.tab_situation = profile.tab_awareness

        # ── Layer 1.5: Element Boundaries ──
        if self.boundary_learner and model.foreground_hwnd:
            boundaries = self.boundary_learner.detect_window_chrome(model.foreground_hwnd)
            model.chrome_boundaries = {
                eid: {
                    "rect": eb.rect, "center": eb.center, "type": eb.element_type,
                    "interaction": eb.interaction_type, "confidence": eb.confidence,
                }
                for eid, eb in boundaries.items()
            }

        # ── Layer 3: CV Analysis ──
        if current_frame is not None:
            if self.cv:
                perception = self.cv.full_analysis(current_frame)
                model.element_map = [
                    {"type": e.element_type, "x": e.x, "y": e.y, "w": e.width, "h": e.height, "label": e.label}
                    for e in perception.elements[:30] # type: ignore
                ]
                model.text_regions = perception.text_regions[:20] # type: ignore
                model.stability_map = perception.stability_classes

            if self.stability and self.stability.is_calibrated:
                model.stability_map = {
                    k: v.stability_class
                    for k, v in self.stability.get_stability_map().items()
                }

            # Structural Pattern Detection
            app_class = model.foreground_state.get("class_name", "")
            patterns = self.detect_patterns(model.element_map, app_class)
            if patterns:
                model.detected_structural_patterns = patterns

            # --- NEW: Speculative Snipping ---
            self._speculative_snip(current_frame, model)

        # ── Navigation Recommendations ──
        model.navigation_recommendations = self._build_navigation_recommendations(model)

        self.world_model = model
        return model

    # ═══════════════════════  Navigation Strategy  ═══════════════════════

    def get_navigation_strategy(self, target_app: str) -> Dict[str, Any]:
        """
        Optimal path to reach a target application.
        Returns: {strategy, steps, estimated_time, risk_factors}
        """
        if not self.os:
            return {"strategy": "UNKNOWN", "steps": [], "estimated_time": 0, "risk_factors": []}

        # Find target in window ledger
        target_windows = []
        for ws in self.os.window_ledger.values():
            if (target_app.lower() in ws.title.lower() or
                target_app.lower() in ws.process_name.lower()):
                target_windows.append(ws)

        if not target_windows:
            return {
                "strategy": "LAUNCH",
                "steps": [f"Application '{target_app}' not found. Need to launch it."],
                "estimated_time": 5.0,
                "risk_factors": ["App may not be installed"],
            }

        target = target_windows[0]

        # Already foreground?
        if target.is_foreground:
            return {
                "strategy": "ALREADY_FOCUSED",
                "steps": [],
                "estimated_time": 0,
                "risk_factors": [],
            }

        # Minimized?
        if target.state == "MINIMIZED":
            return {
                "strategy": "RESTORE_DIRECT",
                "steps": [
                    f"Use 'focus_window' on HWND {target.hwnd} to restore '{target.title[:30]}'", # type: ignore
                ],
                "estimated_time": 0.5,
                "risk_factors": ["Direct OS restore bypasses taskbar"],
                "target_hwnd": target.hwnd
            }

        # Visible but behind?
        if target.is_visible and target.state != "MINIMIZED":
            # Check if covered by other windows
            blockers = self._find_blocking_windows(target)
            if blockers:
                return {
                    "strategy": "MINIMIZE_BLOCKERS_AND_FOCUS",
                    "steps": [
                        f"Minimize {len(blockers)} blocking window(s)",
                        f"Focus on '{target.title[:30]}'",
                    ],
                    "estimated_time": 1.5,
                    "risk_factors": [f"Blocked by: {', '.join(b['title'][:20] for b in blockers)}"], # type: ignore
                }
            return {
                "strategy": "CLICK_TO_FOCUS",
                "steps": [f"Click or Alt-Tab to '{target.title[:30]}'"], # type: ignore
                "estimated_time": 0.5,
                "risk_factors": [],
            }

        return {
            "strategy": "FOCUS",
            "steps": [f"Focus window '{target.title[:30]}'"], # type: ignore
            "estimated_time": 0.5,
            "risk_factors": [],
        }

    def _find_blocking_windows(self, target) -> List[Dict]:
        """Finds windows that are blocking the target window."""
        blockers = []
        if not self.os:
            return blockers

        for ws in self.os.window_ledger.values():
            if ws.hwnd == target.hwnd or ws.state == "MINIMIZED":
                continue
            if ws.z_order < target.z_order:  # Lower z_order = closer to front
                overlap = self.os._overlap_percentage(ws.outer_rect, target.outer_rect)
                if overlap > 10:
                    blockers.append({"hwnd": ws.hwnd, "title": ws.title, "overlap": overlap})

        return blockers

    # ═══════════════════════  Tab Situation  ═══════════════════════

    def assess_tab_situation(self) -> Dict[str, Any]:
        """Assess current tab situation for the foreground app."""
        profile_dict = self.world_model.foreground_profile
        if not profile_dict:
            return {"has_tabs": False}

        tab_bar = profile_dict.get("topology", {}).get("tab_bar", {})
        return {
            "has_tabs": tab_bar.get("present", False),
            "tab_count": tab_bar.get("tab_count", 0),
            "active_tab": tab_bar.get("active_tab", -1),
            "criticality": profile_dict.get("tab_awareness", {}).get("tab_criticality", "LOW"),
            "management": profile_dict.get("tab_awareness", {}).get("tab_management", {}),
        }

    # ═══════════════════════  Overlap Strategy  ═══════════════════════

    def detect_window_overlap_strategy(self) -> List[Dict]:
        """Resolution plan for overlapping windows."""
        strategies = []
        for overlap in self.world_model.overlap_map[:5]: # type: ignore
            a = overlap["window_a"]
            b = overlap["window_b"]
            pct = overlap["overlap_pct"]

            if pct > 80:
                action = "MINIMIZE_BEHIND"
                desc = f"Almost fully covered: minimize the behind window"
            elif pct > 40:
                action = "SNAP_SIDE_BY_SIDE"
                desc = f"Significant overlap: snap windows side by side"
            else:
                action = "ACCEPT"
                desc = f"Minor overlap ({pct:.0f}%): acceptable"

            strategies.append({
                "windows": [a["title"][:30], b["title"][:30]], # type: ignore
                "overlap_pct": pct,
                "recommended_action": action,
                "description": desc,
            })

        return strategies

    # ═══════════════════════  Context for Planner  ═══════════════════════
    
    def get_current_context(self) -> Dict[str, Any]:
        """Returns a summary of the current application context."""
        m = self.world_model
        return {
            "app_name": m.foreground_title or "Desktop",
            "app_class": m.foreground_state.get("class_name", ""),
            "hwnd": m.foreground_hwnd
        }

    def get_context_for_planner(self) -> str:
        """Concise text summary for AgenticPlanner's Gemini prompt."""
        m = self.world_model
        lines = []

        # Foreground
        if m.foreground_title:
            identity = m.foreground_profile.get("identity") or {}
            cat = identity.get("category", "")
            state_info = m.foreground_state.get("state", "")
            snap = m.foreground_state.get("snap_position", "NONE")
            size = m.foreground_state.get("size_category", "")
            lines.append(f"Active: '{m.foreground_title[:40]}' [{cat}] {state_info} {size}") # type: ignore
            if snap and str(snap) != "NONE":
                lines.append(f"  - Snapped: {snap}")
            
            # Check for multiple instances of same app/process
            process_name = m.foreground_state.get("process_name", "")
            instances = [w for w in (m.background + m.minimized_windows) 
                         if w.get("process_name") == process_name or w.get("process") == process_name]
            if instances:
                lines.append(f"  Note: {len(instances) + 1} instances of {process_name} detected. Use HWND for precision.")

        # Chrome
        if m.chrome_boundaries:
            chrome_list = ", ".join(m.chrome_boundaries.keys())
            lines.append(f"  Chrome elements: {chrome_list}")

        # Tabs
        tab = self.assess_tab_situation()
        if tab.get("has_tabs"):
            lines.append(f"  Tabs: {tab['tab_count']} (criticality: {tab['criticality']})")

        # Background
        if m.background:
            bg_slice = list(m.background)[0:5] # type: ignore
            bg_list = ", ".join(f"'{w.get('title', 'Unknown')[:25]}'" for w in bg_slice) # type: ignore
            lines.append(f"Background: {bg_list}")

        # Minimized
        if m.minimized_windows:
            lines.append(f"Minimized: {len(m.minimized_windows)} windows")

        # Topology Semantics
        topology = m.foreground_profile.get("topology", {})
        semantics = []
        for key, value in topology.items():
            if isinstance(value, dict) and value.get("present") and value.get("semantic_utility"):
                semantics.append(f"{key.replace('_', ' ').title()}: {value['semantic_utility']}")
        
        if semantics:
            lines.append("  Semantic Context:")
            for s in semantics:
                lines.append(f"    - {s}")

        # Overlaps
        if m.overlap_map:
            overlap_slice = list(m.overlap_map)
            top_overlap = overlap_slice[0] if overlap_slice else None
            if top_overlap:
                lines.append(f"Overlaps: {len(m.overlap_map)} pairs (worst: {top_overlap.get('overlap_pct', 0):.0f}%)")

        # Elements
        if m.element_map:
            lines.append(f"Detected elements: {len(m.element_map)}")

        # Stability
        if m.stability_map:
            static = sum(1 for v in m.stability_map.values() if v == "STATIC")
            dynamic = sum(1 for v in m.stability_map.values() if v in ("DYNAMIC", "TRANSIENT"))
            lines.append(f"Stability: {static} static, {dynamic} dynamic regions")

        # Structural Patterns
        if m.detected_structural_patterns:
            pattern_labels = ", ".join([p["label"] for p in m.detected_structural_patterns])
            lines.append(f"Detected patterns: {pattern_labels}")

        return "\n".join(lines)

    # ═══════════════════════  Helpers  ═══════════════════════

    def _build_navigation_recommendations(self, model: WorldModel) -> List[str]:
        """Generates navigation hints based on current state."""
        recs = []

        if model.overlap_map:
            worst = model.overlap_map[0]
            if worst["overlap_pct"] > 60:
                recs.append(f"Heavy overlap detected — consider snapping or minimizing windows")

        if len(model.minimized_windows) > 5:
            recs.append(f"{len(model.minimized_windows)} minimized windows — consider closing unused ones")

        if model.resource_state:
            cpu = model.resource_state.get("total_cpu", 0)
            if cpu > 80:
                recs.append(f"High CPU usage ({cpu}%) — avoid switching to heavy apps")

        return recs

    def detect_patterns(self, element_map: List[Dict], app_class: str) -> List[Dict]:
        """Detects clusters of elements that match known structural patterns."""
        if not getattr(self, "experiential_memory", None): return []
        
        known_patterns = self.experiential_memory.recall_patterns_hierarchical(app_class)
        if not known_patterns: return []
        
        detected = []
        element_labels = [str(el.get("text", el.get("label", ""))).lower() for el in element_map]
        
        for pattern in known_patterns:
            confidence = pattern.matches_elements(element_labels)
            if confidence > 0.7:
                detected.append({
                    "id": pattern.pattern_id,
                    "label": pattern.label,
                    "tier": pattern.tier,
                    "confidence": confidence,
                    "signature_match": True
                })
        
        # Spatial Clustering (Naive: elements centered on screen)
        # In a real impl, we'd use DBSCAN or similar on (x,y)
        return detected

    def _speculative_snip(self, frame, model: WorldModel):
        """Proactively snips and saves STATIC elements for speculative template matching."""
        if not self.memory or not self.cv:
            return

        context_name = model.foreground_state.get("class_name") or model.foreground_title
        if not context_name:
            context_name = "Desktop"

        stability_map = model.stability_map
        import hashlib
        import numpy as np # type: ignore
        
        saved_count = 0
        for elem in list(model.element_map): # Cast to list for safe slicing # type: ignore
            # Cap at 15 speculative snips per frame to avoid lag
            if saved_count >= 15:
                break
                
            x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]
            cx, cy = x + w // 2, y + h // 2
            
            # Check stability (default grid size is 32)
            grid_x, grid_y = cx // 32, cy // 32
            cell_key = f"{grid_x},{grid_y}"
            
            # If we have stability data, enforce STATIC. If not, assume STATIC for now.
            if stability_map:
                if stability_map.get(cell_key, "STATIC") != "STATIC":
                    continue
                    
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
                
            # Crop
            template = frame[int(y):int(y+h), int(x):int(x+w)]
            if template.size == 0:
                continue
                
            label = elem.get("label")
            if not label:
                # Use a stable hash of its pixels
                img_hash = hashlib.md5(template.tobytes()).hexdigest()[:8] # type: ignore
                label = f"speculative_{elem['type'].lower()}_{img_hash}"
                
            # Check if this element hash is already vaguely known
            existing = self.memory.recall_element(context_name, label)
            if existing:
                continue
                
            embedding = None
            if hasattr(self.cv, "extract_element_embedding"):
                try:
                    embedding = self.cv.extract_element_embedding(template)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                except Exception:
                    pass
                    
            data = {
                "coords": [int(x), int(y), int(w), int(h)],
                "type": elem["type"],
                "is_speculative": True,
                "stability_class": "STATIC",
                "interaction_type": "CLICK" if elem["type"] in ["BUTTON", "ICON", "TAB"] else "HOVER"
            }
            
            self.memory.remember_element(
                context=context_name,
                label=label,
                data=data,
                template=template,
                embedding=embedding
            )
            saved_count += 1

    def get_world_model(self) -> WorldModel:
        """Returns the current world model."""
        return self.world_model
