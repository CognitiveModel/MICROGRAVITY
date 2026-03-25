import json
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from typing import Dict, Any, List, Optional
from pathlib import Path

class UIMemoryAgent:
    """
    Manages the 'UI Atlas' - a persistent, multi-session map of UI elements, 
    layouts, and behaviors. This reduces VLM reliance by caching known states.
    """
    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.long_term_dir = self.workspace / "agent_memory" / "long_term"
        self.short_term_dir = self.workspace / "agent_memory" / "short_term"
        self.atlas_path = self.long_term_dir / "ui_atlas.json"
        self.templates_dir = self.long_term_dir / "templates"
        
        # Ensure directories exist
        self.long_term_dir.mkdir(parents=True, exist_ok=True)
        self.short_term_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.atlas = self._load_atlas()

    def _load_atlas(self) -> Dict[str, Any]:
        if self.atlas_path.exists():
            try:
                with open(self.atlas_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[UIMemoryAgent] Error loading atlas: {e}")
        return {
            "contexts": {
                "Desktop": {"type": "FIXED", "elements": {}},
                "Taskbar": {"type": "FIXED", "elements": {}}
            },
            "global_elements": {},
            "layout_patterns": {},
            "app_profiles": {},                  # v3: cached AppProfile dicts per app_class
            "learned_chrome_boundaries": {},      # v3: per-app element boundary maps
            "element_embeddings": {},             # v3: element_key → embedding vector path
            "version": "3.0"
        }

    def save_atlas(self):
        try:
            self.atlas_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.atlas_path, 'w', encoding='utf-8') as f:
                json.dump(self.atlas, f, indent=2)
        except Exception as e:
            print(f"[UIMemoryAgent] Error saving atlas: {e}")

    def classify_context(self, context: str, context_type: str = "DYNAMIC"):
        """Classifies a context as FIXED (Desktop/SysUI) or DYNAMIC (Apps)."""
        if context not in self.atlas["contexts"]:
            self.atlas["contexts"][context] = {"elements": {}, "last_seen": None}
        self.atlas["contexts"][context]["type"] = context_type
        self.save_atlas()

    def record_window_state(self, context: str, rect: List[int]):
        """Records the window boundaries [x, y, w, h] to check for stability later."""
        if context not in self.atlas["contexts"]:
            self.classify_context(context)
        self.atlas["contexts"][context]["last_rect"] = rect
        # We don't necessarily need to persist this if it's session-local, 
        # but for cross-step stability we track it in memory.
        
    def is_context_stable(self, context: str, current_rect: List[int]) -> bool:
        """Checks if the window hasn't moved since last recorded state."""
        if context in self.atlas["contexts"]:
            last_rect = self.atlas["contexts"][context].get("last_rect")
            return last_rect == current_rect
        return False

    def remember_element(self, context: str, label: str, data: Dict[str, Any], template: Optional[np.ndarray] = None, embedding: Optional[List[float]] = None):
        """
        Stores an element's coordinates, type, optional CV template, and optional embedding.
        """
        if context not in self.atlas["contexts"]:
            self.atlas["contexts"][context] = {"elements": {}, "last_seen": None, "type": "DYNAMIC"}
        
        element_key = label.lower()
        
        # Heuristic for invariants: search bars, taskbars, menus are usually invariant
        is_invariant = any(keyword in element_key for keyword in ["search", "taskbar", "start", "menu", "address"]) \
                      or data.get("is_invariant", False)

        self.atlas["contexts"][context]["elements"][element_key] = {
            "coords": data.get("coords"), # [x, y, w, h]
            "type": data.get("type", "unknown"),
            "behavior": data.get("behavior", "static"),
            "description": data.get("description", label),
            "is_invariant": is_invariant,
            # v3 metadata
            "stability_class": data.get("stability_class", "UNKNOWN"),
            "interaction_type": data.get("interaction_type", "CLICK"),
            "typical_position": data.get("typical_position", ""),
            "match_count": data.get("match_count", 0),
            "last_match_confidence": data.get("last_match_confidence", 0.0),
            "discovered_via": data.get("discovered_via", "VLM"),
            "last_verified": data.get("last_verified", 0.0),
            "is_speculative": data.get("is_speculative", False),
        }
        self.atlas["contexts"][context]["last_seen"] = os.path.getmtime(self.atlas_path) if self.atlas_path.exists() else 0
        
        if template is not None:
            template_path = self.templates_dir / f"{context}_{element_key}.png"
            cv2.imwrite(str(template_path), template)
            self.atlas["contexts"][context]["elements"][element_key]["template_path"] = str(template_path)
            
        if embedding is not None:
            # Store embedding path or raw embedding. Since embeddings are floats, we can store them in a numpy file.
            # But the atlas uses a global "element_embeddings" dict for paths.
            emb_path = self.templates_dir / f"{context}_{element_key}_emb.npy"
            np.save(str(emb_path), np.array(embedding, dtype=np.float32))
            self.atlas.setdefault("element_embeddings", {})[f"{context}_{element_key}"] = str(emb_path)
            
        self.save_atlas()

    def recall_element(self, context: str, label: str) -> Optional[Dict[str, Any]]:
        """Retrieves element data from the atlas."""
        element_key = label.lower()
        # 1. Search in specific context
        if context in self.atlas["contexts"]:
            elements = self.atlas["contexts"][context]["elements"]
            if element_key in elements:
                return elements[element_key]
        
        # 2. Search in global elements (e.g., Taskbar, Start Button)
        if element_key in self.atlas["global_elements"]:
            return self.atlas["global_elements"][element_key]
            
        return None

    def store_template(self, context: str, label: str, template: np.ndarray):
        """Stores a CV template for an element in the given context."""
        if context not in self.atlas["contexts"]:
            self.atlas["contexts"][context] = {"elements": {}, "last_seen": 0, "type": "DYNAMIC"}
        
        element_key = label.lower()
        if element_key not in self.atlas["contexts"][context]["elements"]:
             self.atlas["contexts"][context]["elements"][element_key] = {"match_count": 0}
             
        # Use simple context-key-safe filename
        safe_ctx = context.replace(" ", "_").replace("/", "_")
        safe_label = element_key.replace(" ", "_").replace("/", "_")
        template_filename = f"{safe_ctx}_{safe_label}.png"
        template_path = self.templates_dir / template_filename
        
        cv2.imwrite(str(template_path), template)
        self.atlas["contexts"][context]["elements"][element_key]["template_path"] = str(template_path)
        self.save_atlas()
        print(f"[UIMemoryAgent] Template stored for '{label}' in {context}: {template_path}")

    def recall_template(self, context: str, label: str) -> Optional[np.ndarray]:
        """Recalls a stored CV template."""
        element_key = label.lower()
        # Specific context search
        if context in self.atlas["contexts"]:
            elements = self.atlas["contexts"][context]["elements"]
            if element_key in elements and "template_path" in elements[element_key]:
                path = elements[element_key]["template_path"]
                if os.path.exists(path):
                    return cv2.imread(path)
        
        # Global search fallback
        global_els = self.atlas.get("global_elements", {})
        if element_key in global_els and "template_path" in global_els[element_key]:
            path = global_els[element_key]["template_path"]
            if os.path.exists(path):
                return cv2.imread(path)
                
        return None

    def get_context_map(self, context: str) -> Dict[str, Any]:
        """Returns the full element map for a given window/app context."""
        return self.atlas["contexts"].get(context, {"elements": {}})

    def update_layout(self, context: str, screen_size: tuple):
        """Records the typical layout/bounds of a context."""
        if context not in self.atlas["contexts"]:
            self.atlas["contexts"][context] = {"elements": {}}
        self.atlas["contexts"][context]["typical_screen_size"] = screen_size
        self.save_atlas()

    def sync_element(self, context: str, label: str, new_coords: List[int]):
        """
        Updates an element's position in the long-term Atlas. 
        Used when CV detects a shift (e.g., icon moved).
        """
        element_key = label.lower()
        if context in self.atlas["contexts"] and element_key in self.atlas["contexts"][context]["elements"]:
            old_coords = self.atlas["contexts"][context]["elements"][element_key]["coords"]
            if old_coords != new_coords:
                print(f"[UIMemoryAgent] Syncing '{label}' in {context}: {old_coords} -> {new_coords}")
                self.atlas["contexts"][context]["elements"][element_key]["coords"] = new_coords
                self.save_atlas()

    def get_short_term_path(self, filename: str) -> Path:
        """Returns a path within the short-term volatile memory dir."""
        return self.short_term_dir / filename

    # ═══════════════════════  v3: Awareness Stack Methods  ═══════════════════════

    def remember_chrome_boundaries(self, app_class: str, boundaries: dict):
        """Stores learned element boundaries for an app class."""
        serializable = {}
        for eid, b in boundaries.items():
            serializable[eid] = {
                "rect": b.rect if hasattr(b, 'rect') else b.get("rect"),
                "center": b.center if hasattr(b, 'center') else b.get("center"),
                "element_type": b.element_type if hasattr(b, 'element_type') else b.get("element_type"),
                "interaction_type": b.interaction_type if hasattr(b, 'interaction_type') else b.get("interaction_type"),
                "confidence": b.confidence if hasattr(b, 'confidence') else b.get("confidence"),
                "detection_method": b.detection_method if hasattr(b, 'detection_method') else b.get("detection_method"),
            }
        self.atlas["learned_chrome_boundaries"][app_class] = serializable
        self.save_atlas()

    def recall_chrome_boundaries(self, app_class: str) -> Optional[dict]:
        """Retrieves stored chrome boundaries for an app class."""
        return self.atlas.get("learned_chrome_boundaries", {}).get(app_class)

    def remember_app_profile(self, app_class: str, profile_dict: dict):
        """Stores an AppProfile dict for cross-session reuse."""
        self.atlas.setdefault("app_profiles", {})[app_class] = profile_dict
        self.save_atlas()

    def recall_app_profile(self, app_class: str) -> Optional[dict]:
        """Retrieves a cached AppProfile."""
        return self.atlas.get("app_profiles", {}).get(app_class)
