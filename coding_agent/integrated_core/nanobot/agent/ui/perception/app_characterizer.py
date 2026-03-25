"""
AppCharacterizer — Builds rich profiles for application windows.

Analyzes each app's:
  1. Identity: name, category, purpose
  2. UI Topology: bars, tabs, panels, input boxes
  3. Interaction Map: clickable, typeable, scrollable, draggable elements
  4. Tab Awareness: criticality, management methods, content summaries
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

try:
    import win32gui # type: ignore
except ImportError:
    win32gui = None


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class AppProfile:
    """Rich profile for a single application window."""
    identity: Dict[str, Any] = field(default_factory=lambda: {
        "name": "", "class_name": "", "category": "UNKNOWN",
        "purpose_guess": "", "exe_path": "",
    })
    topology: Dict[str, Any] = field(default_factory=lambda: {
        "title_bar": {"present": True},
        "menu_bar": {"present": False, "items": []},
        "toolbar": {"present": False, "buttons": []},
        "tab_bar": {"present": False, "tab_count": 0, "active_tab": -1, "tab_labels": [], "position": "TOP"},
        "sidebar": {"present": False, "position": "LEFT", "collapsible": False},
        "address_bar": {"present": False, "content_type": ""},
        "status_bar": {"present": False, "position": "BOTTOM"},
        "content_area": {"type": "UNKNOWN", "scrollable": True},
        "input_boxes": [],
    })
    interactions: Dict[str, List] = field(default_factory=lambda: {
        "clickable_elements": [],
        "typeable_regions": [],
        "scrollable_regions": [],
        "draggable_elements": [],
        "keyboard_shortcuts": [],
    })
    tab_awareness: Dict[str, Any] = field(default_factory=lambda: {
        "tab_criticality": "LOW",
        "tab_management": {},
        "tab_content_summary": [],
    })
    resource_usage: Dict[str, Any] = field(default_factory=lambda: {
        "typical_cpu": 0.0, "typical_ram_mb": 0.0,
    })
    usage_guide: str = ""
    relationships: List[str] = field(default_factory=list)
    confidence: float = 0.0
    last_characterized: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self) # type: ignore

    def to_summary(self) -> str:
        """Concise text summary for planner context."""
        cat = self.identity.get("category", "UNKNOWN")
        name = self.identity.get("name", "Unknown")
        purpose = self.identity.get("purpose_guess", "")
        tabs = self.topology.get("tab_bar", {})
        tab_info = f", {tabs.get('tab_count', 0)} tabs" if tabs.get("present") else ""
        has_menu = "menu" if self.topology.get("menu_bar", {}).get("present") else ""
        has_sidebar = "sidebar" if self.topology.get("sidebar", {}).get("present") else ""
        features = ", ".join(filter(None, [has_menu, has_sidebar]))
        return f"{name} [{cat}]{tab_info}. {features}. {purpose}"


# Category classification by window class and process name
APP_CATEGORIES = {
    "BROWSER": ["chrome", "firefox", "edge", "opera", "brave", "vivaldi", "safari"],
    "EDITOR": ["code", "vscode", "sublime", "notepad", "vim", "emacs", "atom", "jetbrains", "pycharm", "intellij", "rider"],
    "TERMINAL": ["cmd", "powershell", "windowsterminal", "mintty", "conemu", "wezterm", "alacritty", "hyper"],
    "MEDIA": ["vlc", "spotify", "groove", "foobar", "itunes", "media player"],
    "OFFICE": ["word", "excel", "powerpoint", "libreoffice", "openoffice", "onenote"],
    "CHAT": ["slack", "discord", "teams", "telegram", "whatsapp", "signal", "zoom"],
    "FILE_MANAGER": ["explorer", "totalcmd", "dopus", "fman"],
    "SYSTEM": ["taskmgr", "regedit", "mmc", "control", "settings"],
    "DESIGN": ["figma", "photoshop", "gimp", "illustrator", "inkscape", "blender"],
}

# Category-adaptive semantic utility descriptions
SIDEBAR_SEMANTICS = {
    "BROWSER": "Bookmarks, History, and Reading List",
    "EDITOR":  "File Explorer, Search, Source Control, Extensions",
    "CHAT":    "Channel/Contact Navigation and Threads",
    "FILE_MANAGER": "Folder Tree and Quick Access",
    "DESIGN":  "Layer Panel, Assets, and Tool Options",
}
TAB_SEMANTICS = {
    "BROWSER": "Open web pages for multitasking",
    "EDITOR":  "Open documents/files in the editor",
    "CHAT":    "Conversation threads or channels",
    "OFFICE":  "Multiple open documents",
}
TOOLBAR_SEMANTICS = {
    "BROWSER": "Navigation (Back, Forward, Reload), Extensions",
    "EDITOR":  "Run, Debug, Git, and Extension actions",
    "OFFICE":  "Formatting, Insert, and Review tools",
    "DESIGN":  "Drawing tools, Selection, Transform",
}

# ──────────────────────────  Characterizer  ──────────────────────────

class AppCharacterizer:
    """Analyzes application windows to build rich AppProfiles."""

    def __init__(self, cv_pipeline=None, vision_analyzer=None, memory_agent=None):
        self.cv = cv_pipeline
        self.vision = vision_analyzer
        self.memory = memory_agent
        self._profile_cache: Dict[str, AppProfile] = {}

    def characterize(self, hwnd: int, screenshot_frame=None, process_name: str = "", window_class: str = "") -> AppProfile:
        """
        Full analysis of an application window.
        Builds identity, topology, interactions, tab awareness.
        """
        now = time.time()
        profile = AppProfile(last_characterized=now)

        # Resolve identity
        title = ""
        if win32gui:
            try:
                title = win32gui.GetWindowText(hwnd)
                if not window_class:
                    window_class = win32gui.GetClassName(hwnd)
            except Exception:
                pass

        profile.identity = {
            "name": title,
            "class_name": window_class,
            "category": self._classify_category(title, process_name, window_class),
            "purpose_guess": self._guess_purpose(title, process_name),
            "exe_path": "",
            "process_name": process_name,
        }

        # Check cache
        cache_key = f"{window_class}:{process_name}"
        if cache_key in self._profile_cache:
            cached = self._profile_cache[cache_key]
            # Update title (may change) but keep topology
            cached.identity["name"] = title
            cached.last_characterized = now
            return cached

        # Analyze topology with CV if available
        if screenshot_frame is not None and self.cv is not None:
            topology = self._analyze_topology(screenshot_frame, profile.identity["category"])
            profile.topology.update(topology) # type: ignore

        # Build interaction map
        profile.interactions = self._build_interaction_map(profile)

        # Tab awareness
        profile.tab_awareness = {
            "tab_criticality": self._assess_tab_criticality(
                profile.identity["category"],
                profile.topology.get("tab_bar", {}).get("tab_count", 0)
            ),
            "tab_management": self._get_tab_management(profile.identity["category"]),
            "tab_content_summary": [],
        }

        # Usage guide
        profile.usage_guide = self._generate_usage_guide(profile)
        profile.confidence = 0.6 if screenshot_frame is None else 0.8

        # Cache
        self._profile_cache[cache_key] = profile
        return profile

    # ═══════════════════════  Category Classification  ═══════════════════════

    def _classify_category(self, title: str, process_name: str, window_class: str) -> str:
        """Categorizes app by matching process/class/title against known patterns."""
        search_str = f"{title} {process_name} {window_class}".lower()

        for category, keywords in APP_CATEGORIES.items():
            for kw in keywords:
                if kw in search_str:
                    return category

        return "UNKNOWN"

    def _guess_purpose(self, title: str, process_name: str) -> str:
        """Generates a one-sentence purpose guess."""
        search = f"{title} {process_name}".lower()

        if any(kw in search for kw in ["chrome", "firefox", "edge", "brave"]):
            return "Web browser for internet browsing and web applications."
        if any(kw in search for kw in ["code", "vscode"]):
            return "Source code editor with debugging and extensions."
        if any(kw in search for kw in ["cmd", "powershell", "terminal"]):
            return "Command-line interface for system commands and scripts."
        if any(kw in search for kw in ["explorer"]):
            return "File manager for browsing and managing files/folders."
        if any(kw in search for kw in ["notepad"]):
            return "Simple text editor for plain text files."
        if any(kw in search for kw in ["discord", "slack", "teams"]):
            return "Communication app for messaging and collaboration."
        if any(kw in search for kw in ["taskmgr"]):
            return "System monitor for processes, performance, and services."

        return f"Application: {process_name or title}"

    # ═══════════════════════  Topology Analysis  ═══════════════════════

    def _analyze_topology(self, frame, category: str) -> Dict:
        """Analyzes UI topology using CV pipeline."""
        topology = {}

        if frame is None or self.cv is None:
            return topology

        h, w = frame.shape[:2] # type: ignore

        # Detect bars by scanning edge regions
        topology["menu_bar"] = self._detect_menu_bar(frame, category)
        topology["toolbar"] = self._detect_toolbar(frame, category)
        topology["tab_bar"] = self._detect_tab_bar(frame, category)
        topology["sidebar"] = self._detect_sidebar(frame, category)
        topology["status_bar"] = self._detect_status_bar(frame)
        topology["address_bar"] = self._detect_address_bar(frame, category)

        # Input boxes
        topology["input_boxes"] = self._detect_input_boxes(frame) # type: ignore

        return topology

    def _detect_menu_bar(self, frame, category: str) -> Dict:
        """Detects menu bar (horizontal strip near top with text labels)."""
        h, w = frame.shape[:2] # type: ignore
        # Menu bar is typically in the top 60px, full width
        strip = frame[0:min(60, h), :]

        if self.cv:
            text_regions = self.cv.detect_text_regions(strip, merge_distance=10)
            if len(text_regions) >= 2:
                return {
                    "present": True,
                    "items": [{"rect": (tr["x"], tr["y"], tr["w"], tr["h"])} for tr in text_regions],
                    "item_count": len(text_regions),
                    "semantic_utility": "Application-level commands and settings",
                }

        # Heuristic for known categories
        if category in ("EDITOR", "OFFICE", "DESIGN"):
            return {"present": True, "items": [], "item_count": 0, "semantic_utility": "Application-level commands and settings"}

        return {"present": False, "items": [], "semantic_utility": "Application-level commands and settings"}

    def _detect_toolbar(self, frame, category: str) -> Dict:
        """Detects toolbar (icon-rich bar below menu)."""
        h, w = frame.shape[:2] # type: ignore
        strip = frame[40:min(100, h), :]

        if self.cv:
            elements = self.cv.detect_ui_elements(strip, min_area=100)
            icon_count = sum(1 for e in elements if e.element_type == "ICON")
            button_count = sum(1 for e in elements if e.element_type == "BUTTON")
            if icon_count + button_count >= 3:
                return {
                    "present": True,
                    "buttons": [{"type": e.element_type, "x": e.x, "y": e.y + 40} for e in elements[:20]],
                    "semantic_utility": TOOLBAR_SEMANTICS.get(category, "Quick access to frequent actions and tools"),
                }

        return {"present": False, "buttons": [], "semantic_utility": TOOLBAR_SEMANTICS.get(category, "Quick access to frequent actions and tools")}

    def _detect_tab_bar(self, frame, category: str) -> Dict:
        """Detects tab bar."""
        h, w = frame.shape[:2] # type: ignore
        # Tabs usually in top 50px for browsers, or top 40px for editors
        strip = frame[0:min(50, h), :]

        if self.cv:
            elements = self.cv.detect_ui_elements(strip, min_area=100)
            # Tabs are typically similar-width adjacent rectangles
            tab_candidates = [e for e in elements if e.element_type in ("BUTTON", "STRUCTURAL") and e.width > 50]
            if len(tab_candidates) >= 2:
                return {
                    "present": True,
                    "tab_count": len(tab_candidates),
                    "active_tab": 0,
                    "tab_labels": [],
                    "position": "TOP",
                    "semantic_utility": TAB_SEMANTICS.get(category, "Workspace or Document isolation for multitasking"),
                }

        # Heuristic for browsers/editors
        if category in ("BROWSER", "EDITOR"):
            return {"present": True, "tab_count": 1, "active_tab": 0, "tab_labels": [], "position": "TOP", "semantic_utility": TAB_SEMANTICS.get(category, "Workspace or Document isolation for multitasking")}

        return {"present": False, "tab_count": 0, "active_tab": -1, "tab_labels": [], "position": "TOP", "semantic_utility": TAB_SEMANTICS.get(category, "Workspace or Document isolation for multitasking")}

    def _detect_sidebar(self, frame, category: str) -> Dict:
        """Detects sidebar panel."""
        h, w = frame.shape[:2] # type: ignore
        # Sidebar is typically on the left, narrow strip
        strip = frame[:, 0:min(60, w)]

        if self.cv:
            elements = self.cv.detect_ui_elements(strip, min_area=50)
            if len(elements) >= 3:
                return {"present": True, "position": "LEFT", "collapsible": True, "semantic_utility": SIDEBAR_SEMANTICS.get(category, "Primary Navigation or Context Explorer")}

        if category in ("EDITOR", "FILE_MANAGER", "CHAT"):
            return {"present": True, "position": "LEFT", "collapsible": True, "semantic_utility": SIDEBAR_SEMANTICS.get(category, "Primary Navigation or Context Explorer")}

        return {"present": False, "position": "LEFT", "collapsible": False, "semantic_utility": SIDEBAR_SEMANTICS.get(category, "Primary Navigation or Context Explorer")}

    def _detect_status_bar(self, frame) -> Dict:
        """Detects status bar at bottom."""
        h, w = frame.shape[:2] # type: ignore
        strip = frame[max(0, h - 30):h, :]

        if self.cv:
            text_regions = self.cv.detect_text_regions(strip, merge_distance=10)
            if len(text_regions) >= 1:
                return {"present": True, "position": "BOTTOM", "info_fields": len(text_regions), "semantic_utility": "System metadata and zoom controls"}

        return {"present": False, "position": "BOTTOM", "semantic_utility": "System metadata and zoom controls"}

    def _detect_address_bar(self, frame, category: str) -> Dict:
        """Detects address/URL bar."""
        if category == "BROWSER":
            return {"present": True, "content_type": "URL", "semantic_utility": "URL Navigation and Search Bar"}
        if category == "FILE_MANAGER":
            return {"present": True, "content_type": "PATH", "semantic_utility": "File Path Navigation"}
        return {"present": False, "content_type": "", "semantic_utility": "Unknown Context Input"}

    def _detect_input_boxes(self, frame) -> List[Dict]:
        """Detects input box elements."""
        if not self.cv:
            return []

        elements = self.cv.detect_ui_elements(frame, min_area=100)
        inputs = []
        for e in elements:
            if e.element_type == "TEXT_INPUT":
                inputs.append({
                    "x": e.x, "y": e.y, "width": e.width, "height": e.height,
                    "type": "TEXT",
                    "semantic_utility": "Text Input (Search, Messaging, or Data Entry)",
                })
        return inputs[:10] # type: ignore  # Limit

    # ═══════════════════════  Interaction Map  ═══════════════════════

    def _build_interaction_map(self, profile: AppProfile) -> Dict[str, List]:
        """Builds interaction map from detected topology."""
        interactions = {
            "clickable_elements": [],
            "typeable_regions": [],
            "scrollable_regions": [],
            "draggable_elements": [],
            "keyboard_shortcuts": [],
        }

        category = profile.identity.get("category", "UNKNOWN")

        # Universal keyboard shortcuts
        interactions["keyboard_shortcuts"] = [
            {"shortcut": "Ctrl+C", "action": "Copy", "context": "global"},
            {"shortcut": "Ctrl+V", "action": "Paste", "context": "global"},
            {"shortcut": "Ctrl+Z", "action": "Undo", "context": "global"},
            {"shortcut": "Ctrl+S", "action": "Save", "context": "global"},
            {"shortcut": "Ctrl+W", "action": "Close tab", "context": "tabs"},
            {"shortcut": "Alt+F4", "action": "Close window", "context": "global"},
        ]

        # Category-specific shortcuts
        if category == "BROWSER":
            interactions["keyboard_shortcuts"].extend([
                {"shortcut": "Ctrl+T", "action": "New tab", "context": "browser"},
                {"shortcut": "Ctrl+L", "action": "Focus address bar", "context": "browser"},
                {"shortcut": "Ctrl+Tab", "action": "Next tab", "context": "browser"},
                {"shortcut": "F5", "action": "Refresh", "context": "browser"},
            ])
        elif category == "EDITOR":
            interactions["keyboard_shortcuts"].extend([
                {"shortcut": "Ctrl+P", "action": "Quick open file", "context": "editor"},
                {"shortcut": "Ctrl+Shift+P", "action": "Command palette", "context": "editor"},
                {"shortcut": "Ctrl+`", "action": "Toggle terminal", "context": "editor"},
            ])
        elif category == "TERMINAL":
            interactions["keyboard_shortcuts"].extend([
                {"shortcut": "Ctrl+C", "action": "Interrupt command", "context": "terminal"},
                {"shortcut": "Tab", "action": "Autocomplete", "context": "terminal"},
                {"shortcut": "Up arrow", "action": "Previous command", "context": "terminal"},
            ])

        # Scrollable regions
        if profile.topology.get("content_area", {}).get("scrollable", True):
            interactions["scrollable_regions"].append({
                "position": "content_area", "direction": "VERTICAL",
            })

        # Draggable (title bar, tab bar)
        interactions["draggable_elements"].append({
            "label": "title_bar", "position": "TOP", "drag_type": "MOVE",
        })
        if profile.topology.get("tab_bar", {}).get("present"):
            interactions["draggable_elements"].append({
                "label": "tabs", "position": "TOP", "drag_type": "REORDER",
            })

        return interactions

    # ═══════════════════════  Tab Awareness  ═══════════════════════

    def _assess_tab_criticality(self, category: str, tab_count: int) -> str:
        """Determines how important tab awareness is for this app."""
        if category in ("BROWSER", "EDITOR") and tab_count > 1:
            return "HIGH"
        if category in ("BROWSER", "EDITOR"):
            return "MEDIUM"
        if tab_count > 1:
            return "MEDIUM"
        return "LOW"

    def _get_tab_management(self, category: str) -> Dict:
        """Returns tab management methods for the app category."""
        if category == "BROWSER":
            return {
                "add_method": "Ctrl+T or click '+' button",
                "remove_method": "Ctrl+W or click 'x' on tab",
                "navigate_method": "Click tab or Ctrl+Tab / Ctrl+Shift+Tab",
                "reorder_method": "Drag tab to new position",
            }
        if category == "EDITOR":
            return {
                "add_method": "Ctrl+N or open file via Ctrl+O/Ctrl+P",
                "remove_method": "Ctrl+W",
                "navigate_method": "Click tab or Ctrl+Tab",
                "reorder_method": "Drag tab",
            }
        return {}

    # ═══════════════════════  Usage Guide  ═══════════════════════

    def _generate_usage_guide(self, profile: AppProfile) -> str:
        """Generates a 'how to start using this app' guide based on profile."""
        category = profile.identity.get("category", "UNKNOWN")
        name = profile.identity.get("name", "this application")

        guides = {
            "BROWSER": f"Open '{name}': Click address bar (Ctrl+L) to navigate. Use Ctrl+T for new tabs. Right-click for context menus.",
            "EDITOR": f"Open '{name}': Use File > Open or Ctrl+O to open files. Ctrl+P for quick-open. Terminal via Ctrl+`. Sidebar for file tree.",
            "TERMINAL": f"Open '{name}': Type commands and press Enter. Use Tab for autocomplete. Ctrl+C to interrupt. Up arrow for history.",
            "CHAT": f"Open '{name}': Select a conversation from sidebar. Type message in input box at bottom. Enter to send.",
            "FILE_MANAGER": f"Open '{name}': Navigate folders by double-clicking. Right-click for options. Address bar for direct path entry.",
            "OFFICE": f"Open '{name}': Use ribbon toolbar for formatting. File tab for open/save. Ctrl+S to save frequently.",
        }

        return guides.get(category, f"Open '{name}': Explore menus and toolbar for available actions. Look for input fields and buttons.")

    # ═══════════════════════  Cache  ═══════════════════════

    def get_cached_profile(self, window_class: str, process_name: str) -> Optional[AppProfile]:
        """Retrieves a cached profile if available."""
        key = f"{window_class}:{process_name}"
        return self._profile_cache.get(key)

    def clear_cache(self):
        """Clears all cached profiles."""
        self._profile_cache.clear()
