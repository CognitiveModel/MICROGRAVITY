"""
PresumptionEngine — Builds and maintains cached positional knowledge
about UI elements from successful action history.

Presumptions = pre-learned beliefs about where elements typically exist:
  "Search bar is at top-center in Chrome"  → coords=(0.55, 0.04), confidence=0.95
  "Close button is top-right in any app"   → promoted to global

Hierarchy: site → app_instance → app_class → global
Impact weight = f(success_rate, recency, frequency, coord_stability)
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class Presumption:
    """A cached belief about where a UI element typically exists."""
    presumption_id: str
    element_label: str
    element_type: str               # BUTTON, INPUT, ICON, MENU, TAB, etc.
    app_class: str
    app_instance: str
    site: str

    # Positional knowledge
    typical_location: str           # "top-right", "top-center", "sidebar-left"
    typical_coords: List[float]     # [x_norm, y_norm] normalized 0-1
    coord_variance: float = 0.1    # variance of coord observations

    # Scoring
    impact_weight: float = 0.5     # 0-1
    confidence: float = 0.5        # 0-1
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    last_confirmed: float = 0.0

    # Context
    parent_element: str = ""        # "inside toolbar", "inside sidebar"
    adjacent_elements: List[str] = field(default_factory=list)
    edge_cid: str = ""              # correlational ID for structural matching

    # Coord history for variance tracking
    _coord_observations: List[List[float]] = field(default_factory=list)


# ──────────────────────────  PresumptionEngine  ──────────────────────────

class PresumptionEngine:
    """Builds, maintains, and applies experiential presumptions."""

    CONFIDENCE_THRESHOLD = 0.85
    VARIANCE_THRESHOLD = 0.10
    STALE_DAYS = 30

    def __init__(self, storage_dir: str = "presumptions"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Hierarchical presumption store
        self._presumptions: Dict[str, Presumption] = {}  # id → Presumption
        # Quick lookup by label: {(label_lower, app_class): [presumption_ids]}
        self._label_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)  # type: ignore

        self._load()
        self.prune_stale()  # Phase 13.5
        print(f"[PresumptionEngine] Loaded {len(self._presumptions)} presumptions")

    # ═══════════════════════  Build / Update  ═══════════════════════

    def build_presumption(
        self,
        target_label: str,
        coords: List[float],
        element_type: str,
        app_class: str,
        app_instance: str,
        site: str = "",
        parent_element: str = "",
        edge_cid: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Presumption:
        """Creates or updates a presumption from a successful action."""
        # Normalize coords to 0-1
        x_norm = coords[0] / max(screen_width, 1)
        y_norm = coords[1] / max(screen_height, 1)

        # Taskbar/System Tray icons should NOT lead to strong presumptions
        target_lower = target_label.lower()
        if "taskbar" in target_lower or "system tray" in target_lower:
            print(f"[Presumption] Skipping build for dynamic target: '{target_label}'")
            return None # type: ignore
        existing = self._find_existing(target_label, app_class, app_instance, site)

        if existing:
            # Update existing
            existing.success_count += 1
            existing.last_confirmed = time.time()
            existing.last_used = time.time()
            existing._coord_observations.append([x_norm, y_norm])

            # Recalculate typical coords (moving average) and variance
            existing.typical_coords = self._compute_avg_coords(existing._coord_observations)
            existing.coord_variance = self._compute_coord_variance(existing._coord_observations)
            existing.impact_weight = self._compute_impact_weight(existing)
            existing.confidence = self._compute_confidence(existing)

            if edge_cid:
                existing.edge_cid = edge_cid

            print(f"[Presumption] Updated '{target_label}' in {app_instance}: "
                  f"conf={existing.confidence:.2f} weight={existing.impact_weight:.2f} "
                  f"({existing.success_count} OK / {existing.failure_count} FAIL)")

            self._save()
            return existing
        else:
            # Create new
            pid = self._generate_id(target_label, app_class, app_instance, site)
            location = self._classify_location(x_norm, y_norm)

            p = Presumption(
                presumption_id=pid,
                element_label=target_label,
                element_type=element_type,
                app_class=app_class,
                app_instance=app_instance,
                site=site,
                typical_location=location,
                typical_coords=[round(x_norm, 4), round(y_norm, 4)],  # type: ignore
                coord_variance=0.15,  # Initial uncertainty
                impact_weight=0.4,
                confidence=0.4,
                success_count=1,
                failure_count=0,
                last_used=time.time(),
                last_confirmed=time.time(),
                parent_element=parent_element,
                edge_cid=edge_cid,
                _coord_observations=[[x_norm, y_norm]],
            )

            self._presumptions[pid] = p
            self._label_index[(target_label.lower(), app_class)].append(pid)

            # Memory capping for index lists
            if len(self._label_index[(target_label.lower(), app_class)]) > 10:
                self._label_index[(target_label.lower(), app_class)] = self._label_index[(target_label.lower(), app_class)][-10:]  # type: ignore

            print(f"[Presumption] NEW '{target_label}' at {location} in {app_instance}: "
                  f"coords=({x_norm:.3f}, {y_norm:.3f})")

            self._save()
            return p

    # ═══════════════════════  Recall (Hierarchical)  ═══════════════════════

    def recall_presumption(
        self,
        target_label: str,
        app_class: str = "",
        app_instance: str = "",
        site: str = "",
    ) -> Optional[Presumption]:
        """
        Hierachical recall: site-specific → instance → class → global.
        Returns the highest-confidence matching presumption.
        """
        candidates = []

        for pid, p in self._presumptions.items():
            if p.element_label.lower() != target_label.lower():
                continue

            # Score by specificity
            specificity = 0
            if site and p.site == site:
                specificity = 4
            elif app_instance and p.app_instance == app_instance:
                specificity = 3
            elif app_class and p.app_class == app_class:
                specificity = 2
            elif p.app_class == "":
                specificity = 1  # Global
            else:
                continue  # No match at any level

            candidates.append((p, specificity))

        if not candidates:
            return None

        # Sort by specificity (high first), then confidence
        candidates.sort(key=lambda c: (c[1], c[0].confidence), reverse=True)
        return candidates[0][0]

    # ═══════════════════════  Mechanistic Application  ═══════════════════════

    def apply_mechanistic(
        self,
        target_label: str,
        app_class: str,
        app_instance: str = "",
        site: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Optional[Dict]:
        """
        Returns cached coords if presumption is high-confidence.
        Skips VLM entirely → zero-cost action resolution.

        Returns: {"x": pixel_x, "y": pixel_y, "confidence": float, "presumption_id": str}
                 or None if not confident enough.
        """
        p = self.recall_presumption(target_label, app_class, app_instance, site)
        if not p:
            return None

        # Taskbar/System Tray are too dynamic for mechanistic (zero-cost) hits
        target_lower = target_label.lower()
        if "taskbar" in target_lower or "system tray" in target_lower:
            return None

        if p.confidence >= self.CONFIDENCE_THRESHOLD and p.coord_variance < self.VARIANCE_THRESHOLD:
            px = int(p.typical_coords[0] * screen_width)
            py = int(p.typical_coords[1] * screen_height)

            p.last_used = time.time()

            print(f"[Presumption] [FAST] MECHANISTIC hit for '{target_label}': "
                  f"({px},{py}) conf={p.confidence:.2f}")

            return {
                "x": px, "y": py,
                "confidence": p.confidence,
                "presumption_id": p.presumption_id,
                "location": p.typical_location,
                "source": "presumption_cache",
            }

        return None

    # ═══════════════════════  Degradation / Promotion  ═══════════════════════

    def degrade_presumption(self, presumption_id: str, failure_reason: str = ""):
        """Reduces confidence after a failure."""
        p = self._presumptions.get(presumption_id)
        if not p:
            return

        p.failure_count += 1
        p.confidence = self._compute_confidence(p)
        p.impact_weight = self._compute_impact_weight(p)

        print(f"[Presumption] [-] Degraded '{p.element_label}': "
              f"conf={p.confidence:.2f} ({p.success_count} OK / {p.failure_count} FAIL) "
              f"reason={failure_reason}")

        self._save()

    def promote_presumption(self, presumption_id: str):
        """Boosts confidence after verified success."""
        p = self._presumptions.get(presumption_id)
        if not p:
            return

        p.success_count += 1
        p.last_confirmed = time.time()
        p.confidence = self._compute_confidence(p)
        p.impact_weight = self._compute_impact_weight(p)

        print(f"[Presumption] [+] Promoted '{p.element_label}': conf={p.confidence:.2f}")
        self._save()

    # ═══════════════════════  Fast Action Candidates  ═══════════════════════

    def get_fast_action_candidates(
        self,
        app_class: str,
        app_instance: str = "",
        min_weight: float = 0.8,
    ) -> List[Dict]:
        """Returns all presumptions with impact weight above threshold for current context."""
        candidates = []
        for p in self._presumptions.values():
            if p.impact_weight < min_weight:
                continue
            if p.app_class == app_class or p.app_instance == app_instance or p.app_class == "":
                candidates.append({
                    "label": p.element_label,
                    "location": p.typical_location,
                    "coords": p.typical_coords,
                    "confidence": p.confidence,
                    "weight": p.impact_weight,
                    "type": p.element_type,
                })
        candidates.sort(key=lambda c: c["weight"], reverse=True)
        return candidates

    # ═══════════════════════  Staleness  ═══════════════════════

    def prune_stale(self, max_age_days: Optional[int] = None):
        """Removes old low-confidence presumptions."""
        if max_age_days is None:
            max_age_days = self.STALE_DAYS

        cutoff = time.time() - (max_age_days * 86400)
        to_remove = [
            pid for pid, p in self._presumptions.items()
            if p.last_confirmed < cutoff and p.confidence < 0.5
        ]
        for pid in to_remove:
            p = self._presumptions.pop(pid, None)
            if p:
                print(f"[Presumption] [X] Pruned stale '{p.element_label}' "
                      f"(last confirmed {max_age_days}+ days ago)")
        if to_remove:
            self._save()

    # ═══════════════════════  Computation Helpers  ═══════════════════════

    def _compute_impact_weight(self, p: Presumption) -> float:
        """
        impact = 0.35*success_rate + 0.25*recency + 0.20*frequency + 0.10*stability + 0.10*(1-variance)
        """
        total = p.success_count + p.failure_count
        success_rate = p.success_count / max(total, 1)

        # Recency: exponential decay, half-life = 7 days
        age_days = (time.time() - p.last_confirmed) / 86400
        recency = math.exp(-0.1 * age_days)

        # Frequency: log scale
        frequency = min(math.log(total + 1, 10) / 2, 1.0)

        # Stability
        stability = 1.0 - min(p.coord_variance, 1.0)

        weight = (
            0.35 * success_rate +
            0.25 * recency +
            0.20 * frequency +
            0.10 * stability +
            0.10 * (1 - min(p.coord_variance, 1.0))
        )
        return round(min(max(weight, 0), 1), 4)  # type: ignore

    def _compute_confidence(self, p: Presumption) -> float:
        """Bayesian-inspired confidence from success/failure counts."""
        # Beta distribution mean: (alpha) / (alpha + beta)
        alpha = p.success_count + 1
        beta = p.failure_count + 1
        base_conf = alpha / (alpha + beta)

        # Boost if many observations
        n = p.success_count + p.failure_count
        n_factor = min(n / 10, 1.0)  # Ramps up to 1.0 at 10 observations

        return round(base_conf * (0.5 + 0.5 * n_factor), 4)  # type: ignore

    def _compute_avg_coords(self, observations: List[List[float]]) -> List[float]:
        """Moving average of coordinate observations."""
        if not observations:
            return [0.0, 0.0]
        # Weighted toward recent (last 10)
        recent = observations[-10:]  # type: ignore
        avg_x = sum(o[0] for o in recent) / len(recent)
        avg_y = sum(o[1] for o in recent) / len(recent)
        return [round(avg_x, 4), round(avg_y, 4)]  # type: ignore

    def _compute_coord_variance(self, observations: List[List[float]]) -> float:
        """Standard deviation of coordinate observations."""
        if len(observations) < 2:
            return 0.15  # Default uncertainty
        recent = observations[-10:]  # type: ignore
        xs = [o[0] for o in recent]
        ys = [o[1] for o in recent]
        var_x = sum((x - sum(xs) / len(xs))**2 for x in xs) / len(xs)
        var_y = sum((y - sum(ys) / len(ys))**2 for y in ys) / len(ys)
        return round(math.sqrt(var_x + var_y), 4)  # type: ignore

    def _classify_location(self, x_norm: float, y_norm: float) -> str:
        """Classifies a normalized position into a human-readable region."""
        h = "left" if x_norm < 0.33 else ("center" if x_norm < 0.67 else "right")
        v = "top" if y_norm < 0.33 else ("middle" if y_norm < 0.67 else "bottom")
        return f"{v}-{h}"

    def _generate_id(self, label: str, app_class: str, app_instance: str, site: str) -> str:
        raw = f"{label}|{app_class}|{app_instance}|{site}|{time.time()}"
        return "pres_" + hashlib.sha256(raw.encode()).hexdigest()[:12]  # type: ignore

    def _find_existing(self, label: str, app_class: str, app_instance: str, site: str) -> Optional[Presumption]:
        """Finds an existing presumption matching the exact context."""
        for p in self._presumptions.values():
            if (p.element_label.lower() == label.lower() and
                p.app_class == app_class and
                p.app_instance == app_instance and
                p.site == site):
                return p
        return None

    # ═══════════════════════  Persistence  ═══════════════════════

    def _save(self):
        path = os.path.join(self.storage_dir, "presumptions.json")
        try:
            data = {}
            for pid, p in self._presumptions.items():
                d = asdict(p)  # type: ignore
                data[pid] = d
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[PresumptionEngine] Save failed: {e}")

    def _load(self):
        path = os.path.join(self.storage_dir, "presumptions.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for pid, d in data.items():
                d.setdefault("_coord_observations", [])
                d.setdefault("adjacent_elements", [])
                p = Presumption(**d)  # type: ignore
                self._presumptions[pid] = p
                self._label_index[(p.element_label.lower(), p.app_class)].append(pid)
        except Exception as e:
            print(f"[PresumptionEngine] Load failed: {e}")
