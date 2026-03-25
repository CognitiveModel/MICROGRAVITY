"""
DecisionManager — Multi-tier action resolution pipeline.

Routes action requests through the optimal path:
  1. ProcessMemory replay (fastest — 0ms if match exists)
  2. Learned element boundaries (0ms — exact pixel coordinates)
  3. CV cache / template match (< 15ms)
  4. Multi-scale CV + embedding similarity (< 50ms)
  5. Live API (if connected) (< 500ms)
  6. Static VLM fallback (< 2s)

Each successful resolution updates the cache/memory for future reuse.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

try:
    from nanobot.agent.ui.perception.cv_logger import CVLogger # type: ignore
except ImportError:
    CVLogger = None


@dataclass
class Resolution:
    """Result of a decision/resolution attempt."""
    success: bool
    method: str                      # PROCESS_REPLAY | BOUNDARY_LOOKUP | CV_CACHE | CV_MULTISCALE | LIVE_API | STATIC_VLM | FAILED
    target_coords: Optional[Tuple[int, int]] = None   # (x, y) click target
    target_rect: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionManager:
    """Multi-tier routing enriched with experiential memory."""

    def __init__(
        self,
        experiential_memory=None,
        boundary_learner=None,
        cv_pipeline=None,
        stability_classifier=None,
        live_streamer=None,
        vision_analyzer=None,
        ui_memory_agent=None,
        cv_logger=None,
        presumption_engine=None,
    ):
        self.memory = experiential_memory
        self.ui_memory = ui_memory_agent
        self.boundary_learner = boundary_learner
        self.cv = cv_pipeline
        self.stability = stability_classifier
        self.live = live_streamer
        self.vision = vision_analyzer
        self.cv_logger = cv_logger  # CVLogger instance
        self.presumptions = presumption_engine  # Tier 0 cache

        # Cache of recent CV matches: {element_desc: (x, y, w, h, confidence, timestamp)}
        self._cv_cache: Dict[str, Tuple] = {}
        self._cache_ttl = 30.0  # seconds

        # Stats
        self._resolution_stats: Dict[str, int] = {
            "PRESUMPTION_CACHE": 0, "PROCESS_REPLAY": 0, "BOUNDARY_LOOKUP": 0, "CV_CACHE": 0,
            "ATLAS_LOOKUP": 0, "CV_MULTISCALE": 0, "LIVE_API": 0, "STATIC_VLM": 0, "FAILED": 0,
        }

    # ═══════════════════════  Main Resolution  ═══════════════════════

    def resolve_action(
        self,
        action: str,
        target_desc: str,
        current_frame=None,
        app_class: str = "",
        task: str = "",
        template=None,
    ) -> Resolution:
        """
        Attempts to resolve an action through the multi-tier pipeline.
        Returns the first successful resolution.
        """
        t0 = time.perf_counter()
        skipped_tiers = []

        def _log_tier(tier_name, tier_num, res):
            """Helper to log tier resolution via CVLogger."""
            if self.cv_logger:
                self.cv_logger.log_tier_resolution(
                    target=target_desc, tier_name=tier_name, tier_num=tier_num,
                    success=res.success if res else False,
                    confidence=res.confidence if res else 0,
                    latency_ms=res.latency_ms if res else 0,
                    skipped_tiers=skipped_tiers[:], # type: ignore
                )

        # ── Tier 0: Experiential Presumption (Zero-cost) ──
        if current_frame is not None:
            result = self._try_presumption_cache(target_desc, app_class, current_frame.shape[1], current_frame.shape[0]) # type: ignore
            if result and result.success:
                result.latency_ms = (time.perf_counter() - t0) * 1000
                self._resolution_stats["PRESUMPTION_CACHE"] += 1
                _log_tier("PRESUMPTION_CACHE", 0, result)
                return result
        skipped_tiers.append("PRESUMPTION(no_confident_match)")

        # ── Tier 1: Process Memory Replay ──
        result = self._try_process_replay(task, app_class)
        if result and result.success:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            self._resolution_stats["PROCESS_REPLAY"] += 1
            _log_tier("PROCESS_REPLAY", 1, result)
            return result
        skipped_tiers.append("PROCESS_REPLAY(no_match)")

        # ── Tier 2: Learned Element Boundaries ──
        result = self._try_boundary_lookup(target_desc, app_class)
        if result and result.success:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            self._resolution_stats["BOUNDARY_LOOKUP"] += 1
            _log_tier("BOUNDARY_LOOKUP", 2, result)
            return result
        skipped_tiers.append("BOUNDARY(not_learned)")

        # ── Tier 3: CV Cache ──
        result = self._try_cv_cache(target_desc)
        if result and result.success:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            self._resolution_stats["CV_CACHE"] += 1
            _log_tier("CV_CACHE", 3, result)
            return result
        skipped_tiers.append("CV_CACHE(miss)")

        # ── Tier 3.5: Atlas Lookup (Speculative & Saved Elements) ──
        if current_frame is not None:
            result = self._try_atlas_lookup(target_desc, app_class, current_frame)
            if result and result.success:
                result.latency_ms = (time.perf_counter() - t0) * 1000
                self._resolution_stats["ATLAS_LOOKUP"] += 1
                _log_tier("ATLAS_LOOKUP", 3.5, result)
                return result
        skipped_tiers.append("ATLAS(miss)")

        # ── Tier 4: Multi-Scale CV + Embedding Similarity ──
        if current_frame is not None:
            result = self._try_cv_multiscale(target_desc, current_frame, template)
            if result and result.success:
                result.latency_ms = (time.perf_counter() - t0) * 1000
                # Update cache
                if result.target_rect:
                    self._cv_cache[target_desc] = result.target_rect + (result.confidence, time.time()) # type: ignore
                self._resolution_stats["CV_MULTISCALE"] += 1
                _log_tier("CV_MULTISCALE", 4, result)
                return result
        skipped_tiers.append("CV_MULTISCALE(no_match)")

        # ── Tier 5: Live API ──
        result = self._try_live_api(target_desc)
        if result and result.success:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            self._resolution_stats["LIVE_API"] += 1
            _log_tier("LIVE_API", 5, result)
            return result
        skipped_tiers.append("LIVE_API(miss)")

        # ── Tier 6: Static VLM ──
        if current_frame is not None:
            result = self._try_static_vlm(target_desc, current_frame)
            if result and result.success:
                result.latency_ms = (time.perf_counter() - t0) * 1000
                self._resolution_stats["STATIC_VLM"] += 1
                _log_tier("STATIC_VLM", 6, result)
                return result
        skipped_tiers.append("STATIC_VLM(fail)")

        # All tiers failed
        self._resolution_stats["FAILED"] += 1
        elapsed = (time.perf_counter() - t0) * 1000
        fail_result = Resolution(
            success=False,
            method="FAILED",
            latency_ms=elapsed,
            metadata={"reason": f"All resolution tiers failed for '{target_desc}'"},
        )
        _log_tier("FAILED", 0, fail_result)
        return fail_result

    # ═══════════════════════  Tier Implementations  ═══════════════════════

    def _try_presumption_cache(self, target_desc: str, app_class: str, screen_width: int, screen_height: int) -> Optional[Resolution]:
        """Tier 0: Mechanistic resolution based on high-confidence experiential presumptions."""
        if not self.presumptions:
            return None

        result = self.presumptions.apply_mechanistic(
            target_label=target_desc,
            app_class=app_class,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        if result:
            return Resolution(
                success=True,
                method="PRESUMPTION_CACHE",
                target_coords=(result["x"], result["y"]),
                confidence=result["confidence"],
                metadata={"presumption_id": result["presumption_id"], "location": result["location"]}
            )
        return None

    def _try_process_replay(self, task: str, app_class: str) -> Optional[Resolution]:
        """Tier 1: Check if there's a stored process for this task."""
        if not self.memory or not task:
            return None

        proc = self.memory.find_matching_process(task, app_class)
        if proc and proc.category in ("TYPICAL", "SPECIAL", "GENERAL"):
            return Resolution(
                success=True,
                method="PROCESS_REPLAY",
                confidence=min(1.0, proc.success_count / max(proc.run_count, 1)),
                metadata={
                    "process_id": proc.process_id,
                    "task_pattern": proc.task_pattern,
                    "steps": proc.steps,
                    "category": proc.category,
                },
            )
        return None

    def _try_boundary_lookup(self, target_desc: str, app_class: str) -> Optional[Resolution]:
        """Tier 2: Check learned element boundaries."""
        if not self.boundary_learner:
            return None

        # Normalize target description to element_id
        element_id = self._normalize_target(target_desc)

        center = self.boundary_learner.get_clickable_center(app_class, element_id)
        if center:
            boundaries = self.boundary_learner.learned_boundaries.get(app_class, {})
            boundary = boundaries.get(element_id)
            return Resolution(
                success=True,
                method="BOUNDARY_LOOKUP",
                target_coords=center,
                target_rect=boundary.rect if boundary else None,
                confidence=boundary.confidence if boundary else 0.8,
                metadata={"element_id": element_id},
            )
        return None

    def _try_cv_cache(self, target_desc: str) -> Optional[Resolution]:
        """Tier 3: Check recent CV cache."""
        cached = self._cv_cache.get(target_desc)
        if cached:
            x, y, w, h, conf, ts = cached
            if time.time() - ts < self._cache_ttl:
                cx = x + w // 2
                cy = y + h // 2
                return Resolution(
                    success=True,
                    method="CV_CACHE",
                    target_coords=(cx, cy),
                    target_rect=(x, y, x + w, y + h),
                    confidence=conf * 0.9,  # Slight decay for cached
                )
        return None

    def _try_atlas_lookup(self, target_desc: str, app_class: str, frame) -> Optional[Resolution]:
        """Tier 3.5: Check UIMemoryAgent Atlas for saved/speculative elements matching the description."""
        if not self.ui_memory:
            return None
            
        context_name = app_class or "Desktop"
        cmap = self.ui_memory.get_context_map(context_name)
        elements = cmap.get("elements", {})
        
        target_lower = target_desc.lower()
        
        # 1. Exact or partial fuzzy match by description/label
        best_match = None
        for key, data in elements.items():
            if target_lower in key or target_lower in data.get("description", "").lower():
                best_match = data
                break
                
        if best_match and "coords" in best_match: # type: ignore
            template_path = best_match.get("template_path")
            
            # Verify via visual template match if available to handle state changes
            if template_path and self.cv and frame is not None:
                import cv2 # type: ignore
                import os
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    if template is not None:
                        match = self.cv.match_template_multiscale(frame, template) # type: ignore
                        if match and match["confidence"] > 0.7:
                            cx = match["x"] + match["w"] // 2
                            cy = match["y"] + match["h"] // 2
                            return Resolution(
                                success=True,
                                method="ATLAS_LOOKUP",
                                target_coords=(cx, cy),
                                target_rect=(match["x"], match["y"], match["x"] + match["w"], match["y"] + match["h"]),
                                confidence=match["confidence"],
                                metadata={"is_speculative": best_match.get("is_speculative", False), "verified": True}
                            )
                            
            # If no template or visual match failed, but it claims to be static layout, trust coordinates
            if best_match.get("stability_class") == "STATIC":
                x, y, w, h = best_match["coords"]
                return Resolution(
                    success=True,
                    method="ATLAS_LOOKUP",
                    target_coords=(x + w//2, y + h//2),
                    target_rect=(x, y, x+w, y+h),
                    confidence=0.8,
                    metadata={"is_speculative": best_match.get("is_speculative", False), "fallback": "static_coords"}
                )
                
        return None

    def _try_cv_multiscale(self, target_desc: str, frame, template=None) -> Optional[Resolution]:
        """Tier 4: Multi-scale template match + embedding similarity."""
        if not self.cv:
            return None

        # If we have a template, try multi-scale match
        if template is not None:
            match = self.cv.match_template_multiscale(frame, template)
            if match:
                cx = match["x"] + match["w"] // 2
                cy = match["y"] + match["h"] // 2
                return Resolution(
                    success=True,
                    method="CV_MULTISCALE",
                    target_coords=(cx, cy),
                    target_rect=(match["x"], match["y"], match["x"] + match["w"], match["y"] + match["h"]),
                    confidence=match["confidence"],
                    metadata={"scale": match["scale"]},
                )

        # Heuristic: detect all elements and find best match by type/position
        elements = self.cv.detect_ui_elements(frame)
        target_lower = target_desc.lower()

        for el in elements:
            if el.element_type.lower() in target_lower or el.label.lower() in target_lower:
                cx, cy = el.center()
                return Resolution(
                    success=True,
                    method="CV_MULTISCALE",
                    target_coords=(cx, cy),
                    target_rect=(el.x, el.y, el.x + el.width, el.y + el.height),
                    confidence=el.confidence * 0.7,
                    metadata={"element_type": el.element_type},
                )

        return None

    def _try_live_api(self, target_desc: str) -> Optional[Resolution]:
        """Tier 5: Live API query."""
        if not self.live:
            return None

        try:
            # Check if live streamer is connected and has a query method
            if hasattr(self.live, 'query_sync') and hasattr(self.live, 'is_connected'):
                if self.live.is_connected:
                    response = self.live.query_sync(
                        f"Find the exact screen coordinates of: {target_desc}. "
                        f"Return as x,y pixel coordinates."
                    )
                    if response:
                        coords = self._parse_coordinates(response)
                        if coords:
                            return Resolution(
                                success=True,
                                method="LIVE_API",
                                target_coords=coords,
                                confidence=0.75,
                                metadata={"raw_response": response[:200]},
                            )
        except Exception as e:
            print(f"[DecisionManager] Live API failed: {e}")

        return None

    def _try_static_vlm(self, target_desc: str, frame) -> Optional[Resolution]:
        """Tier 6: Static VLM screenshot analysis."""
        if not self.vision:
            return None

        try:
            if hasattr(self.vision, 'find_element_coordinates'):
                result = self.vision.find_element_coordinates(frame, target_desc)
                if result and "x" in result and "y" in result:
                    return Resolution(
                        success=True,
                        method="STATIC_VLM",
                        target_coords=(result["x"], result["y"]),
                        confidence=result.get("confidence", 0.6),
                        metadata={"vlm_method": "find_element_coordinates"},
                    )
        except Exception as e:
            print(f"[DecisionManager] Static VLM failed: {e}")

        return None

    # ═══════════════════════  Optimization  ═══════════════════════

    def should_skip_llm(self, action: str, target_desc: str, app_class: str) -> bool:
        """
        Determines if LLM call can be skipped entirely.
        True if action can be resolved via Process replay or cached boundaries.
        """
        # Check process memory
        if self.memory:
            proc = self.memory.find_matching_process(action, app_class)
            if proc and proc.category == "TYPICAL":
                return True

        # Check boundary lookup
        if self.boundary_learner:
            element_id = self._normalize_target(target_desc)
            center = self.boundary_learner.get_clickable_center(app_class, element_id)
            if center:
                # Also check if region is stable
                if self.stability and self.stability.is_region_static(
                    (center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10)
                ):
                    return True

        return False

    # ═══════════════════════  Stats  ═══════════════════════

    def get_stats(self) -> Dict[str, int]:
        """Returns resolution statistics."""
        return dict(self._resolution_stats)

    def get_summary(self) -> str:
        """Concise text summary."""
        total = sum(self._resolution_stats.values())
        if total == 0:
            return "DecisionManager: No resolutions yet"

        lines = ["Resolution stats:"]
        for method, count in self._resolution_stats.items():
            if count > 0:
                pct = count / total * 100
                lines.append(f"  {method}: {count} ({pct:.0f}%)")
        return "\n".join(lines)

    # ═══════════════════════  Helpers  ═══════════════════════

    def _normalize_target(self, target_desc: str) -> str:
        """Normalizes a target description to a likely element_id."""
        desc = target_desc.lower().strip()

        # Common mappings
        mappings = {
            "close": "close_button",
            "close button": "close_button",
            "minimize": "minimize_button",
            "maximize": "maximize_button",
            "menu": "menu_bar",
            "scroll": "scrollbar_v",
            "title bar": "title_bar",
            "system icon": "system_icon",
        }

        for key, val in mappings.items():
            if key in desc:
                return val

        return desc.replace(" ", "_")

    def _parse_coordinates(self, text: str) -> Optional[Tuple[int, int]]:
        """Attempts to parse x,y coordinates from API response text."""
        import re
        # Try patterns like "x=123, y=456" or "123, 456" or "(123, 456)"
        patterns = [
            r'x\s*[=:]\s*(\d+)\s*,?\s*y\s*[=:]\s*(\d+)',
            r'\((\d+)\s*,\s*(\d+)\)',
            r'(\d{2,4})\s*,\s*(\d{2,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        return None

    def clear_cache(self):
        """Clears the CV resolution cache."""
        self._cv_cache.clear()
