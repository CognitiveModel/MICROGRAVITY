"""
CVLogger — Structured logging for all OpenCV / CV pipeline operations.

Every template match, ORB match, snip save, embedding comparison,
stability scan, and fingerprint computation is logged as a structured
JSON-Lines entry to both the console and a rotating log file.

Two log modes:
  ACTIVE  — Agent-triggered during action resolution (high priority).
  PASSIVE — Background observation cycle scan (cacheable, discoverable).
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ──────────────────────────  Log Entry  ──────────────────────────

@dataclass
class CVLogEntry:
    """A single structured log record."""
    ts: float
    session: str
    op: str                       # TEMPLATE_MATCH, ORB_MATCH, SNIP_SAVE, etc.
    mode: str = "PASSIVE"         # ACTIVE or PASSIVE
    target: str = ""
    matched: bool = False
    confidence: float = 0.0
    coords: Optional[List[int]] = None
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────  CVLogger  ──────────────────────────

class CVLogger:
    """Structured logging for all CV operations.
    
    Outputs structured JSON-Lines to:
      1. Console (prefixed with [CV:<OP>])
      2. Rotating log file in `log_dir/`
    """

    def __init__(self, log_dir: str = "cv_logs", console: bool = True,
                 file_logging: bool = True):
        self.log_dir = log_dir
        self.console = console
        self.file_logging = file_logging
        self._session_id = f"s_{int(time.time())}"
        self._entries: List[CVLogEntry] = []

        # Aggregate counters for session summary
        self._counters: Dict[str, int] = {
            "template_match_attempts": 0,
            "template_match_hits": 0,
            "orb_match_attempts": 0,
            "orb_match_hits": 0,
            "snip_saves": 0,
            "embedding_searches": 0,
            "stability_scans": 0,
            "fingerprint_ops": 0,
            "text_region_scans": 0,
            "edge_detections": 0,
            "structural_diffs": 0,
            "cid_assignments": 0,
            "active_ops": 0,
            "passive_ops": 0,
        }

        if file_logging:
            os.makedirs(log_dir, exist_ok=True)
            self._log_path: Optional[str] = os.path.join(log_dir, f"session_{self._session_id}.jsonl")
        else:
            self._log_path: Optional[str] = None

    # ═══════════════════════  Core Writer  ═══════════════════════

    def _emit(self, entry: CVLogEntry):
        """Writes a log entry to console and/or file."""
        self._entries.append(entry)

        # Update counters
        if entry.mode == "ACTIVE":
            self._counters["active_ops"] += 1
        else:
            self._counters["passive_ops"] += 1

        # Console
        if self.console:
            tag = f"[CV:{entry.op}]"
            mode_tag = f"[{entry.mode}]"
            parts = [tag, mode_tag, f'target="{entry.target}"']
            if entry.confidence > 0:
                parts.append(f"conf={entry.confidence:.4f}")
            if entry.coords:
                parts.append(f"coords=({entry.coords[0]},{entry.coords[1]})") # type: ignore
            parts.append(f"matched={entry.matched}")
            parts.append(f"latency={entry.latency_ms:.1f}ms")
            if entry.details:
                for k, v in entry.details.items():
                    parts.append(f"{k}={v}")
            print(" ".join(parts))

        # File
        if self.file_logging and self._log_path:
            try:
                record = asdict(entry) # type: ignore
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass  # Never crash the agent over logging

    # ═══════════════════════  Template Match  ═══════════════════════

    def log_template_match(self, target: str, scale: float, confidence: float,
                           coords: Optional[List[int]], matched: bool,
                           latency_ms: float, mode: str = "PASSIVE"):
        """Logs a template match attempt with full metadata."""
        self._counters["template_match_attempts"] += 1
        if matched:
            self._counters["template_match_hits"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="TEMPLATE_MATCH", mode=mode,
            target=target, matched=matched,
            confidence=confidence, coords=coords,
            latency_ms=latency_ms,
            details={"scale": scale},
        ))

    # ═══════════════════════  ORB Feature Match  ═══════════════════════

    def log_orb_match(self, target: str, num_keypoints: int, num_matches: int,
                      confidence: float, matched: bool,
                      latency_ms: float, mode: str = "PASSIVE"):
        """Logs an ORB feature matching attempt."""
        self._counters["orb_match_attempts"] += 1
        if matched:
            self._counters["orb_match_hits"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="ORB_MATCH", mode=mode,
            target=target, matched=matched,
            confidence=confidence,
            latency_ms=latency_ms,
            details={"keypoints": num_keypoints, "matches": num_matches},
        ))

    # ═══════════════════════  Snip Save  ═══════════════════════

    def log_snip_save(self, element_id: str, bbox: List[int], source: str,
                      is_speculative: bool, mode: str = "PASSIVE"):
        """Logs when a UI element crop is saved."""
        self._counters["snip_saves"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="SNIP_SAVE", mode=mode,
            target=element_id, matched=True,
            coords=bbox[:2] if bbox else None, # type: ignore
            details={
                "bbox": bbox, "source": source,
                "is_speculative": is_speculative,
            },
        ))

    # ═══════════════════════  Embedding Search  ═══════════════════════

    def log_embedding_compare(self, query_id: str, top_match_id: str,
                              top_score: float, candidates_count: int,
                              latency_ms: float, mode: str = "PASSIVE"):
        """Logs an embedding similarity search."""
        self._counters["embedding_searches"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="EMBEDDING_SEARCH", mode=mode,
            target=query_id, matched=top_score > 0.7,
            confidence=top_score,
            latency_ms=latency_ms,
            details={
                "top_match": top_match_id,
                "candidates": candidates_count,
            },
        ))

    # ═══════════════════════  Stability Scan  ═══════════════════════

    def log_stability_scan(self, num_static: int, num_dynamic: int,
                           num_transient: int, scan_time_ms: float,
                           mode: str = "PASSIVE"):
        """Logs a stability classification pass."""
        self._counters["stability_scans"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="STABILITY_SCAN", mode=mode,
            target="full_screen", matched=True,
            latency_ms=scan_time_ms,
            details={
                "static": num_static, "dynamic": num_dynamic,
                "transient": num_transient,
            },
        ))

    # ═══════════════════════  Text Region  ═══════════════════════

    def log_text_region(self, num_regions: int, total_area_pct: float,
                        latency_ms: float, mode: str = "PASSIVE"):
        """Logs text region detection results."""
        self._counters["text_region_scans"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="TEXT_REGION", mode=mode,
            target="text_scan", matched=num_regions > 0,
            latency_ms=latency_ms,
            details={"regions": num_regions, "area_pct": round(total_area_pct, 2)}, # type: ignore
        ))

    # ═══════════════════════  Fingerprint  ═══════════════════════

    def log_fingerprint(self, element_id: str, similarity: float,
                        state_change_detected: bool,
                        latency_ms: float, mode: str = "PASSIVE"):
        """Logs a color fingerprint computation/comparison."""
        self._counters["fingerprint_ops"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="FINGERPRINT", mode=mode,
            target=element_id, matched=not state_change_detected,
            confidence=similarity,
            latency_ms=latency_ms,
            details={"state_change": state_change_detected},
        ))

    # ═══════════════════════  Active / Passive Wrappers  ═══════════════════════

    def log_active_match(self, target: str, method: str, result: Dict,
                         latency_ms: float):
        """Convenience: logs an ACTIVE match (triggered by agent decision)."""
        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op=f"ACTIVE_{method.upper()}", mode="ACTIVE",
            target=target,
            matched=result.get("matched", result.get("success", False)), # type: ignore
            confidence=result.get("confidence", 0.0),
            coords=result.get("coords"),
            latency_ms=latency_ms,
            details=result,
        ))

    def log_passive_match(self, target: str, method: str, result: Dict,
                          latency_ms: float):
        """Convenience: logs a PASSIVE match (triggered by observation cycle)."""
        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op=f"PASSIVE_{method.upper()}", mode="PASSIVE",
            target=target,
            matched=result.get("matched", result.get("success", False)), # type: ignore
            confidence=result.get("confidence", 0.0),
            coords=result.get("coords"),
            latency_ms=latency_ms,
            details=result,
        ))

    # ═══════════════════════  Decision Manager Logging  ═══════════════════════

    def log_tier_resolution(self, target: str, tier_name: str, tier_num: int,
                            success: bool, confidence: float,
                            latency_ms: float, skipped_tiers: Optional[List[str]] = None):
        """Logs which DecisionManager tier resolved the action."""
        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="TIER_RESOLVE", mode="ACTIVE",
            target=target, matched=success,
            confidence=confidence,
            latency_ms=latency_ms,
            details={
                "tier": tier_num, "tier_name": tier_name,
                "skipped": skipped_tiers or [],
            },
        ))

    # ═══════════════════════  Edge Detection  ═══════════════════════

    def log_edge_detection(self, num_elements: int, edge_density_pct: float,
                           num_cids_assigned: int, latency_ms: float,
                           mode: str = "PASSIVE"):
        """Logs an edge detection + CID assignment pass."""
        self._counters["edge_detections"] += 1
        self._counters["cid_assignments"] += num_cids_assigned

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="EDGE_DETECT", mode=mode,
            target="full_frame", matched=num_elements > 0,
            latency_ms=latency_ms,
            details={
                "elements": num_elements,
                "edge_density_pct": round(edge_density_pct, 2), # type: ignore
                "cids_assigned": num_cids_assigned,
            },
        ))

    def log_structural_diff(self, num_added: int, num_removed: int,
                            num_stable: int, change_ratio: float,
                            latency_ms: float):
        """Logs a structural layout diff between frames."""
        self._counters["structural_diffs"] += 1

        self._emit(CVLogEntry(
            ts=time.time(), session=self._session_id,
            op="STRUCTURAL_DIFF", mode="PASSIVE",
            target="layout_diff", matched=change_ratio < 0.3,
            confidence=1.0 - change_ratio,
            latency_ms=latency_ms,
            details={
                "added": num_added, "removed": num_removed,
                "stable": num_stable, "change_ratio": round(change_ratio, 3), # type: ignore
            },
        ))

    # ═══════════════════════  Session Summary  ═══════════════════════

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns aggregated stats for the current session."""
        total = len(self._entries)
        hits = sum(1 for e in self._entries if e.matched)
        avg_latency = (
            sum(e.latency_ms for e in self._entries) / total
            if total > 0 else 0
        )

        return {
            "session_id": self._session_id,
            "total_operations": total,
            "total_hits": hits,
            "hit_rate": round(hits / total, 3) if total > 0 else 0, # type: ignore
            "avg_latency_ms": round(avg_latency, 2), # type: ignore
            **self._counters,
            "template_hit_rate": (
                round(self._counters["template_match_hits"] / # type: ignore
                      max(1, self._counters["template_match_attempts"]), 3)
            ),
            "orb_hit_rate": (
                round(self._counters["orb_match_hits"] / # type: ignore
                      max(1, self._counters["orb_match_attempts"]), 3)
            ),
        }

    def print_session_summary(self):
        """Prints a formatted session summary to console."""
        s = self.get_session_summary()
        print("\n" + "=" * 60)
        print(f"  CV SESSION SUMMARY — {s['session_id']}")
        print("=" * 60)
        print(f"  Total Ops:        {s['total_operations']}")
        print(f"  Total Hits:       {s['total_hits']} ({s['hit_rate']*100:.1f}%)")
        print(f"  Avg Latency:      {s['avg_latency_ms']:.1f}ms")
        print(f"  Active / Passive: {s['active_ops']} / {s['passive_ops']}")
        print(f"  Template Match:   {s['template_match_hits']}/{s['template_match_attempts']} ({s['template_hit_rate']*100:.1f}%)")
        print(f"  ORB Match:        {s['orb_match_hits']}/{s['orb_match_attempts']} ({s['orb_hit_rate']*100:.1f}%)")
        print(f"  Snip Saves:       {s['snip_saves']}")
        print(f"  Embedding Searches: {s['embedding_searches']}")
        print(f"  Stability Scans:  {s['stability_scans']}")
        print(f"  Fingerprint Ops:  {s['fingerprint_ops']}")
        print(f"  Text Scans:       {s['text_region_scans']}")
        print("=" * 60 + "\n")
