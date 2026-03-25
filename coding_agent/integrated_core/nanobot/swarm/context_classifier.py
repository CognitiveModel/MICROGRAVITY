"""Context Classifier — relevance-scored context inclusion for channel messages.

Provides multi-signal relevance scoring to filter unrelated context while
preserving related and semi-related messages.  Replaces naive "dump all
history" with intelligent classification.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.swarm.engine import SwarmEngine # type: ignore


# ---------------------------------------------------------------------------
# Relevance categories
# ---------------------------------------------------------------------------

class RelevanceCategory(Enum):
    """Relevance classification tiers."""
    DIRECT = "direct"           # 1.0 — same topic/thread, must include
    RELATED = "related"         # 0.7 — clearly connected to current context
    SEMI_RELATED = "semi"       # 0.4 — tangentially relevant
    SUSPECTED = "suspected"     # 0.2 — might be relevant, uncertain
    UNRELATED = "unrelated"     # 0.0 — different topic entirely


# Configurable thresholds
INCLUDE_THRESHOLD = 0.3     # Include messages scoring ≥ this
SUSPECT_THRESHOLD = 0.1     # Tag as [POSSIBLY RELATED] between this and INCLUDE
EXCLUDE_THRESHOLD = 0.1     # Exclude messages scoring < this

# Signal weights (must sum to ~1.0)
WEIGHT_TOPIC = 0.35
WEIGHT_TEMPORAL = 0.15
WEIGHT_SENDER = 0.10
WEIGHT_SESSION = 0.20
WEIGHT_SEMANTIC = 0.15
WEIGHT_CHANNEL = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedMessage:
    """A message with its relevance classification."""
    message: dict[str, Any]
    score: float
    category: RelevanceCategory
    signals: dict[str, float] = field(default_factory=dict)
    tag: str = ""               # e.g. "[POSSIBLY RELATED]"


@dataclass
class ConsolidatedChunk:
    """A group of related messages consolidated into a summary."""
    messages: list[ClassifiedMessage]
    summary: str
    combined_score: float
    token_estimate: int


# ---------------------------------------------------------------------------
# Context Classifier
# ---------------------------------------------------------------------------

class ContextClassifier:
    """Multi-signal relevance scoring for channel message context inclusion.

    Scores messages on 6 dimensions and classifies them into 5 tiers
    to decide what context reaches the LLM prompt.
    """

    def __init__(
        self,
        engine: SwarmEngine | None = None,
        max_context_tokens: int = 4000,
    ):
        self._engine = engine
        self._max_context_tokens = max_context_tokens
        # Compile stop words for topic extraction
        self._stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "up", "down", "over",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "don", "now", "and", "but", "or", "if", "that", "this",
            "these", "those", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "it", "its", "they", "them",
        }

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify_and_filter(
        self,
        history: list[dict[str, Any]],
        current_query: str,
        current_sender: str | None = None,
        current_channel: str | None = None,
        current_session: str | None = None,
    ) -> list[dict[str, Any]]:
        """Classify history messages and return filtered, ordered context.

        Returns messages that score above the inclusion threshold,
        with suspected messages tagged.
        """
        if not history:
            return []

        query_terms = self._extract_terms(current_query)
        now = time.time()

        classified: list[ClassifiedMessage] = []
        for msg in history:
            cm = self._classify_message(
                msg, query_terms, now,
                current_sender, current_channel, current_session,
            )
            classified.append(cm)

        # Filter and sort
        included: list[ClassifiedMessage] = []
        total_tokens = 0

        for cm in sorted(classified, key=lambda c: c.score, reverse=True):
            if cm.score < EXCLUDE_THRESHOLD:
                continue

            msg_tokens = self._estimate_tokens(cm.message.get("content", ""))
            if total_tokens + msg_tokens > self._max_context_tokens: # type: ignore
                continue  # Budget exhausted

            if cm.score >= INCLUDE_THRESHOLD:
                included.append(cm)
                total_tokens += msg_tokens # type: ignore
            elif cm.score >= SUSPECT_THRESHOLD:
                cm.tag = "[POSSIBLY RELATED] "
                included.append(cm)
                total_tokens += msg_tokens # type: ignore

        # Restore chronological order for included messages
        included.sort(key=lambda c: history.index(c.message) if c.message in history else 0)

        # Build output
        result: list[dict[str, Any]] = []
        for cm in included:
            msg_copy = dict(cm.message)
            if cm.tag and isinstance(msg_copy.get("content"), str):
                msg_copy["content"] = cm.tag + msg_copy["content"]
            result.append(msg_copy)

        if len(result) < len(history):
            logger.debug(
                "ContextClassifier: filtered {}/{} messages (included: {}, suspected: {})",
                len(history) - len(result), len(history),
                sum(1 for c in included if not c.tag),
                sum(1 for c in included if c.tag),
            )

        return result

    def _classify_message(
        self,
        msg: dict[str, Any],
        query_terms: set[str],
        now: float,
        sender: str | None,
        channel: str | None,
        session: str | None,
    ) -> ClassifiedMessage:
        """Compute multi-signal relevance score for a single message."""
        signals: dict[str, float] = {}
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # 1. Topic overlap
        msg_terms = self._extract_terms(content)
        signals["topic"] = self._topic_overlap(query_terms, msg_terms)

        # 2. Temporal proximity
        msg_time = self._parse_timestamp(msg)
        signals["temporal"] = self._temporal_score(msg_time, now) if msg_time else 0.5

        # 3. Sender match
        msg_sender = msg.get("sender_id") or msg.get("metadata", {}).get("sender_id")
        signals["sender"] = 1.0 if (sender and msg_sender == sender) else 0.3

        # 4. Session continuity
        msg_session = msg.get("session_key") or msg.get("metadata", {}).get("session_key")
        signals["session"] = 1.0 if (session and msg_session == session) else 0.1

        # 5. Semantic similarity (via VectorStore if available)
        signals["semantic"] = self._semantic_score(content, " ".join(query_terms))

        # 6. Channel scope
        msg_channel = msg.get("channel") or msg.get("metadata", {}).get("channel")
        signals["channel"] = 1.0 if (channel and msg_channel == channel) else 0.3

        # Weighted combination
        score = (
            WEIGHT_TOPIC * signals["topic"]
            + WEIGHT_TEMPORAL * signals["temporal"]
            + WEIGHT_SENDER * signals["sender"]
            + WEIGHT_SESSION * signals["session"]
            + WEIGHT_SEMANTIC * signals["semantic"]
            + WEIGHT_CHANNEL * signals["channel"]
        )

        # Always-include rules
        role = msg.get("role", "")
        if role == "system":
            score = 1.0  # System messages always included
        elif role == "tool":
            # Tool results are always tied to their call
            score = max(score, 0.8)

        # Categorize
        if score >= 0.7:
            category = RelevanceCategory.DIRECT
        elif score >= 0.5:
            category = RelevanceCategory.RELATED
        elif score >= 0.3:
            category = RelevanceCategory.SEMI_RELATED
        elif score >= 0.1:
            category = RelevanceCategory.SUSPECTED
        else:
            category = RelevanceCategory.UNRELATED

        return ClassifiedMessage(
            message=msg,
            score=score,
            category=category,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Signal computation helpers
    # ------------------------------------------------------------------

    def _extract_terms(self, text: str) -> set[str]:
        """Extract meaningful terms from text, filtering stop words."""
        words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        return words - self._stop_words

    def _topic_overlap(self, query_terms: set[str], msg_terms: set[str]) -> float:
        """Compute Jaccard-like topic overlap score."""
        if not query_terms or not msg_terms:
            return 0.0

        intersection = query_terms & msg_terms
        union = query_terms | msg_terms

        if not union:
            return 0.0

        # Weighted: intersection matters more relative to query size
        recall = len(intersection) / len(query_terms) if query_terms else 0
        precision = len(intersection) / len(msg_terms) if msg_terms else 0

        if recall + precision == 0:
            return 0.0

        # F1-like score
        return 2 * (recall * precision) / (recall + precision)

    def _temporal_score(self, msg_time: float, now: float) -> float:
        """Exponential recency decay (1hr half-life)."""
        age_s = max(0, now - msg_time) # type: ignore
        half_life_s = 3600.0  # 1 hour
        return 2 ** (-age_s / half_life_s)

    def _semantic_score(self, content: str, query: str) -> float:
        """Compute semantic similarity via VectorStore if available."""
        if not self._engine:
            # Fallback: simple word overlap ratio
            content_words = set(content.lower().split())
            query_words = set(query.lower().split())
            if not query_words:
                return 0.0
            return len(content_words & query_words) / len(query_words)

        try:
            # Use vector store for proper embedding similarity
            results = self._engine.vector.search(query, k=1)
            if results:
                return min(1.0, results[0][1])
        except Exception:
            pass

        return 0.3  # Default mid-range if vector search fails

    def _parse_timestamp(self, msg: dict[str, Any]) -> float | None:
        """Extract timestamp from message."""
        ts = msg.get("timestamp")
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts)
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return None

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (~4 chars per token)."""
        if not isinstance(text, str):
            return 0
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate_related(
        self, messages: list[ClassifiedMessage], max_summary_tokens: int = 500
    ) -> list[ConsolidatedChunk]:
        """Group related classified messages into consolidated chunks.

        Used for old context — instead of including raw old messages,
        provides summaries.
        """
        if not messages:
            return []

        # Group by category
        groups: dict[RelevanceCategory, list[ClassifiedMessage]] = {}
        for cm in messages:
            groups.setdefault(cm.category, []).append(cm)

        chunks: list[ConsolidatedChunk] = []
        for category, group in groups.items():
            if category == RelevanceCategory.UNRELATED:
                continue

            # Build summary from content
            contents = [
                cm.message.get("content", "")[:200]
                for cm in group
                if isinstance(cm.message.get("content"), str)
            ]
            summary = f"[{category.value.upper()} context — {len(group)} messages] " + " | ".join(contents[:5]) # type: ignore

            if len(summary) > max_summary_tokens * 4:
                summary = summary[:max_summary_tokens * 4] + "..." # type: ignore

            chunks.append(ConsolidatedChunk(
                messages=group,
                summary=summary,
                combined_score=sum(cm.score for cm in group) / len(group),
                token_estimate=self._estimate_tokens(summary),
            ))

        return sorted(chunks, key=lambda c: c.combined_score, reverse=True)
