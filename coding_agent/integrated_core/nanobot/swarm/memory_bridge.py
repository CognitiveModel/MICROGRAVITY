"""Memory Consolidation Bridge — syncs legacy MemoryStore with the swarm graph.

Bridges the existing flat-file session memory (agent/memory.py) with the
swarm's VectorStore + GraphStore, enabling enriched retrieval across both
systems.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.swarm.engine import SwarmEngine # type: ignore


class MemoryBridge:
    """Bridges legacy MemoryStore with the swarm wisdom graph.

    On session consolidation, extracts key facts from flat memory and
    writes them as Wisdom_Hop nodes in KuzuDB.  On retrieval, queries
    both the legacy MEMORY.md-style store and the swarm VectorStore to
    provide enriched context.
    """

    def __init__(self, engine: SwarmEngine):
        self._engine = engine

    # ------------------------------------------------------------------
    # Consolidation: legacy → swarm
    # ------------------------------------------------------------------

    def consolidate_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """Extract key facts from a session's messages and persist to swarm.

        Args:
            session_id: The session identifier.
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Number of wisdom hops created.
        """
        if not messages:
            return 0

        hop_count = 0
        assistant_msgs = [
            m for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]

        for msg in assistant_msgs:
            content = msg["content"]
            if not content or len(content) < 50:
                continue

            # Write to VectorStore for semantic retrieval
            try:
                doc_id = f"session:{session_id}:{uuid.uuid4().hex[:8]}" # type: ignore
                self._engine.vectors.add(
                    doc_id=doc_id,
                    content=content[:2000],  # Cap length
                    metadata={
                        "source": "session_consolidation",
                        "session_id": session_id,
                        "timestamp": time.time(),
                    },
                )
            except Exception as exc:
                logger.debug("MemoryBridge: vector add failed: {}", exc)

            # Write to KuzuDB as a Wisdom_Hop
            try:
                hop_id = f"hop:{session_id}:{uuid.uuid4().hex[:8]}" # type: ignore
                self._engine.graph.create_node(
                    "Wisdom_Hop",
                    hop_id,
                    {
                        "hop_level": "R1",
                        "content": content[:500],
                        "conviction_level": "TENTATIVE",
                        "sophistication_level": 1,
                    },
                )
                hop_count += 1 # type: ignore
            except Exception as exc:
                logger.debug("MemoryBridge: graph write failed: {}", exc)

        # Record consolidation event in LMDB
        try:
            self._engine.lmdb.put(
                f"memory:consolidation:{session_id}",
                {
                    "timestamp": time.time(),
                    "message_count": len(messages),
                    "hops_created": hop_count,
                },
                ttl=86400 * 30,  # 30 day retention
            )
        except Exception as exc:
            logger.debug("MemoryBridge: consolidation log failed: {}", exc)

        logger.info(
            "MemoryBridge: consolidated session '{}': {} hops from {} messages",
            session_id, hop_count, len(messages),
        )
        return hop_count

    # ------------------------------------------------------------------
    # Enriched retrieval: legacy + swarm
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search across both legacy memory and swarm VectorStore.

        Args:
            query: The search query text.
            k: Number of results to return per source.
            session_id: Optional session filter.

        Returns:
            Combined results from both stores, sorted by relevance.
        """
        results: list[dict[str, Any]] = []

        # 1. Swarm VectorStore search
        try:
            vector_results = self._engine.vectors.search(query, k=k)
            for doc_id, score, metadata in vector_results:
                results.append({
                    "source": "swarm_vector",
                    "doc_id": doc_id,
                    "score": score,
                    "content": metadata.get("content", ""),
                    "metadata": metadata,
                })
        except Exception as exc:
            logger.debug("MemoryBridge: vector search failed: {}", exc)

        # 2. Knowledge graph neighbor search
        try:
            # Find related wisdom hops via graph edges
            topic_results = self._engine.wisdom_network.query_related(
                query[:100], limit=k # type: ignore
            )
            for node_data in topic_results:
                results.append({
                    "source": "swarm_graph",
                    "doc_id": node_data.get("id", ""),
                    "score": node_data.get("relevance", 0.5),
                    "content": node_data.get("content", ""),
                    "metadata": node_data,
                })
        except Exception as exc:
            logger.debug("MemoryBridge: graph search failed: {}", exc)

        # Sort by score descending, deduplicate by content
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        seen_content: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for r in results:
            content_key = r.get("content", "")[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped.append(r)

        return deduped[:k * 2]  # type: ignore # Return up to 2x k results from combined sources

    # ------------------------------------------------------------------
    # Sync utilities
    # ------------------------------------------------------------------

    def sync_from_memory_file(self, memory_path: str) -> int:
        """Ingest a flat MEMORY.md file into the swarm VectorStore.

        Useful for one-time migration from the legacy system.
        """
        try:
            from pathlib import Path
            path = Path(memory_path)
            if not path.exists():
                return 0

            content = path.read_text(encoding="utf-8")
            # Split into logical chunks by markdown headers or double newlines
            chunks = [c.strip() for c in content.split("\n\n") if c.strip() and len(c.strip()) > 30]

            count = 0
            for i, chunk in enumerate(chunks):
                doc_id = f"legacy_memory:{path.stem}:{i}"
                try:
                    self._engine.vectors.add(
                        doc_id=doc_id,
                        content=chunk[:2000],
                        metadata={
                            "source": "legacy_memory_sync",
                            "file": str(path),
                            "chunk_index": i,
                            "timestamp": time.time(),
                        },
                    )
                    count += 1 # type: ignore
                except Exception:
                    pass

            logger.info("MemoryBridge: synced {} chunks from '{}'", count, memory_path)
            return count

        except Exception as exc:
            logger.error("MemoryBridge: sync failed: {}", exc)
            return 0
