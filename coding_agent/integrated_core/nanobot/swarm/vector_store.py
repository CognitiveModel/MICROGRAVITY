"""Vector memory store using optional Gemini/Transformers or TF-IDF + LMDB for semantic search.

Ported and adapted from microgravity-copy.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import lmdb # type: ignore
import numpy as np # type: ignore
from loguru import logger # type: ignore


# ── Embedding backends ────────────────────────────────────────────────

class _TfidfEmbedder:
    """Lightweight TF-IDF-based embedder using scikit-learn."""
    def embed(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
        vec = TfidfVectorizer() # type: ignore
        matrix = vec.fit_transform(texts) # type: ignore
        return matrix.toarray().astype(np.float32) # type: ignore

    def embed_query(self, query: str, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
        vec = TfidfVectorizer() # type: ignore
        all_texts = corpus + [query]
        matrix = vec.fit_transform(all_texts).toarray().astype(np.float32) # type: ignore
        return matrix[-1], matrix[:-1] # type: ignore


class _SentenceTransformerEmbedder:
    """Optional high-quality embedder using sentence-transformers."""
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2") # type: ignore
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._get_model().encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        model = self._get_model()
        doc_vecs = model.encode(corpus, normalize_embeddings=True)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        return query_vec, doc_vecs


class _GeminiEmbedder:
    """High-quality embedder using Gemini's model via LiteLLM."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        import litellm # type: ignore # noqa: F401

    def embed(self, texts: list[str]) -> np.ndarray:
        from litellm import embedding # type: ignore
        response = embedding( # type: ignore
            model="gemini/gemini-embedding-2-preview",
            input=texts,
            api_key=self.api_key
        )
        embeddings = [item['embedding'] for item in response['data']] # type: ignore
        matrix = np.array(embeddings, dtype=np.float32) # type: ignore
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) # type: ignore
        norms[norms == 0] = 1e-10 # type: ignore
        return matrix / norms # type: ignore

    def embed_query(self, query: str, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        all_texts = [query] + corpus
        matrix = self.embed(all_texts)
        return matrix[0], matrix[1:]


def _pick_embedder():
    """Pick the best available embedder."""
    try:
        from nanobot.config.loader import load_config # type: ignore
        config = load_config()
        # Handle the new unified provider structure if it exists
        gemini_key = None
        if hasattr(config, "providers") and hasattr(config.providers, "gemini"):
            gemini_key = getattr(config.providers.gemini, "api_key", None)
            
        if gemini_key:
            try:
                embedder = _GeminiEmbedder(gemini_key)
                embedder.embed(["test"])
                logger.info("Using Gemini (gemini-embedding-2-preview) for vector embeddings")
                return embedder
            except Exception as e:
                logger.warning("Gemini embedding failed, falling back: {}", e)
    except Exception:
        pass # config loading failed
        
    try:
        import sentence_transformers  # noqa: F401 # type: ignore
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME", str(Path.home() / ".cache" / "huggingface" / "hub"))
        embedder = _SentenceTransformerEmbedder()
        logger.info("Using TF-IDF for vector embeddings (fast local fallback)") # type: ignore
        return _TfidfEmbedder()
    except Exception as e:
        logger.info(f"Using TF-IDF for vector embeddings ({e})") # type: ignore
        return _TfidfEmbedder()


# ── Main Vector Store ─────────────────────────────────────────────────

class VectorStore:
    """LMDB-backed vector store for semantic search over swarm memory."""

    def __init__(self, workspace: Path, map_size: int = 50 * 1024 * 1024):
        self.workspace = workspace
        db_dir = workspace / "swarm_vectors"
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_dir)
        self._env: lmdb.Environment | None = None
        self._map_size = map_size
        self._embedder = _pick_embedder()

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, map_size=self._map_size, create=True)
        return self._env

    @staticmethod
    def _stable_id(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] # type: ignore

    def _store_entry(self, prefix: str, text: str, labels: list[str] | None = None) -> None:
        env = self._ensure_env()
        key = f"{prefix}{self._stable_id(text)}".encode("utf-8")
        payload: dict[str, Any] = {"text": text}
        if labels:
            payload["labels"] = labels
            
        if not isinstance(self._embedder, _TfidfEmbedder): # type: ignore
            try:
                emb = self._embedder.embed([text])[0]
                payload["embedding"] = emb.tolist()
            except Exception as e:
                logger.warning("Failed to pre-compute embedding: {}", e)
                
        payload_bytes = json.dumps(payload).encode("utf-8")
        with env.begin(write=True) as txn:
            txn.put(key, payload_bytes) # type: ignore

    def _search(self, prefix: str, query: str, n_results: int, labels: list[str] | None = None) -> list[str]:
        env = self._ensure_env()
        entries = []
        prefix_bytes = prefix.encode("utf-8")
        req_set = set(labels) if labels else None

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(prefix_bytes):
                    try:
                        data = json.loads(value.decode("utf-8"))
                        if req_set:
                            doc_labels = set(data.get("labels", [])) # type: ignore
                            if not req_set.intersection(doc_labels): # type: ignore
                                continue
                        entries.append(data)
                    except Exception:
                        continue
                        
        if not entries:
            return []
            
        texts = [e["text"] for e in entries]
        
        if isinstance(self._embedder, _TfidfEmbedder):
            try:
                query_vec, doc_vecs = self._embedder.embed_query(query, texts)
                scores = doc_vecs @ query_vec # type: ignore
                top_indices = np.argsort(scores)[::-1][:n_results] # type: ignore
                return [texts[i] for i in top_indices if scores[i] > 0.0]
            except Exception as e:
                logger.warning("TF-IDF vector search failed: {}", e)
                return []
                
        try:
            stored_vecs = [e.get("embedding") for e in entries]
            missing_indices = [i for i, vec in enumerate(stored_vecs) if not vec]
            
            texts_to_embed = [query]
            for i in missing_indices:
                texts_to_embed.append(texts[i])
                
            matrix = self._embedder.embed(texts_to_embed)
            query_vec = matrix[0]
            new_doc_vecs = matrix[1:]
            
            doc_vecs = []
            missing_idx = 0
            for i in range(len(texts)):
                if stored_vecs[i]:
                    doc_vecs.append(stored_vecs[i])
                else:
                    doc_vecs.append(new_doc_vecs[missing_idx])
                    key = f"{prefix}{self._stable_id(texts[i])}".encode("utf-8")
                    e = entries[i]
                    e["embedding"] = new_doc_vecs[missing_idx].tolist() # type: ignore
                    with env.begin(write=True) as txn:
                        txn.put(key, json.dumps(e).encode("utf-8")) # type: ignore
                    missing_idx += 1
                    
            doc_vecs_np = np.array(doc_vecs, dtype=np.float32)
            scores = doc_vecs_np @ query_vec # type: ignore
            top_indices = np.argsort(scores)[::-1][:n_results] # type: ignore
            return [texts[i] for i in top_indices if scores[i] > 0.0]
        except Exception as e:
            logger.warning("Pre-computed vector search failed: {}", e)
            return []

    # ── Public API ───────────────────────────────────────────────────

    def add_history(self, entry: str, labels: list[str] | None = None) -> None:
        self._store_entry("vh:", entry, labels=labels)

    def add_longterm(self, content: str, labels: list[str] | None = None) -> None:
        for chunk in self._chunk_text(content):
            self._store_entry("vm:", chunk, labels=labels)

    def search_history(self, query: str, n_results: int = 5, labels: list[str] | None = None) -> list[str]:
        return self._search("vh:", query, n_results, labels=labels)

    def search_longterm(self, query: str, n_results: int = 5, labels: list[str] | None = None) -> list[str]:
        return self._search("vm:", query, n_results, labels=labels)
        
    @staticmethod
    def _chunk_text(text: str, max_chunk: int = 500) -> list[str]:
        if not text or not text.strip():
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        for para in paragraphs:
            if len(para) <= max_chunk:
                chunks.append(para)
            else:
                lines = para.split("\n")
                buf = ""
                for line in lines:
                    if len(buf) + len(line) + 1 > max_chunk:
                        if buf:
                            chunks.append(buf.strip())
                        buf = line
                    else:
                        buf = (buf + "\n" + line) if buf else line
                if buf.strip():
                    chunks.append(buf.strip())
        return chunks
