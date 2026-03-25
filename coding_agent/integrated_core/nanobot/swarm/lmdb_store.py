"""LMDB-backed state store for the swarm architecture."""

import json
import os
from pathlib import Path
from typing import Any, Iterator

import lmdb # type: ignore
import msgpack # type: ignore
from loguru import logger # type: ignore


class LMDBStore:
    """
    Zero-latency key-value store based on LMDB.
    Implements the namespace registry from the Swarm Architecture blueprint.
    """

    def __init__(self, workspace: Path, db_name: str = "swarm_state.lmdb", map_size: int = 100 * 1024 * 1024):
        self.db_path = workspace / db_name
        
        # Ensure parent directories exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize LMDB environment
        self.env = lmdb.open(
            str(self.db_path),
            map_size=map_size,
            max_dbs=1,
            # Thread-safe read/write without explicit locking needed for short transactions
            lock=True,
            sync=True,  # Safer durability default
        )
        self.db = self.env.open_db(b"primary")
        logger.debug("Initialized LMDB store at {}", self.db_path)

    def _serialize(self, value: dict | list | str | int | float | bool) -> bytes:
        """Serialize mostly with msgpack for speed, fallback to JSON."""
        try:
            return msgpack.packb(value, use_bin_type=True)
        except (TypeError, ValueError):
             return json.dumps(value).encode("utf-8")

    def _deserialize(self, value: bytes) -> Any:
        try:
            return msgpack.unpackb(value, raw=False)
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.FormatError, TypeError, ValueError):
            return json.loads(value.decode("utf-8"))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key. Returns default if not found."""
        with self.env.begin(db=self.db) as txn:
            data = txn.get(key.encode("utf-8"))
            if data is None:
                return default
            try:
                return self._deserialize(data)
            except Exception as e:
                logger.error("Failed to deserialize value for {}: {}", key, e)
                return default

    def put(self, key: str, value: Any) -> None:
        """Store a value by key."""
        try:
            serialized_val = self._serialize(value)
            with self.env.begin(write=True, db=self.db) as txn:
                txn.put(key.encode("utf-8"), serialized_val)
        except Exception as e:
            logger.error("Failed to store value for {}: {}", key, e)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed and was deleted."""
        with self.env.begin(write=True, db=self.db) as txn:
            return txn.delete(key.encode("utf-8"))

    def prefix_scan(self, prefix: str) -> Iterator[tuple[str, Any]]:
        """Yield (key, value) pairs for all keys starting with the given prefix."""
        prefix_bytes = prefix.encode("utf-8")
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix_bytes):
                for k, v in cursor:
                    if not k.startswith(prefix_bytes):
                        break
                    yield (k.decode("utf-8"), self._deserialize(v))

    def prune_expired(self) -> int:
        """Prune logic would go here if TTL was added directly to payload."""
        # For a full TTL implementation, we would wrap values in {"_ttl": ts, "data": ...}
        # and do a full scan to delete. For now, this is a placeholder mimicking the spec.
        return 0

    def close(self):
        self.env.close()
