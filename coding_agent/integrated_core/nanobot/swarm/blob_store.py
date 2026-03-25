import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from loguru import logger # type: ignore


class BlobStore:
    """
    Manages raw, unstructured data for the Swarm Architecture:
    - exposure_logs
    - experiential_ledger
    - discernment_raw
    - interrupt_traces
    """

    def __init__(self, workspace: Path):
        self.blob_root = workspace / "swarm_blobs"
        self.categories = [
            "exposure_logs",
            "experiential_ledger",
            "discernment_raw",
            "interrupt_traces"
        ]
        
        for category in self.categories:
            cat_path = self.blob_root / category
            os.makedirs(cat_path, exist_ok=True)
            
        logger.debug("Initialized BlobStore at {}", self.blob_root)

    def _get_path(self, category: str, session: str | None, filename: str) -> Path:
        """Resolve the full path for a blob."""
        if category not in self.categories:
            raise ValueError(f"Invalid blob category: {category}")
            
        base_path = self.blob_root / category
        if session:
            base_path = base_path / self._sanitize_session(session)
            os.makedirs(base_path, exist_ok=True)
            
        return base_path / filename

    @staticmethod
    def _sanitize_session(session: str) -> str:
        """Sanitize session string for folder names."""
        return "".join(c if c.isalnum() else "_" for c in session)

    def write_blob(self, category: str, filename: str, data: str | bytes, session: str | None = None) -> str:
        """Write raw data to blob storage. Returns the absolute file path."""
        path = self._get_path(category, session, filename)
        
        mode = "wb" if isinstance(data, bytes) else "w"
        encoding = None if isinstance(data, bytes) else "utf-8"
        
        with open(path, mode, encoding=encoding) as f:
            f.write(data)
            
        return str(path)

    def write_json(self, category: str, filename: str, data: dict | list, session: str | None = None) -> str:
        """Helper to write JSON directly."""
        return self.write_blob(category, filename, json.dumps(data, indent=2), session)

    def read_blob(self, path: str | Path, as_bytes: bool = False) -> str | bytes | None:
        """Read a blob payload."""
        try:
            p = Path(path)
            mode = "rb" if as_bytes else "r"
            encoding = None if as_bytes else "utf-8"
            with open(p, mode, encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error("Error reading blob {}: {}", path, e)
            return None

    def read_json(self, path: str | Path) -> dict | list | None:
        """Helper to read JSON payload directly."""
        content = self.read_blob(path, as_bytes=False)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON blob at {}", path)
        return None

    def list_blobs(self, category: str, session: str | None = None) -> list[str]:
        """List absolute paths to blobs in a category/session."""
        base_path = self.blob_root / category
        if session:
            base_path = base_path / self._sanitize_session(session)
            
        if not base_path.exists():
            return []
            
        # Recursive glob if session wasn't provided, otherwise flat
        pattern = "**/*" if not session else "*"
        return [str(p) for p in base_path.glob(pattern) if p.is_file()]

    def log_exposure(self, session: str, content: str, source: str) -> str:
        """Helper mapping for swarm.md §2.4 exposure_logs."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        h = hashlib.md5(content.encode()).hexdigest()[:6] # type: ignore
        filename = f"{ts}_{source}_{h}.log"
        return self.write_blob("exposure_logs", filename, content, session)

    def log_interrupt(self, interrupt_id: str, trace_data: dict) -> str:
        """Helper mapping for swarm.md §2.4 interrupt_traces."""
        return self.write_json("interrupt_traces", f"{interrupt_id}.json", trace_data)
