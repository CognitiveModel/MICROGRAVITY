"""LMDB Pruner."""

from loguru import logger
from ..lmdb_store import LMDBStore

class LMDBPruner:
    """Scans state and deletes expired TTL entries. Swarm Architecture §21.1."""
    
    def __init__(self, lmdb: LMDBStore):
        self.lmdb = lmdb
        
    def run(self) -> int:
        """Execute pruning pass."""
        count = self.lmdb.prune_expired()
        if count > 0:
            logger.debug("Pruner: Removed {} expired state entries.", count)
        return count
