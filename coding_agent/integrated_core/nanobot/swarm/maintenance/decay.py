"""Trigger Decay."""

from loguru import logger
from ..lmdb_store import LMDBStore

class TriggerDecay:
    """Decays success_rate for unfired learned triggers. Swarm Architecture §21.4."""
    
    def __init__(self, lmdb: LMDBStore):
        self.lmdb = lmdb
        
    def run(self) -> int:
        """Execute decay pass."""
        decayed = 0
        for key, val in self.lmdb.prefix_scan("trigger:def:"):
            if val.get("status") == "ACTIVE":
                rate = val.get("success_rate", 1.0)
                # Apply 5% decay
                new_rate = rate * 0.95
                val["success_rate"] = new_rate
                
                if new_rate < 0.3:
                    val["status"] = "ARCHIVED"
                    logger.info("Archived trigger {} due to decay", val["id"])
                    
                self.lmdb.put(key, val)
                decayed += 1
                
        if decayed > 0:
            logger.debug("TriggerDecay: Applied decay to {} triggers.", decayed)
            
        return decayed
