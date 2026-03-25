import pytest
import time
from nanobot.swarm.lmdb_store import LMDBStore

def test_lmdb_put_get(tmp_path):
    store = LMDBStore(tmp_path)
    store.put("test:key:1", {"name": "test1", "val": 42})
    
    data = store.get("test:key:1")
    assert data is not None
    assert data["name"] == "test1"
    assert data["val"] == 42
    
def test_lmdb_delete(tmp_path):
    store = LMDBStore(tmp_path)
    store.put("test:key:2", {"del": True})
    assert store.get("test:key:2") is not None
    
    store.delete("test:key:2")
    assert store.get("test:key:2") is None
    
def test_lmdb_prefix_scan(tmp_path):
    store = LMDBStore(tmp_path)
    store.put("group:a:1", {"id": 1})
    store.put("group:a:2", {"id": 2})
    store.put("group:b:1", {"id": 3})
    
    results = list(store.prefix_scan("group:a:"))
    assert len(results) == 2
    keys = [k for k, v in results]
    assert "group:a:1" in keys
    assert "group:a:2" in keys

def test_lmdb_ttl_expiry(tmp_path):
    store = LMDBStore(tmp_path)
    # TTL of 1 second
    store.put("ttl:key:1", {"temp": True}, ttl=1)
    assert store.get("ttl:key:1") is not None
    
    time.sleep(1.1)  # wait for expire
    
    assert store.get("ttl:key:1") is None
    count = store.prune_expired()
    assert count >= 1 # it might have pruned it lazily or explicitly
