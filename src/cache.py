"""Small thread-safe TTL/LRU cache used for deterministic intermediate results."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class MemoryCache:
    def __init__(self, max_items: int = 256, ttl_seconds: int = 1800):
        self.max_items = max(1, int(max_items))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, default=None):
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return default
            expires_at, value = item
            if expires_at < now:
                self._data.pop(key, None)
                return default
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def get_or_compute(self, key: str, factory: Callable[[], Any]):
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def stable_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()
