import hashlib
import time
from typing import Optional


class ResponseCache:
        def __init__(self, ttl_seconds: int = 300):
                self.ttl = ttl_seconds
                self.cache = {}
                self.hits = 0
                self.misses = 0

        def make_key(self, query: str):
                normalized = query.lower().strip()
                return hashlib.sha256(normalized.encode()).hexdigest()

        def get(self, query: str):
                key = self.make_key(query)

                if key in self.cache:
                        entry = self.cache[key]

                        if time.time() - entry["timestamp"] < self.ttl:
                                self.hits += 1
                                return entry["response"]

                        else:
                                del self.cache[key]

                self.misses += 1
                return None

        def set(self, query: str, response: str):
                key = self.make_key(query)
                self.cache[key] = {
                        "response": response,
                        "timestamp": time.time(),
                        "query": query
                }

        @property
        def stats(self):
                total = self.hits + self.misses
                hit_rate = (self.hits / total) * 100 if total > 0 else 0

                return {
                        "hits": self.hits,
                        "misses": self.misses,
                        "hit_rate": f"{hit_rate:.2f}%",
                        "cache_size": len(self.cache)
                }