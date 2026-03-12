"""
Caching layer for emotion detection results
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib

class CacheEntry:
    """Represents a cached entry"""
    
    def __init__(self, value: Any, ttl: int = 3600):
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)

class EmotionCache:
    """Cache for emotion detection predictions"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _hash_key(self, audio_data: bytes) -> str:
        """Generate hash key for audio data"""
        return hashlib.sha256(audio_data).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value"""
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store value in cache"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        self.cache[key] = CacheEntry(value, ttl)
    
    def _evict_oldest(self):
        """Remove oldest entry from cache"""
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k].created_at)
        del self.cache[oldest_key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
