"""
LRU cache for model predictions with configurable TTL.

This module provides a caching mechanism for predictions to avoid redundant
model inference calls for identical inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PredictionCache:
    """LRU cache for predictions with TTL support.

    This cache stores prediction results keyed by a hash of the input features.
    Entries expire after a configurable TTL (time-to-live) period.

    Attributes:
        max_size: Maximum number of entries in the cache.
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize the prediction cache.

        Args:
            max_size: Maximum number of entries to store (default: 1000).
            ttl_seconds: Time-to-live for entries in seconds (default: 300 = 5 minutes).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _generate_key(self, features: list[list[float]]) -> str:
        """Generate a cache key from input features.

        Args:
            features: List of feature vectors.

        Returns:
            A hash string representing the input features.
        """
        # Serialize features to JSON for consistent hashing
        features_json = json.dumps(features, sort_keys=True)
        return hashlib.sha256(features_json.encode()).hexdigest()

    def get(self, features: list[list[float]]) -> Optional[list[int]]:
        """Get cached prediction if available and not expired.

        Args:
            features: Input feature vectors.

        Returns:
            Cached predictions if available and not expired, None otherwise.
        """
        key = self._generate_key(features)
        current_time = time.time()

        if key in self._cache:
            cached_time, predictions = self._cache[key]

            # Check if entry has expired
            if current_time - cached_time < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Cache hit for key {key[:8]}...")
                return predictions
            else:
                # Entry expired, remove it
                del self._cache[key]
                logger.debug(f"Cache entry expired for key {key[:8]}...")

        self._misses += 1
        return None

    def set(self, features: list[list[float]], predictions: list[int]) -> None:
        """Store predictions in the cache.

        Args:
            features: Input feature vectors.
            predictions: Model predictions to cache.
        """
        key = self._generate_key(features)
        current_time = time.time()

        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (first) item

        # Store new entry
        self._cache[key] = (current_time, predictions)
        self._cache.move_to_end(key)  # Mark as most recently used

        logger.debug(f"Cached predictions for key {key[:8]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Prediction cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hits, misses, hit rate, and size.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }

