"""
Read-write lock implementation for async operations.

This module provides a read-write lock that allows multiple concurrent readers
or a single exclusive writer, improving scalability for model operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AsyncRWLock:
    """Asynchronous read-write lock.

    This lock allows multiple concurrent readers or a single exclusive writer.
    Readers can acquire the lock simultaneously, but writers require exclusive access.

    Example:
        lock = AsyncRWLock()
        
        # Multiple readers can read concurrently
        async with lock.read_lock():
            # Read operation
            pass
        
        # Writers get exclusive access
        async with lock.write_lock():
            # Write operation
            pass
    """

    def __init__(self):
        """Initialize the read-write lock."""
        self._read_ready = asyncio.Condition(asyncio.Lock())
        self._readers = 0

    async def read_lock(self):
        """Acquire a read lock.

        Multiple readers can acquire read locks simultaneously.
        This method blocks if a writer is currently holding the lock.
        """
        async with self._read_ready:
            self._readers += 1
            logger.debug(f"Read lock acquired (readers: {self._readers})")

    async def read_unlock(self):
        """Release a read lock."""
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
            logger.debug(f"Read lock released (readers: {self._readers})")

    async def write_lock(self):
        """Acquire a write lock.

        This method blocks until all readers have released their locks.
        Only one writer can hold the lock at a time.
        """
        async with self._read_ready:
            while self._readers > 0:
                await self._read_ready.wait()
            logger.debug("Write lock acquired")

    async def write_unlock(self):
        """Release a write lock."""
        async with self._read_ready:
            self._read_ready.notify_all()
            logger.debug("Write lock released")

    async def __aenter__(self):
        """Async context manager entry (for read lock)."""
        await self.read_lock()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit (for read lock)."""
        await self.read_unlock()

    class ReadContext:
        """Context manager for read locks."""

        def __init__(self, lock: AsyncRWLock):
            self.lock = lock

        async def __aenter__(self):
            await self.lock.read_lock()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.lock.read_unlock()

    class WriteContext:
        """Context manager for write locks."""

        def __init__(self, lock: AsyncRWLock):
            self.lock = lock

        async def __aenter__(self):
            await self.lock.write_lock()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.lock.write_unlock()

    def read(self) -> ReadContext:
        """Get a read lock context manager."""
        return self.ReadContext(self)

    def write(self) -> WriteContext:
        """Get a write lock context manager."""
        return self.WriteContext(self)

