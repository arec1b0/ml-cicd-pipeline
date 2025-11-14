"""
Prediction history persistence using TimescaleDB.

This module provides functionality to store prediction history in TimescaleDB
for historical analysis and A/B testing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class PredictionHistoryStore:
    """Store prediction history in TimescaleDB.

    This class provides methods to persist prediction requests and responses
    to TimescaleDB for historical analysis and A/B testing.
    """

    def __init__(
        self,
        database_url: str,
        table_name: str = "prediction_history",
        enabled: bool = True,
    ):
        """Initialize the prediction history store.

        Args:
            database_url: PostgreSQL connection string (e.g., postgresql://user:pass@host/db).
            table_name: Name of the table to store predictions.
            enabled: Whether prediction history is enabled.
        """
        self.database_url = database_url
        self.table_name = table_name
        self.enabled = enabled and ASYNCPG_AVAILABLE
        self._pool: Optional[asyncpg.Pool] = None

        if not ASYNCPG_AVAILABLE:
            logger.warning(
                "asyncpg not available, prediction history will be disabled",
                extra={"enabled": False}
            )

    async def initialize(self) -> None:
        """Initialize database connection pool and create table if needed."""
        if not self.enabled:
            return

        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
            )
            await self._create_table()
            logger.info("Prediction history store initialized", extra={"table": self.table_name})
        except Exception as exc:
            logger.error(
                "Failed to initialize prediction history store",
                extra={"error": str(exc), "error_type": type(exc).__name__}
            )
            self.enabled = False

    async def _create_table(self) -> None:
        """Create the prediction history table if it doesn't exist."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            # Create table with TimescaleDB hypertable
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    features JSONB NOT NULL,
                    predictions JSONB NOT NULL,
                    model_version TEXT,
                    model_stage TEXT,
                    correlation_id TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                -- Create hypertable if TimescaleDB extension is available
                SELECT create_hypertable('{self.table_name}', 'timestamp', if_not_exists => TRUE);
            """)

            # Create indexes for common queries
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                ON {self.table_name} (timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_correlation_id 
                ON {self.table_name} (correlation_id) 
                WHERE correlation_id IS NOT NULL;
                
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_model_version 
                ON {self.table_name} (model_version) 
                WHERE model_version IS NOT NULL;
            """)

    async def store(
        self,
        features: list[list[float]],
        predictions: list[int],
        model_version: Optional[str] = None,
        model_stage: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a prediction in the history.

        Args:
            features: Input feature vectors.
            predictions: Model predictions.
            model_version: Model version identifier.
            model_stage: Model stage (e.g., "Production", "Staging").
            correlation_id: Request correlation ID.
            metadata: Additional metadata to store.
        """
        if not self.enabled or not self._pool:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} 
                    (timestamp, features, predictions, model_version, model_stage, correlation_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    datetime.now(timezone.utc),
                    features,
                    predictions,
                    model_version,
                    model_stage,
                    correlation_id,
                    metadata,
                )
        except Exception as exc:
            logger.error(
                "Failed to store prediction history",
                extra={"error": str(exc), "error_type": type(exc).__name__}
            )

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

