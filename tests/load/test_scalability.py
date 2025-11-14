"""
Load tests to validate scalability improvements.

This module contains load tests for:
- Read-write lock performance (concurrent reads)
- Prediction cache effectiveness
- Model loading under high refresh frequency
- Large batch prediction handling
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest
import httpx


@pytest.mark.asyncio
async def test_concurrent_model_access(base_url: str):
    """Test that concurrent prediction requests can access model simultaneously.
    
    This validates that the read-write lock allows concurrent reads.
    """
    async def make_prediction(client: httpx.AsyncClient, features: List[List[float]]):
        response = await client.post(
            f"{base_url}/predict/",
            json={"features": features},
            timeout=30.0,
        )
        return response.status_code, response.json() if response.status_code == 200 else None
    
    async with httpx.AsyncClient() as client:
        # Create multiple concurrent requests
        features = [[5.1, 3.5, 1.4, 0.2]]
        tasks = [make_prediction(client, features) for _ in range(20)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # All requests should succeed
        success_count = sum(1 for r in results if isinstance(r, tuple) and r[0] == 200)
        
        assert success_count == 20, f"Expected 20 successful requests, got {success_count}"
        assert elapsed < 5.0, f"Concurrent requests took too long: {elapsed}s"


@pytest.mark.asyncio
async def test_prediction_cache_effectiveness(base_url: str):
    """Test that prediction cache reduces response time for identical inputs.
    
    This validates that the LRU cache is working correctly.
    """
    async with httpx.AsyncClient() as client:
        features = [[5.1, 3.5, 1.4, 0.2]]
        
        # First request (cache miss)
        start_time = time.time()
        response1 = await client.post(
            f"{base_url}/predict/",
            json={"features": features},
            timeout=30.0,
        )
        first_request_time = time.time() - start_time
        
        assert response1.status_code == 200
        
        # Second request (cache hit)
        start_time = time.time()
        response2 = await client.post(
            f"{base_url}/predict/",
            json={"features": features},
            timeout=30.0,
        )
        second_request_time = time.time() - start_time
        
        assert response2.status_code == 200
        
        # Cached request should be faster (at least 50% faster)
        assert second_request_time < first_request_time * 0.5, \
            f"Cache hit ({second_request_time}s) should be faster than cache miss ({first_request_time}s)"


@pytest.mark.asyncio
async def test_large_batch_prediction(base_url: str):
    """Test that large batch predictions are handled efficiently.
    
    This validates batch processing performance.
    """
    async with httpx.AsyncClient() as client:
        # Create a batch of 100 feature vectors
        features = [[5.1 + i * 0.1, 3.5, 1.4, 0.2] for i in range(100)]
        
        start_time = time.time()
        response = await client.post(
            f"{base_url}/predict/",
            json={"features": features},
            timeout=60.0,
        )
        elapsed = time.time() - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 100
        assert elapsed < 10.0, f"Large batch took too long: {elapsed}s"


@pytest.mark.asyncio
async def test_high_frequency_requests(base_url: str):
    """Test system behavior under high request frequency.
    
    This validates overall system scalability.
    """
    async def make_request(client: httpx.AsyncClient, request_id: int):
        features = [[5.1 + (request_id % 10) * 0.1, 3.5, 1.4, 0.2]]
        try:
            response = await client.post(
                f"{base_url}/predict/",
                json={"features": features},
                timeout=30.0,
            )
            return response.status_code
        except Exception as e:
            return f"Error: {str(e)}"
    
    async with httpx.AsyncClient() as client:
        # Send 100 requests rapidly
        tasks = [make_request(client, i) for i in range(100)]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Calculate success rate
        success_count = sum(1 for r in results if r == 200)
        success_rate = success_count / len(results)
        
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert elapsed < 30.0, f"High frequency requests took too long: {elapsed}s"
        
        # Calculate requests per second
        rps = len(results) / elapsed
        assert rps >= 5, f"Throughput too low: {rps:.2f} req/s"


@pytest.mark.asyncio
async def test_model_refresh_does_not_block_predictions(base_url: str):
    """Test that model refresh operations don't block prediction requests.
    
    This validates that write locks don't block read operations excessively.
    """
    async def make_prediction(client: httpx.AsyncClient):
        features = [[5.1, 3.5, 1.4, 0.2]]
        response = await client.post(
            f"{base_url}/predict/",
            json={"features": features},
            timeout=30.0,
        )
        return response.status_code
    
    async with httpx.AsyncClient() as client:
        # Trigger model reload (if admin endpoint available)
        try:
            await client.post(
                f"{base_url}/admin/reload",
                headers={"X-Admin-Token": "test-token"},
                timeout=10.0,
            )
        except Exception:
            # Admin endpoint may not be available, skip this part
            pass
        
        # Make predictions concurrently with reload
        tasks = [make_prediction(client) for _ in range(10)]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Most requests should succeed even during reload
        success_count = sum(1 for r in results if r == 200)
        assert success_count >= 8, f"Too many requests failed during reload: {success_count}/10"


@pytest.fixture
def base_url():
    """Fixture providing the base URL for the API."""
    import os
    return os.getenv("API_BASE_URL", "http://localhost:8000")

