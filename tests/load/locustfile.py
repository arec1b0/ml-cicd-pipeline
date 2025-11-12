"""
Locust load tests for /predict endpoint.

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Run with web UI:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Run headless:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 60s
"""

from __future__ import annotations

import random
from locust import HttpUser, task, between


class MLPredictUser(HttpUser):
    """Locust user class for ML prediction endpoint load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Sample feature vectors for Iris dataset (4 features)
        self.feature_samples = [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.0, 3.3, 1.4, 0.2],
            [6.3, 3.3, 6.0, 2.5],
            [5.5, 3.2, 1.5, 0.3],
        ]
    
    @task(3)
    def predict_single(self):
        """Test single prediction request."""
        payload = {
            "features": [random.choice(self.feature_samples)]
        }
        with self.client.post(
            "/predict/",
            json=payload,
            catch_response=True,
            name="predict_single"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def predict_batch(self):
        """Test batch prediction request."""
        # Random batch size between 1 and 10
        batch_size = random.randint(1, 10)
        payload = {
            "features": random.choices(self.feature_samples, k=batch_size)
        }
        with self.client.post(
            "/predict/",
            json=payload,
            catch_response=True,
            name="predict_batch"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == batch_size:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get(
            "/health/",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

