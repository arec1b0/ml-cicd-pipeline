"""
Locust load tests for ML prediction API.

Usage:
    # Run with Locust web UI
    locust -f tests/load/locustfile.py --host=http://localhost:8000

    # Run headless with 100 users, spawn rate of 10/sec, for 60 seconds
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 60s --headless

    # Run with custom configuration
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 120s --headless \
           --csv=results/load_test
"""

from __future__ import annotations

import random
import json
from locust import HttpUser, task, between, tag


class MLPredictionUser(HttpUser):
    """
    Simulates a user making predictions against the ML API.

    This class defines the behavior of users hitting the prediction endpoint
    with various load patterns and validates responses.
    """

    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts."""
        # Check if service is healthy before starting load test
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

    @task(10)
    @tag("predict", "single")
    def predict_single_sample(self):
        """Make a prediction with a single sample (most common use case)."""
        # Generate random features (assuming 2 features based on typical model)
        features = [[random.uniform(0, 10), random.uniform(0, 10)]]

        payload = {"features": features}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [single]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) == 1:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(5)
    @tag("predict", "batch")
    def predict_small_batch(self):
        """Make a prediction with a small batch (5-10 samples)."""
        batch_size = random.randint(5, 10)
        features = [
            [random.uniform(0, 10), random.uniform(0, 10)]
            for _ in range(batch_size)
        ]

        payload = {"features": features}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name=f"/predict [batch:{batch_size}]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) == batch_size:
                        response.success()
                    else:
                        response.failure(f"Expected {batch_size} predictions")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)
    @tag("predict", "large_batch")
    def predict_large_batch(self):
        """Make a prediction with a larger batch (50-100 samples)."""
        batch_size = random.randint(50, 100)
        features = [
            [random.uniform(0, 10), random.uniform(0, 10)]
            for _ in range(batch_size)
        ]

        payload = {"features": features}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name=f"/predict [large_batch:{batch_size}]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) == batch_size:
                        response.success()
                    else:
                        response.failure(f"Expected {batch_size} predictions")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 413:
                # Payload too large is acceptable for very large batches
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    @tag("explain")
    def explain_prediction(self):
        """Request an explanation for a prediction."""
        features = [[random.uniform(0, 10), random.uniform(0, 10)]]
        payload = {"features": features}

        with self.client.post(
            "/explain/",
            json=payload,
            catch_response=True,
            name="/explain/"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["prediction", "shap_values", "feature_values", "explanation_type"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure("Missing required fields in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(3)
    @tag("health")
    def check_health(self):
        """Check the health endpoint."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") in ["ready", "degraded"]:
                        response.success()
                    else:
                        response.failure(f"Unexpected status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    @tag("metrics")
    def get_metrics(self):
        """Fetch Prometheus metrics."""
        with self.client.get("/metrics", catch_response=True, name="/metrics") as response:
            if response.status_code == 200:
                if len(response.text) > 0:
                    response.success()
                else:
                    response.failure("Empty metrics response")
            else:
                response.failure(f"Got status code {response.status_code}")


class MLPredictionStressUser(HttpUser):
    """
    Stress test user that makes requests as fast as possible.

    This user has no wait time and is used for stress testing the system limits.
    """

    wait_time = between(0, 0.1)  # Very short wait time

    @task
    @tag("stress", "predict")
    def stress_predict(self):
        """Make rapid prediction requests."""
        features = [[random.uniform(0, 10), random.uniform(0, 10)]]
        payload = {"features": features}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [stress]"
        ) as response:
            if response.status_code in [200, 503, 429]:
                # 200 = success, 503 = service unavailable, 429 = rate limited
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")


class MLPredictionEdgeCaseUser(HttpUser):
    """
    User that tests edge cases and error conditions.

    This user intentionally sends malformed requests to test error handling.
    """

    wait_time = between(2, 5)

    @task(1)
    @tag("edge_case", "invalid")
    def send_invalid_features(self):
        """Send request with invalid feature format."""
        invalid_payloads = [
            {"features": []},  # Empty features
            {"features": [[]]},  # Empty sample
            {"wrong_key": [[1, 2]]},  # Wrong key
            {},  # Missing features
        ]

        payload = random.choice(invalid_payloads)

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [invalid]"
        ) as response:
            if response.status_code in [400, 422]:
                # Expecting validation error
                response.success()
            else:
                response.failure(f"Expected 400/422, got {response.status_code}")

    @task(1)
    @tag("edge_case", "dimension_mismatch")
    def send_wrong_dimensions(self):
        """Send request with wrong feature dimensions."""
        # Intentionally wrong number of features
        wrong_feature_counts = [1, 3, 5, 10]
        feature_count = random.choice(wrong_feature_counts)

        features = [[random.uniform(0, 10) for _ in range(feature_count)]]
        payload = {"features": features}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [wrong_dims]"
        ) as response:
            # May succeed if model accepts variable features, or fail with validation error
            if response.status_code in [200, 400, 422]:
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")

    @task(1)
    @tag("edge_case", "special_values")
    def send_special_values(self):
        """Send request with special numeric values (NaN, Inf)."""
        special_features = [
            [[float('nan'), 1.0]],
            [[float('inf'), 1.0]],
            [[float('-inf'), 1.0]],
            [[1e308, 1.0]],  # Very large number
        ]

        payload = {"features": random.choice(special_features)}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [special_values]"
        ) as response:
            # Server should handle these gracefully (either accept or reject with 400)
            if response.status_code in [200, 400, 422]:
                response.success()
            else:
                response.failure(f"Unexpected status code {response.status_code}")
