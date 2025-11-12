# Load Tests

This directory contains Locust load tests for the MLOps pipeline API.

## Prerequisites

Install Locust:
```bash
poetry add --group dev locust
# or
pip install locust
```

## Running Load Tests

### With Web UI (Interactive)

```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser to access the Locust web UI.

### Headless Mode (CI/CD)

```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 60s
```

Parameters:
- `-u 100`: Number of concurrent users
- `-r 10`: Spawn rate (users per second)
- `-t 60s`: Test duration (60 seconds)

### Custom Configuration

```bash
locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --headless \
  -u 200 \
  -r 20 \
  -t 5m \
  --html=load_test_report.html
```

## Test Scenarios

The load tests include:

1. **Single Prediction** (weight: 3) - Tests single prediction requests
2. **Batch Prediction** (weight: 1) - Tests batch prediction requests (1-10 samples)
3. **Health Check** (weight: 1) - Tests health endpoint

## Metrics

Locust reports:
- Total requests per second (RPS)
- Response time percentiles (p50, p95, p99)
- Error rate
- Number of failures

## Target Performance

Expected performance targets:
- p95 latency: < 200ms for single predictions
- p99 latency: < 500ms for batch predictions
- Error rate: < 0.1%
- Throughput: > 100 RPS

