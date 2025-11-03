# Model gating rules (automated)

Purpose: numeric criteria used by CI/CD to decide promotion or rollback of canary.

Rules:
- Accuracy threshold: model.metrics.accuracy >= 0.70 required for automatic promotion.
- Health check: /health/ must return ready=true.
- Smoke predict: /predict/ must return a valid prediction for sample input.
- Error rate: HTTP 5xx in canary must remain < 1% during observation window.
- Latency: 95th percentile API latency < 500ms (optional, requires metrics).

Notes:
- These rules are intentionally conservative for production. Adjust thresholds per domain.
- If any automated check fails, system performs `helm uninstall <release>-canary` and notifies operators.
