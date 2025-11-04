# Deploy runbook (canary)

1. Trigger workflow: Go to Actions -> "Deploy Canary and Promote" -> Run workflow.
2. Confirm registry credentials and kubeconfig secrets exist.
3. Monitor Actions logs for:
   - image build success
   - helm upgrade success for stable and canary
   - smoke tests pass
4. If smoke tests pass, workflow promotes canary automatically.
5. Refresh the running pods with the new model (skip if `MODEL_AUTO_REFRESH_SECONDS` is non-zero):
   - Retrieve the admin token value from Kubernetes (e.g., `kubectl get secret <secret-name> -o jsonpath='{.data.<key>}' | base64 -d`). If `env.adminTokenSecretValue` is set in the Helm values, the chart creates a secret named `<release>-admin-token` you can reference here.
   - Call the secured reload endpoint: `curl -X POST https://<host>/admin/reload -H "X-Admin-Token: <token>"`
6. If fails, workflow leaves stable unchanged and attempts to uninstall canary.
7. Manual rollback:
   - kubectl --kubeconfig <kubeconfig> get pods -l release=canary
   - kubectl --kubeconfig <kubeconfig> delete deploy <release>-canary
