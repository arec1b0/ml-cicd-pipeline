# Deploy runbook (canary)

1. Trigger workflow: Go to Actions -> "Deploy Canary and Promote" -> Run workflow.
2. Confirm registry credentials and kubeconfig secrets exist.
3. Monitor Actions logs for:
   - image build success
   - helm upgrade success for stable and canary
   - smoke tests pass
4. If smoke tests pass, workflow promotes canary automatically.
5. If fails, workflow leaves stable unchanged and attempts to uninstall canary.
6. Manual rollback:
   - kubectl --kubeconfig <kubeconfig> get pods -l release=canary
   - kubectl --kubeconfig <kubeconfig> delete deploy <release>-canary
