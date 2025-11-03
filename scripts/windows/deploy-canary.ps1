<#
.SYNOPSIS
  Helper for building image, pushing to registry and performing helm canary deploy from Windows.

.DESCRIPTION
  - Requires: docker, helm, kubectl in PATH and login to registry.
  - Uses environment variables for registry and kubeconfig path.
  - Provides three commands: deploy-canary, promote, rollback.
#>

param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("deploy-canary","promote","rollback")]
  [string] $action
)

$ErrorActionPreference = "Stop"

# Configuration - change or export as environment variables before calling
$registryHost = $env:REGISTRY_HOST
$registryRepo = $env:REGISTRY_REPO
$kubeconfigPath = $env:KUBECONFIG_PATH  # path to kubeconfig file
$releaseName = "ml-model"
$chartPath = "infra/helm/ml-model-chart"

function Build-And-Push([string]$tag) {
    Write-Host "Building image $registryHost/$registryRepo:$tag"
    docker build -t "$registryHost/$registryRepo:$tag" -f docker/Dockerfile .
    docker push "$registryHost/$registryRepo:$tag"
}

function Deploy-Canary([string]$tag, [int]$weight) {
    Write-Host "Deploying canary $tag with weight $weight"
    helm upgrade --install $releaseName $chartPath --kubeconfig $kubeconfigPath --set image.repository="$registryHost/$registryRepo" --set image.tag="$tag"
    helm upgrade --install "$releaseName-canary" $chartPath --kubeconfig $kubeconfigPath --set canary.enabled=true --set canary.image.tag="$tag" --set canary.weight=$weight
}

function Promote([string]$tag) {
    Write-Host "Promoting canary tag $tag to stable"
    helm upgrade --install $releaseName $chartPath --kubeconfig $kubeconfigPath --set image.repository="$registryHost/$registryRepo" --set image.tag="$tag"
    helm uninstall "$releaseName-canary" --kubeconfig $kubeconfigPath || Write-Host "canary release not present"
}

function Rollback-Canary() {
    Write-Host "Rolling back: uninstall canary release"
    helm uninstall "$releaseName-canary" --kubeconfig $kubeconfigPath || Write-Host "no canary release"
}

switch ($action) {
    "deploy-canary" {
        $tag = (Get-Date -Format yyyyMMddHHmmss)
        Build-And-Push $tag
        Deploy-Canary $tag 10
        break
    }
    "promote" {
        $tag = Read-Host "Enter image tag to promote"
        Promote $tag
        break
    }
    "rollback" {
        Rollback-Canary
        break
    }
}
