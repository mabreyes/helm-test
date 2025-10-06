# MLflow Helm Chart

A Helm chart for deploying the MLflow tracking server.

## Quick start

```bash
# create namespace (optional)
kubectl create ns mlflow

# install
helm upgrade --install mlflow ./ -n mlflow \
  --set mlflow.backendStore.type=sqlite \
  --set persistence.enabled=true

# get URL
helm status mlflow -n mlflow
```

## Configuration
See `values.yaml` for all available options, including backend store, artifact store, ingress, persistence, and autoscaling.

## Backend store options
- sqlite (default): stores `mlflow.db` on a PVC
- mysql/postgresql: set `mlflow.backendStore.type` and `mlflow.backendStore.connectionString`

## Artifact store options
- filesystem (default): PVC mounted at `/mlruns`
- s3/gcs/azure: set `mlflow.artifactStore.type` and the respective `cloudStorage.*.enabled` and `cloudStorage.*.env` values

## Security
- Set `serviceAccount.create` and `serviceAccount.annotations` as needed
- Consider enabling `networkPolicy`

## Uninstall
```bash
helm uninstall mlflow -n mlflow
```
