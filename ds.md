Got it—let’s narrow this to what DS folks care about most: **creating/organizing experiments, starting runs, logging params/metrics/artifacts/models, autologging, reading results back, and quickly sanity-checking connectivity**. No server ops—just the tracking workflow from JupyterHub.

---

# JupyterHub → MLflow Tracking: DS-Focused Guide

## TL;DR quick start (copy/paste this cell first)

```python
import mlflow, os
# 1) Point to your team’s MLflow (or rely on env var)
os.environ.setdefault("MLFLOW_TRACKING_URI", "https://mlflow.example.com")

# 2) Choose an experiment (created if missing)
mlflow.set_experiment("team-<name>/project-<name>")

# 3) Smoke test a run
with mlflow.start_run(run_name="connectivity-check") as run:
    mlflow.set_tags({"owner":"<you>", "env":"jhub", "purpose":"smoke"})
    mlflow.log_param("ping", "ok")
    mlflow.log_metric("ping_metric", 1.0)
    print("OK. Run:", run.info.run_id, "Tracking URI:", mlflow.get_tracking_uri())
```

If that runs without error and shows up in the MLflow UI under your experiment, tracking is wired.

---

## 1) Environment variables you’ll actually use

Set these once per Jupyter **user image / profile** (Helm `singleuser.extraEnv`) or in a notebook before use.

**Required**

* `MLFLOW_TRACKING_URI` – URL to the tracking server (e.g., `https://mlflow.example.com`)

**Auth (if your org enabled it)**

* Basic auth: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
* Token/bearer: `MLFLOW_TRACKING_TOKEN`

**TLS (only if custom certs)**

* `REQUESTS_CA_BUNDLE` – path to your org CA `.pem`
  *(Prefer this over disabling TLS verification.)*

> Tip: You can always override in-notebook with `mlflow.set_tracking_uri(...)`.

---

## 2) Experiments: how to organize your runs

**Naming convention (suggested)**

```
team-<teamname>/project-<project>/phase-<exp|prod>
```

Examples:

* `team-risk/credit-approval/phase-exp`
* `team-growth/recsys/phase-exp`

**Create/choose experiment**

```python
mlflow.set_experiment("team-growth/recsys/phase-exp")  # creates if not present
exp = mlflow.get_experiment_by_name("team-growth/recsys/phase-exp")
print(exp.experiment_id, exp.name)
```

**Use a default experiment for scratch**

```python
mlflow.set_experiment("scratch/<username>")
```

---

## 3) Runs: the 90% you’ll do daily

### Minimal pattern

```python
with mlflow.start_run(run_name="rf-baseline") as run:
    mlflow.set_tags({
        "dataset":"v2025-10-01",
        "data_source":"feature_store://core/v1",
        "git_commit":"<shortsha>",
        "stage":"exp"
    })
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 12)

    # Log metrics over time/steps
    for step, val in enumerate([0.69, 0.71, 0.73]):
        mlflow.log_metric("val_auc", val, step=step)

    # Artifacts: figures, tables, configs
    import json, pathlib
    pathlib.Path("cfg.json").write_text(json.dumps({"seed": 123, "folds": 5}))
    mlflow.log_artifact("cfg.json")
    print("Run URL should now show this stuff:", run.info.run_id)
```

### Nested runs (optional, for pipelines)

```python
with mlflow.start_run(run_name="train-pipeline") as parent:
    with mlflow.start_run(run_name="feature-build", nested=True):
        mlflow.log_metric("rows", 1_234_567)
    with mlflow.start_run(run_name="model-train", nested=True):
        mlflow.log_metric("train_auc", 0.812)
```

---

## 4) Log models (and optionally register them)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import mlflow.sklearn, mlflow

X, y = make_classification(n_samples=5000, n_features=20, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

with mlflow.start_run(run_name="rf-v1"):
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    mlflow.log_metric("acc", acc)

    # Log the model under artifacts "model"
    mlflow.sklearn.log_model(clf, artifact_path="model")

    # If your org uses the Model Registry, set a name to register:
    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="risk_approval_model"
    )
    print("Registered model version:", result.version)
```

---

## 5) Autologging (fastest way to capture runs)

```python
import mlflow
mlflow.set_experiment("team-risk/credit-approval/phase-exp")

# Pick your framework:
mlflow.sklearn.autolog()          # scikit-learn
# mlflow.xgboost.autolog()        # XGBoost
# mlflow.lightgbm.autolog()       # LightGBM
# mlflow.pytorch.autolog()        # PyTorch
# mlflow.tensorflow.autolog()     # TensorFlow/Keras
# mlflow.catboost.autolog()       # CatBoost

with mlflow.start_run(run_name="auto-sklearn"):
    # Train as usual; params, metrics, model, and artifacts are captured
    ...
```

Autologging captures:

* Params/metrics the library exposes
* The model artifact
* Sometimes plots (e.g., feature importances, confusion matrix) depending on framework

---

## 6) Reading results back (querying runs)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = client.get_experiment_by_name("team-risk/credit-approval/phase-exp")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.acc > 0.80 and tags.stage = 'exp'",
    order_by=["metrics.acc DESC"],
    max_results=10
)
for r in runs:
    print(r.info.run_id, r.data.metrics.get("acc"), r.data.params)
```

**Download artifacts**

```python
import mlflow
dst = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{runs[0].info.run_id}/model"
)
print("Downloaded to:", dst)
```

---

## 7) Connectivity checks (quick debugging)

**Python check**

```python
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
print("URI:", mlflow.get_tracking_uri())
mlflow.set_experiment("connectivity-smoketest")
with mlflow.start_run():
    mlflow.log_metric("ping_metric", 1)
print("OK if no exception was raised.")
```

**REST check (Terminal inside Jupyter)**

```bash
curl -sS -H "Authorization: Bearer $MLFLOW_TRACKING_TOKEN" \
  "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/list" | head
```

If you get JSON (not HTML or an error), you’re talking to the tracking server.

---

## 8) Tags you’ll thank yourself for later

Use `mlflow.set_tags({...})` to record searchable lineage:

* `git_commit`, `git_branch`, `data_snapshot`, `feature_view`, `training_date`
* `owner`, `team`, `jira`, `stage` (`exp`, `canary`, `prod`)
* `run_type` (`baseline`, `ablation`, `sweep`, `retrain`)
* `input_signature` (hash of data schema/features)

These make filtering/searching in the UI and with `search_runs` much easier.

---

## 9) Common DS-side issues & quick fixes

| Symptom                                     | What it usually means                  | Fix fast                                                                                |
| ------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------- |
| **Run/experiment doesn’t appear**           | Wrong `MLFLOW_TRACKING_URI` or auth    | Print `mlflow.get_tracking_uri()`. Re-set `MLFLOW_TRACKING_TOKEN` or username/password. |
| **401/403**                                 | Missing/invalid token/creds            | Export correct token or basic auth envs; re-start kernel.                               |
| **SSL cert error**                          | Org CA not trusted                     | Set `REQUESTS_CA_BUNDLE=/path/to/company-ca.pem`. Don’t disable verify in prod.         |
| **Artifacts won’t download**                | Permissions or artifact backend policy | You can still view params/metrics; ask platform about artifact ACL or use the registry. |
| **Client/server version mismatch warnings** | Older client in the notebook image     | `pip install -U mlflow` (pin to org-approved version if required).                      |
| **Autologging logs nothing**                | Autolog not enabled for that framework | Call the correct `mlflow.<lib>.autolog()` *before* training.                            |

---

## 10) Handy patterns for real projects

**Capture the data schema & feature list**

```python
import mlflow, json
schema = {"features": ["age","income","tenure"], "target": "approve", "ver":"2025-10-01"}
mlflow.log_dict(schema, "data/schema.json")
```

**Record CV results compactly**

```python
for fold, auc in enumerate([0.79, 0.81, 0.80, 0.82, 0.81]):
    mlflow.log_metric("cv_auc", auc, step=fold)
mlflow.log_metric("cv_auc_mean", sum([0.79,0.81,0.80,0.82,0.81])/5)
```

**Attach the exact training command**

```python
import sys, shlex
mlflow.set_tag("cmd", " ".join(map(shlex.quote, sys.argv)) if hasattr(sys, "argv") else "notebook")
```

---

### That’s it

With the env var for the **tracking URI**, a consistent **experiment name**, and the **`start_run` → log → (optional) register** pattern, you’re fully covered for day-to-day model experimentation and lineage from JupyterHub.
