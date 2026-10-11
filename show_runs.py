import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tracking_dir = ROOT / "mlruns"
mlflow.set_tracking_uri(tracking_dir.as_uri())

runs = mlflow.search_runs(experiment_names=["bearing-fault-diagnosis"])
print(runs[["run_id", "metrics.best_val_acc", "metrics.val_acc", "params.lr", "params.epochs"]])