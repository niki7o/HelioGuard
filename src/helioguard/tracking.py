"""
MLflow tracking helpers.

Day 1: skeleton only. The real experiments start in notebook 03 (baselines).
Putting the boilerplate here means every notebook stays focused on
modelling, not on tracking ceremony.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from helioguard.config import MLRUNS_DIR


def setup_mlflow(experiment_name: str = "helioguard") -> str:
    """Point MLflow at a project-local SQLite store and select an experiment.

    MLflow 3.x deprecated the plain-file backend; SQLite gives us a
    single ``mlruns/mlflow.db`` file that DVC can track if needed and
    works with the standard ``mlflow ui`` command. Returns the
    experiment id."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = Path(MLRUNS_DIR).resolve() / "mlflow.db"
    artifact_root = Path(MLRUNS_DIR).resolve() / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{db_path.as_posix()}")
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.as_uri()
        )
        mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id if exp else ""
