"""One solution (solution/, or solutions/<name>/) as an MLflow pyfunc model, logged from code by register.py.

The artifact `code` holds the solution in a copy of the repository's layout (solution/ or solutions/<name>/,
starter/, games.py); model_config["solution"] names it. predict runs its run.py.

Input: one row per run, column `data_dir` (the unpacked match assets; empty means games.py's
default: MTAH_DATA_DIR, else ./data). Output: the submission, one row per tactical video frame:
frame, label (H / A / D, X ungraded).
"""

import os
import subprocess
import sys
import tempfile

import mlflow
import pandas as pd
from mlflow.models import set_model


class Possession(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        path = (context.model_config.get("path")
                or os.path.join("solutions", context.model_config["solution"]))
        run_py = os.path.join(context.artifacts["code"], path, "run.py")
        out = []
        for row in model_input.to_dict("records"):
            env = dict(os.environ)
            if row.get("data_dir"):
                env["MTAH_DATA_DIR"] = row["data_dir"]
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([sys.executable, run_py, "--out", tmp, "--name", "labels"],
                               check=True, stdout=subprocess.DEVNULL, env=env)
                out.append(pd.read_csv(os.path.join(tmp, "labels.csv")))
        return pd.concat(out, ignore_index=True)


set_model(Possession())
