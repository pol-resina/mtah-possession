"""The possession pipeline as an MLflow pyfunc model, logged from code by register.py.

Input: one row per game, columns `game` (e.g. "BAR-ATH") and, optionally, `data_dir` (the unpacked
match assets; empty means games.py's default: MTAH_DATA_DIR, else ./data).
Output: one row per tactical video frame of each game: game, frame, label (H / A / D, X ungraded).

The model's code/ folder carries solution/, starter/ and games.py, so `solution` imports as is.
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import set_model


class Possession(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from solution import broadcast, run, tracking

        out = []
        for row in model_input.to_dict("records"):
            data_dir = row.get("data_dir") or None
            match = tracking.load(row["game"], data_dir)
            second = broadcast.contact_evidence(match, row["game"], data_dir)
            labels, _, _ = run.build(match, second)
            out.append(pd.DataFrame({"game": row["game"], "frame": np.arange(len(labels)),
                                     "label": labels}))
        return pd.concat(out, ignore_index=True)


set_model(Possession())
