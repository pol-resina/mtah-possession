# Model registry (MLflow)

Every commit pushed to `main` is registered as a new version of the MLflow model `possession`.
The match data can't go on GitHub, so this runs locally, in a git `pre-push` hook, not in CI.

## Setup (once per clone)

```bash
pip install -r mlops/requirements.txt
git config core.hooksPath .githooks
```

The hook uses `~/.venvs/mtah-possession/bin/python` if it exists, else `python3`; set `MTAH_PYTHON`
to use another. It reads the data from `MTAH_DATA_DIR`, else `./data`.

## What a push does

For each commit pushed to `main`, `.githooks/pre-push` exports that commit to a temporary folder
and runs `mlops/register.py` on it (about 15 s). That logs one run and registers one version:

| | |
|---|---|
| params | every constant in `tracking`, `deadball`, `possession`, `broadcast`, `run` |
| metrics | `public_*`: public window spellF1 (with bootstrap band) and macroF1. `full_*`: the whole-match checks from `solution/diagnostics.py`, no ground truth |
| tags | `git_commit`, `git_message`, `submission_sha256` (same hash = same submission file) |
| artifacts | the submission CSV, `solution/README.md` |
| model | `mlops/model.py` + `solution/`, `starter/`, `games.py`. Alias `@main` = newest version |

A commit that already has a run is skipped. A failed registration prints a warning and never
blocks the push. The hook runs before the push, so a push the remote rejects still registers.

The store is `.mlflow/` (git-ignored, local to each clone: it holds the frame-level submission,
which must not be published).

## Using it

```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db       # browse runs and versions
```

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///.mlflow/mlflow.db")
model = mlflow.pyfunc.load_model("models:/possession@main")
labels = model.predict(pd.DataFrame({"game": ["BAR-ATH"], "data_dir": ["/path/to/data"]}))
```

Register by hand (e.g. the current commit): `python mlops/register.py --commit $(git rev-parse HEAD)`.
