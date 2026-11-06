# Model registry (MLflow)

Pushed commits are registered as versions of the MLflow model `possession`:

- **a push to `main`** is always registered, and the alias `@main` moves to it;
- **a push to any other branch** is registered only if it beats `main` on the public window:
  neither spellF1 nor macroF1 lower, and at least one higher.

So the workflow for a change is: open a branch, change `solution/`, commit, `git push -u origin
<branch>`. The hook says whether it was registered, with both scores next to main's.

The match data can't go on GitHub, so this runs locally, in a git `pre-push` hook, not in CI.

## Setup (once per clone)

```bash
pip install -r mlops/requirements.txt
git config core.hooksPath .githooks
```

The hook uses `~/.venvs/mtah-possession/bin/python` if it exists, else `python3`; set `MTAH_PYTHON`
to use another. It reads the data from `MTAH_DATA_DIR`, else `./data`.

## What a push does

For each branch pushed, `.githooks/pre-push` exports the pushed commit to a temporary folder and
runs `mlops/register.py` on it (about 10 s). For a branch other than `main`, it first fetches the
remote's `main` and gets its scores: from the store if that commit is registered, else by running
`main`'s code (another 10 s).

A registered version holds:

| | |
|---|---|
| metrics | `public_spell_f1`, `public_macro_f1` |
| description | the commit message, the scores, main's scores (for a branch), branch and commit |
| params | every constant in `tracking`, `deadball`, `possession`, `broadcast`, `run` |
| tags | `git_commit`, `branch`, `submission_sha256` (same hash = same submission file) |
| artifacts | the submission CSV, `solution/README.md`, and the model: `mlops/model.py` + `solution/`, `starter/`, `games.py` |

A commit that already has a run is skipped. A failed registration prints a warning and never
blocks the push. The hook runs before the push, so a push the remote rejects still registers.

The store is `.mlflow/` (git-ignored, local to each clone: it holds the frame-level submission,
which must not be published).

## Using it

```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db       # http://127.0.0.1:5000
```

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///.mlflow/mlflow.db")
model = mlflow.pyfunc.load_model("models:/possession@main")
labels = model.predict(pd.DataFrame({"game": ["BAR-ATH"], "data_dir": ["/path/to/data"]}))
```

Register by hand: `python mlops/register.py --commit $(git rev-parse HEAD)` for main; a branch
also needs `--branch <name> --baseline-commit <main sha> --baseline-tree <export of main>`.
