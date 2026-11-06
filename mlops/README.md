# Model registry (MLflow)

Every solution lives in its own folder, `solutions/solution_<who>_<commit>/`, with a `run.py`
that accepts `--out <dir> --name <file>` and writes the submission to `<dir>/<file>.csv`
(see `solutions/solution_pol_5e9d334/` for one). Pushed solutions are registered as versions of
the MLflow model `possession`:

- **on `main`**, every new or changed solution is registered;
- **on any other branch**, a solution that differs from main's copy is registered only if it beats
  the best solution on `main` on the public window: neither spellF1 nor macroF1 lower, and at
  least one higher.

So the workflow is: open a branch, add your folder under `solutions/`, commit, `git push -u
origin <branch>`. The hook prints, for each solution, whether it was registered and its scores
next to main's best.

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
runs `mlops/register.py` on it. A solution is identified by a hash of its files, so one already
registered is skipped. Each new one is run (about 10 s) and scored. For a branch other than
`main`, the hook first fetches the remote's `main`; its solutions' scores come from the store when
registered, else from running them.

A registered version holds:

| | |
|---|---|
| metrics | `public_spell_f1`, `public_macro_f1` |
| description | the commit message, the scores, main's best (for a branch), branch and commit |
| params | every upper-case literal constant in the solution's `.py` files |
| tags | `solution`, `solution_hash`, `git_commit`, `branch`, `submission_sha256` (same hash = same submission file) |
| artifacts | the submission CSV, the solution's `README.md`, and the model: `mlops/model.py` running the solution's `run.py`, shipped with `starter/` and `games.py` |
| aliases | `@<solution>`: that solution's newest version. `@best`: the best version registered |

A failed registration prints a warning and never blocks the push. The hook runs before the push,
so a push the remote rejects still registers.

The store is `.mlflow/` (git-ignored, local to each clone: it holds the frame-level submissions,
which must not be published).

## Using it

```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db       # http://127.0.0.1:5000
```

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///.mlflow/mlflow.db")
model = mlflow.pyfunc.load_model("models:/possession@best")    # or @solution_pol_5e9d334
labels = model.predict(pd.DataFrame({"data_dir": ["/path/to/data"]}))
```

Register by hand: `python mlops/register.py --commit $(git rev-parse HEAD)` for main; a branch
also needs `--branch <name> --baseline-commit <main sha> --baseline-tree <export of main>`.
