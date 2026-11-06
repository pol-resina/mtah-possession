"""Log the possession pipeline of one commit to MLflow and register it as a new model version.

    python mlops/register.py --commit <sha> [--message "<subject>"] [--store <dir>]

.githooks/pre-push calls this for every commit pushed to main, from a clean export of that commit,
so what gets registered is exactly what was pushed. One MLflow run per commit:

  params     every constant of the pipeline modules (tracking, deadball, possession, broadcast, run)
  metrics    public_*: the public window, spellF1 with its bootstrap band, and macroF1
             full_*:   the whole match, no ground truth (solution/diagnostics.py full_match_checks)
  artifacts  the submission CSV (its sha256 is the tag submission_sha256) and solution/README.md
  model      mlops/model.py with solution/, starter/ and games.py, registered as `possession`;
             the alias `main` points at the newest version pushed to main

The store (sqlite + artifacts) defaults to .mlflow/ in the repository, git-ignored. A commit that
already has a run is skipped, so pushing it again registers nothing.
"""

import argparse
import hashlib
import os
import pathlib
import sys
import tempfile
import warnings

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import broadcast, deadball, diagnostics, possession, run, tracking  # noqa: E402
import load_smart_tagging  # noqa: E402  (starter/, put on sys.path by solution.tracking)
from baseline import write_submission  # noqa: E402

# Params
MODEL_NAME = "possession"
EXPERIMENT = "possession"
ALIAS = "main"
MODULES = (tracking, deadball, possession, broadcast, run)
CODE_PATHS = [os.path.join(ROOT, p) for p in ("solution", "starter", "games.py")]
SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "game"), ColSpec("string", "data_dir", required=False)]),
    outputs=Schema([ColSpec("string", "game"), ColSpec("long", "frame"),
                    ColSpec("string", "label")]))


def pipeline_params():
    """Every upper-case constant of the pipeline modules, except file paths."""
    params = {}
    for module in MODULES:
        for name, value in vars(module).items():
            if name.isupper() and not (isinstance(value, str) and os.sep in value):
                params["{}.{}".format(module.__name__.split(".")[-1], name)] = value
    return params


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def experiment_id(store):
    """The experiment's id, creating it (with its artifacts under the store) the first time."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    artifacts = pathlib.Path(store, "artifacts").resolve().as_uri()
    return mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True, help="git commit the code was exported from")
    parser.add_argument("--message", default="", help="its subject line, used as the run name")
    parser.add_argument("--store", default=os.path.join(ROOT, ".mlflow"),
                        help="folder holding mlflow.db and the artifacts")
    args = parser.parse_args()

    os.makedirs(args.store, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.path.abspath(args.store), "mlflow.db"))
    exp_id = experiment_id(args.store)
    if len(mlflow.search_runs([exp_id], "tags.git_commit = '{}'".format(args.commit))):
        print("mlflow: {} already registered, skipping".format(args.commit[:7]))
        return

    match = tracking.load()
    events = load_smart_tagging.load()
    labels, _, stoppages = run.build(match, broadcast.contact_evidence(match))
    public = diagnostics.public_scores(labels, diagnostics.BOOTSTRAP_N)
    full = diagnostics.full_match_checks(match, labels, stoppages, events)

    run_name = "{} {}".format(args.commit[:7], args.message).strip()
    with tempfile.TemporaryDirectory() as tmp, \
            mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        csv = write_submission(os.path.join(tmp, "solution.csv"), labels)
        mlflow.set_tags({"git_commit": args.commit, "git_message": args.message,
                         "submission_sha256": sha256(csv)})
        mlflow.log_params(pipeline_params())
        mlflow.log_metrics({
            "public_spell_f1": public["spell_f1_mean"],
            "public_spell_f1_low": public["spell_bootstrap_low"],
            "public_spell_f1_high": public["spell_bootstrap_high"],
            "public_macro_f1": public["macro_f1"],
            **{"full_" + key: value for key, value in full.items()},
        })
        mlflow.log_artifact(csv, "submission")
        mlflow.log_artifact(os.path.join(ROOT, "solution", "README.md"), "docs")
        with warnings.catch_warnings():
            # No input example: MLflow would run the whole pipeline on it. The signature is explicit.
            warnings.filterwarnings("ignore", message=".*An input example was not provided")
            info = mlflow.pyfunc.log_model(
                name="model", python_model=os.path.join(HERE, "model.py"), code_paths=CODE_PATHS,
                signature=SIGNATURE, registered_model_name=MODEL_NAME,
                pip_requirements=["numpy=={}".format(np.__version__),
                                  "pandas=={}".format(pd.__version__)])

    version = info.registered_model_version
    client = mlflow.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "git_commit", args.commit)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print("mlflow: {} -> {} version {} (@{})  public spellF1 {:.3f}  macroF1 {:.3f}".format(
        args.commit[:7], MODEL_NAME, version, ALIAS, public["spell_f1_mean"], public["macro_f1"]))


if __name__ == '__main__':
    main()
