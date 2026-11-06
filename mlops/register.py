"""Register the possession pipeline of one commit as a new MLflow model version.

    python mlops/register.py --commit <sha> [--branch main] [--message "<commit message>"]
                             [--baseline-commit <sha> --baseline-tree <dir>] [--store <dir>]

.githooks/pre-push calls this for every branch pushed, from a clean export of the pushed commit:

  main      always registered; the alias `main` moves to the new version
  a branch  registered only if it beats main (--baseline-*: the remote's main, exported) on the
            public window: neither spellF1 nor macroF1 lower, and at least one higher

A version holds:
  metrics      public_spell_f1, public_macro_f1: the public window, as the scorer computes them
  description  the commit message, and the scores next to main's
  params       every constant of the pipeline modules (tracking, deadball, possession, broadcast, run)
  tags         git_commit, branch, submission_sha256 (same hash = same submission file)
  artifacts    the submission CSV, solution/README.md, and the model: mlops/model.py with
               solution/, starter/ and games.py

The store (sqlite + artifacts) defaults to .mlflow/ in the repository, git-ignored. A commit that
already has a run is skipped.
"""

import argparse
import hashlib
import os
import pathlib
import subprocess
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
import games  # noqa: E402
from solution import broadcast, deadball, diagnostics, possession, run, tracking  # noqa: E402
from baseline import write_submission  # noqa: E402  (starter/, put on sys.path by solution.tracking)

# Params
MODEL_NAME = "possession"
EXPERIMENT = "possession"
ALIAS = "main"
METRICS = ("public_spell_f1", "public_macro_f1")
MODEL_DESCRIPTION = (
    "Ball possession for every video frame of the match: H (home), A (away) or D (dead ball), "
    "from both cameras' tracking data, with no video and no ground truth as input. The method is "
    "in solution/README.md. @main is the newest version pushed to main; any other version is a "
    "branch that beat main on the public window when it was pushed.")
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


def public_scores(labels):
    """spellF1 and macroF1 on the public window (labels: an array, or a submission DataFrame)."""
    scores = diagnostics.public_scores(labels)
    return {"public_spell_f1": scores["spell_f1_mean"], "public_macro_f1": scores["macro_f1"]}


def baseline_scores(exp_id, commit, tree):
    """main's scores: from its run in the store if it has one, else by running main's own code."""
    found = mlflow.search_runs([exp_id], "tags.git_commit = '{}'".format(commit))
    if len(found):
        return {key: float(found.iloc[0]["metrics." + key]) for key in METRICS}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(tree, "solution", "run.py"),
                        "--out", tmp, "--name", "main"], check=True, stdout=subprocess.DEVNULL,
                       env=dict(os.environ, MTAH_DATA_DIR=games.DATA_DIR))
        return public_scores(diagnostics.pm.read_label_csv(os.path.join(tmp, "main.csv")))


def beats(scores, base):
    """Neither metric lower, at least one higher."""
    return (all(scores[k] >= base[k] for k in METRICS)
            and any(scores[k] > base[k] for k in METRICS))


def describe(scores, base, args):
    line = "public spellF1 {:.3f}, macroF1 {:.3f}".format(*(scores[k] for k in METRICS))
    if base is not None:
        line += " (main {}: {:.3f}, {:.3f})".format(args.baseline_commit[:7],
                                                    *(base[k] for k in METRICS))
    return "{}\n\n{}\nbranch {}, commit {}".format(args.message.strip(), line, args.branch,
                                                  args.commit[:7]).strip()


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
    parser.add_argument("--branch", default="main", help="branch it was pushed to")
    parser.add_argument("--message", default="", help="its commit message")
    parser.add_argument("--baseline-commit", help="main's commit, to beat (branches only)")
    parser.add_argument("--baseline-tree", help="an export of that commit")
    parser.add_argument("--store", default=os.path.join(ROOT, ".mlflow"),
                        help="folder holding mlflow.db and the artifacts")
    args = parser.parse_args()
    if args.branch != ALIAS and not (args.baseline_commit and args.baseline_tree):
        parser.error("a branch other than main needs --baseline-commit and --baseline-tree")

    os.makedirs(args.store, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.path.abspath(args.store), "mlflow.db"))
    exp_id = experiment_id(args.store)
    if len(mlflow.search_runs([exp_id], "tags.git_commit = '{}'".format(args.commit))):
        print("mlflow: {} already registered, skipping".format(args.commit[:7]))
        return

    match = tracking.load()
    labels, _, _ = run.build(match, broadcast.contact_evidence(match))
    scores = public_scores(labels)
    base = None
    if args.branch != ALIAS:
        base = baseline_scores(exp_id, args.baseline_commit, args.baseline_tree)
        if not beats(scores, base):
            print("mlflow: {} not registered, does not beat main: {}".format(
                args.branch, describe(scores, base, args).splitlines()[-2]))
            return
    description = describe(scores, base, args)

    run_name = "{} {}".format(args.commit[:7], args.message.strip().split("\n")[0]).strip()
    with tempfile.TemporaryDirectory() as tmp, \
            mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description):
        csv = write_submission(os.path.join(tmp, "solution.csv"), labels)
        mlflow.set_tags({"git_commit": args.commit, "branch": args.branch,
                         "submission_sha256": sha256(csv)})
        mlflow.log_params(pipeline_params())
        mlflow.log_metrics(scores)
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
    client.update_registered_model(MODEL_NAME, description=MODEL_DESCRIPTION)
    client.update_model_version(MODEL_NAME, version, description=description)
    client.set_model_version_tag(MODEL_NAME, version, "git_commit", args.commit)
    client.set_model_version_tag(MODEL_NAME, version, "branch", args.branch)
    if args.branch == ALIAS:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print("mlflow: {} -> {} version {}{}  {}".format(
        args.branch, MODEL_NAME, version, " (@main)" if args.branch == ALIAS else "",
        description.splitlines()[-2]))


if __name__ == '__main__':
    main()
