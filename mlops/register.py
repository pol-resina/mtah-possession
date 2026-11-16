"""Register solution/ (and any solutions/<name>/) as versions of the MLflow model `possession`.

    python mlops/register.py --commit <sha> [--branch main] [--message "<commit message>"]
                             [--baseline-commit <sha> --baseline-tree <dir>] [--store <dir>]

.githooks/pre-push calls this for every branch pushed, from a clean export of the pushed commit.
solution/, and each solutions/<name>/, with a run.py is one solution. run.py must accept
--out <dir> and --name <file> and write the submission to <dir>/<file>.csv. A solution is identified by a hash of
its files, so one that is already registered is skipped:

  main      every new or changed solution is registered
  a branch  a solution that differs from main's copy is registered only if it beats the best
            solution on main (--baseline-*: the remote's main, exported) on the public window:
            neither spellF1 nor macroF1 lower, and at least one higher

A version holds:
  metrics      public_spell_f1, public_macro_f1
  description  the commit message, the scores, main's best (for a branch), branch and commit
  params       every upper-case literal constant in the solution's .py files
  tags         solution, solution_hash, git_commit, branch, submission_sha256
  artifacts    the submission CSV, the solution's README.md, and the model: mlops/model.py, which
               runs the solution's run.py, shipped with the solution, starter/ and games.py
  aliases      @<solution>: that solution's newest version; @best: the best version registered

The store (sqlite + artifacts) defaults to .mlflow/ in the repository, git-ignored.
"""

import argparse
import ast
import hashlib
import os
import pathlib
import re
import shutil
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
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
MODEL_NAME = "possession"
EXPERIMENT = "possession"
BEST = "best"
METRICS = ("public_spell_f1", "public_macro_f1")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
MODEL_DESCRIPTION = (
    "Ball possession for every video frame of the match: H (home), A (away) or D (dead ball). "
    "Each version is one solution (solution/ or solutions/<name>/) at one commit (tags `solution`, `git_commit`); "
    "its method is in that solution's README.md. @<solution> is a solution's newest version, "
    "@best the best version registered. Versions from branches other than main were registered "
    "because they beat main's best solution on the public window when pushed.")
SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "data_dir", required=False)]),
    outputs=Schema([ColSpec("long", "frame"), ColSpec("string", "label")]))
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc")


def solutions(root):
    """{name: folder} of every solution: root/solution/ (named "solution") and root/solutions/*/."""
    found = {}
    if os.path.isfile(os.path.join(root, "solution", "run.py")):
        found["solution"] = os.path.join(root, "solution")
    base = os.path.join(root, "solutions")
    if os.path.isdir(base):
        found.update({name: os.path.join(base, name) for name in sorted(os.listdir(base))
                      if os.path.isfile(os.path.join(base, name, "run.py"))})
    return found


def layout_path(folder, name):
    """Where a solution sits in the repository: solutions/<name>/ or solution/."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(folder)))
    return os.path.join("solutions", name) if parent == "solutions" else "solution"


def folder_hash(folder):
    """sha256 of a folder's files (paths and bytes), ignoring Python caches."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(f for f in filenames if not f.endswith(".pyc")):
            path = os.path.join(dirpath, name)
            digest.update(os.path.relpath(path, folder).encode() + b"\0")
            with open(path, "rb") as f:
                digest.update(f.read() + b"\0")
    return digest.hexdigest()


def run_solution(folder, out):
    """Run a solution's run.py; the path of the submission it wrote."""
    subprocess.run([sys.executable, os.path.join(folder, "run.py"), "--out", out,
                    "--name", "submission"], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, MTAH_DATA_DIR=games.DATA_DIR))
    return os.path.join(out, "submission.csv")


def public_scores(csv):
    """spellF1 and macroF1 of a submission on the public window, as the scorer computes them."""
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    scores = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(csv)), bootstrap_n=0)
    return {"public_spell_f1": scores["spell_f1_mean"], "public_macro_f1": scores["macro_f1"]}


def registered_scores(exp_id, digest):
    """The scores of the solution with this hash if it is registered, else None."""
    found = mlflow.search_runs([exp_id], "tags.solution_hash = '{}'".format(digest))
    if not len(found):
        return None
    return {key: float(found.iloc[0]["metrics." + key]) for key in METRICS}


def rank(scores):
    return scores["public_spell_f1"], scores["public_macro_f1"]


def beats(scores, base):
    """Neither metric lower, at least one higher."""
    return (all(scores[k] >= base[k] for k in METRICS)
            and any(scores[k] > base[k] for k in METRICS))


def best_on_main(exp_id, tree):
    """(name, scores) of the best solution in an export of main, or (None, None) if it has none.

    Scores come from the store when that exact solution is registered, else from running it.
    """
    ranked = []
    for name, folder in solutions(tree).items():
        scores = registered_scores(exp_id, folder_hash(folder))
        if scores is None:
            with tempfile.TemporaryDirectory() as tmp:
                scores = public_scores(run_solution(folder, tmp))
        ranked.append((name, scores))
    return max(ranked, key=lambda item: rank(item[1])) if ranked else (None, None)


def solution_params(folder):
    """Every upper-case constant with a literal value in the solution's top-level .py files."""
    params = {}
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".py"):
            continue
        with open(os.path.join(folder, filename)) as f:
            body = ast.parse(f.read()).body
        for node in body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper()):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                params["{}.{}".format(filename[:-3], node.targets[0].id)] = value
    return params


def score_line(name, scores, main_name=None, main_scores=None):
    line = "{}: public spellF1 {:.3f}, macroF1 {:.3f}".format(name, *(scores[k] for k in METRICS))
    if main_scores is not None:
        line += " (main's best, {}: {:.3f}, {:.3f})".format(main_name,
                                                             *(main_scores[k] for k in METRICS))
    return line


def stage_code(folder, name, out):
    """A folder laid out like the repository, holding only what the solution's run.py needs."""
    shutil.copytree(folder, os.path.join(out, layout_path(folder, name)), ignore=SKIP)
    shutil.copytree(os.path.join(ROOT, "starter"), os.path.join(out, "starter"), ignore=SKIP)
    shutil.copy(os.path.join(ROOT, "games.py"), out)
    return out


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def register(exp_id, name, folder, digest, args, main_name, main_scores):
    """Run, score and (if it qualifies) register one solution. The new version, or None."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = run_solution(folder, tmp)
        scores = public_scores(csv)
        line = score_line(name, scores, main_name, main_scores)
        if main_scores is not None and not beats(scores, main_scores):
            print("mlflow: not registered, does not beat main's best: " + line)
            return None
        description = "{}\n\n{}\nbranch {}, commit {}".format(
            args.message.strip(), line, args.branch, args.commit[:7]).strip()
        run_name = "{} {}".format(name, args.commit[:7])
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description):
            mlflow.set_tags({"solution": name, "solution_hash": digest, "git_commit": args.commit,
                             "branch": args.branch, "submission_sha256": sha256(csv)})
            mlflow.log_params(solution_params(folder))
            mlflow.log_metrics(scores)
            mlflow.log_artifact(csv, "submission")
            if os.path.isfile(os.path.join(folder, "README.md")):
                mlflow.log_artifact(os.path.join(folder, "README.md"), "docs")
            with warnings.catch_warnings():
                # No input example: MLflow would run the whole solution on it. The signature is
                # explicit.
                warnings.filterwarnings("ignore", message=".*An input example was not provided")
                info = mlflow.pyfunc.log_model(
                    name="model", python_model=os.path.join(HERE, "model.py"),
                    artifacts={"code": stage_code(folder, name, os.path.join(tmp, "code"))},
                    model_config={"solution": name, "path": layout_path(folder, name)},
                    signature=SIGNATURE,
                    registered_model_name=MODEL_NAME,
                    pip_requirements=["numpy=={}".format(np.__version__),
                                      "pandas=={}".format(pd.__version__)])
    version = info.registered_model_version
    client = mlflow.MlflowClient()
    client.update_model_version(MODEL_NAME, version, description=description)
    for key, value in (("solution", name), ("git_commit", args.commit), ("branch", args.branch)):
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    client.set_registered_model_alias(MODEL_NAME, re.sub(r"[^\w-]", "-", name), version)
    print("mlflow: {} -> {} version {}  {}".format(args.branch, MODEL_NAME, version, line))
    return version


def point_best(exp_id):
    """Move @best to the registered version with the best scores."""
    found = mlflow.search_runs([exp_id])
    if "tags.solution_hash" not in found:
        return
    found = found[found["tags.solution_hash"].notna()]
    if not len(found):
        return
    oldest_first = sorted(found.to_dict("records"), key=lambda r: r["start_time"])
    best = max(oldest_first, key=lambda r: rank({k: r["metrics." + k] for k in METRICS}))  # ties: oldest
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("run_id = '{}'".format(best["run_id"]))
    if versions:
        client.set_registered_model_alias(MODEL_NAME, BEST, versions[0].version)


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
    on_main = args.branch == "main"
    if not on_main and not (args.baseline_commit and args.baseline_tree):
        parser.error("a branch other than main needs --baseline-commit and --baseline-tree")

    os.makedirs(args.store, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.path.abspath(args.store), "mlflow.db"))
    exp_id = experiment_id(args.store)

    main_name, main_scores, main_hashes = None, None, {}
    if not on_main:
        main_name, main_scores = best_on_main(exp_id, args.baseline_tree)
        main_hashes = {name: folder_hash(folder)
                       for name, folder in solutions(args.baseline_tree).items()}

    registered = 0
    for name, folder in solutions(ROOT).items():
        digest = folder_hash(folder)
        if main_hashes.get(name) == digest:
            continue  # unchanged from main: nothing to propose
        if registered_scores(exp_id, digest) is not None:
            print("mlflow: {} already registered, skipping".format(name))
            continue
        registered += register(exp_id, name, folder, digest, args, main_name, main_scores) is not None
    if registered:
        mlflow.MlflowClient().update_registered_model(MODEL_NAME, description=MODEL_DESCRIPTION)
        point_best(exp_id)


if __name__ == '__main__':
    main()
