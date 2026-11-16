# Training the team model

This folder holds the code that trained the shipped team model, `../team_model.txt` and
`../team_model_features.txt`, the model behind our submission. Adrià wrote it, as `bench/` in his
own workspace. From the raw open data it reproduces both files **byte for byte**.

It trains only on open data from **other games**. It never reads the provided match's labels,
which it has none of, or its public ground truth.

## Data

Seven matches from the German Bundesliga and 2. Bundesliga (Bassek, Rein, Weber, Memmert 2025, *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data; figshare
article 28196177; CC BY 4.0). Each frame comes labelled with which team has the ball
(`BallPossession`) and whether the ball is in play (`BallStatus`). These labels are part of the
official DFL position data produced by Sportec Solutions; we labelled nothing. None of the data is
in this repository.

## Reproduce the shipped model

From the repository root, after `pip install -r solution/requirements.txt`:

```bash
python solution/training/download.py              # 21 files, 2.6 GB -> cache/dfl/raw/
python solution/training/bench/export_team.py     # -> cache/team_model/team_model.txt (+ _features.txt)
cmp cache/team_model/team_model.txt solution/team_model.txt && echo identical
cmp cache/team_model/team_model_features.txt solution/team_model_features.txt && echo identical
```

A from-scratch run (parse the XML, degrade, build features, train) takes about 10 minutes.
Parsed matches and feature tables are cached under `cache/team_model/` (git-ignored), so a second
run is quicker. The script never writes into `solution/`.

## The recipe

| file | what it does |
|---|---|
| `bench/dfl.py` | parses the DFL XML into tracking arrays and per-frame labels: `BallStatus` 0 → D, else `BallPossession` → H / A |
| `bench/degrade.py` | makes each clean DFL match look like our automatic tracking. The ball is lost for long stretches, mostly in stoppages. Lofted balls are projected onto the grass from the camera. Players drop out and their tracks break into anonymous slots. A slot sometimes carries the wrong team, and goalkeepers swap tracks. Its settings are in `bench/degrade_knobs.json` (id `b8d8d8f61e`), fitted to label-free statistics of our own tracking, never to any ground truth |
| `bench/stats.py`, `bench/obs.py` | the statistics the settings are fitted to, and the shared tracking format |
| `bench/features.py` | the 530 columns at 5 Hz. `../features.py` builds the same columns for the provided match |
| `bench/gbdt.py` | the training set: every match degraded with seeds 100 and 101, each copy also home/away-swapped and mirrored, rows while the ball is alive. LightGBM binary, 400 trees, learning rate 0.05, 63 leaves, at least 200 rows per leaf, feature fraction 0.5, bagging 0.8, L2 1.0, 8 threads, deterministic |
| `bench/harness.py` | loads and degrades the bench matches, and scores held-out matches with the organisers' `evaluation/possession_metrics.py` |
| `bench/export_team.py` | trains on all seven matches and writes the model |
| `download.py` | downloads the open data |

**Adaptations from Adrià's workspace.** Only file paths were changed: the data comes from
`cache/dfl/raw/`, caches and output go to `cache/team_model/`, and the degradation settings sit next
to the code. `bench/models.py` is a stand-in: Adrià's own file holds his experiment models, and
`features.py` only needs its `_pol_paths()`, which finds the solution's `tracking.py`.
`team_per_frame` there is unchanged since training. Parts of `gbdt.py`, `harness.py` and `obs.py`
also call into his evaluation experiments (`bench.models.REGISTRY`, `bench.hsmm`,
`pipeline.atd`). Those files are not included, and training does not use them.

## Evidence

Adrià's leave-one-match-out bench trains on six degraded matches and tests the whole pipeline on
the seventh. On it the team model reaches spellF1 0.434 and macroF1 0.800, against 0.289 and 0.738
for the rule-based team step, and it is better on all 7 matches (sign test p = 0.008). The 1.0-nat
switch cost of the decoder was the best value in every held-out fold.
