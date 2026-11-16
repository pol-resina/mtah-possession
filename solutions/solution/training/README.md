# Training the team model

This folder trains the team model's recipe on open data from **other games**. It never reads
labels of the provided match, which has none, or its public ground truth. Only `domain.py`
touches the provided match, and only its tracking: it checks that the degraded training data
looks like our tracking.

## Data

Seven matches from the German Bundesliga and 2. Bundesliga (Bassek, Rein, Weber, Memmert 2025, *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data; figshare
article 28196177; CC BY 4.0). The per-frame labels come with the data: which team has the ball
(`BallPossession`) and whether the ball is in play (`BallStatus`). They are part of the official
DFL position data produced by Sportec Solutions. We labelled nothing. `download.py` fetches
21 files, 2.6 GB, into `cache/dfl/raw/`, which git ignores; none of the data is in this
repository.

## Steps

From the repository root, after `pip install -r solutions/solution/requirements.txt`:

```bash
python solutions/solution/training/download.py          # -> cache/dfl/raw/  (2.6 GB)
python solutions/solution/training/dfl.py               # -> cache/dfl/npz/  (one file per match)
python solutions/solution/training/export.py            # -> cache/dfl/export/team_model.txt
python solutions/solution/training/experiments.py base  # leave-one-match-out bench
```

`export.py` takes about 10 minutes on 16 threads, and the bench about 15. Neither overwrites the
solution's `team_model.txt`.

## The recipe

| file | what it does |
|---|---|
| `download.py` | downloads the open DFL data |
| `dfl.py` | parses the XML into arrays: positions, teams, goalkeepers, ball, `BallPossession`, `BallStatus` |
| `sim.py` | degrades each DFL match so it looks like our tracking: the ball is hidden through most long stoppages and lost in runs during play; gaps up to 4 s are bridged by interpolation, as our tracker does; the camera closes in on long stoppages, so distant players drop out; single players are briefly lost; positions are smoothed and get homography noise; two opponents next to the ball trade tracks 12 times per match. Every amount comes from aggregate measurements of the provided match's tracking, with no labels involved |
| `core.py` | builds the solution's own `features.py` columns at 5 Hz rows while the ball is in play, trains LightGBM, and decodes with the solution's Viterbi (1.0 nat per change of team) |
| `export.py` | trains on all seven matches |
| `experiments.py`, `touch.py`, `domain.py` | the bench experiments below |

LightGBM settings: binary objective, 400 rounds, learning rate 0.05, 63 leaves, at least 200
rows per leaf, bagging 0.8, feature fraction 0.5, L2 1.0. These are the settings stored in the
shipped model file. Each match gets two degraded copies (seeds 100 and 101), and each copy is
also home/away-swapped and mirrored.

## The shipped model and this folder

Adrià trained the shipped `../team_model.txt` with his own first version of this recipe, in his
own workspace, and that code is not available. This folder is our re-implementation: same data,
features, LightGBM settings and seeds, but our own degradation code. Retraining with it gives a
different model file. Here is the submission with each model:

| team model | public spellF1 | public macroF1 | possessions | set pieces found | restart team | phase agreement |
|---|---|---|---|---|---|---|
| **shipped** (`../team_model.txt`) | **0.680** | **0.946** | 271 | 0.93 | 0.84 | 0.848 |
| retrained by `export.py` | 0.653 | 0.944 | 290 | 0.93 | 0.84 | 0.848 |

The dead ball is identical, and the two models disagree on 4.1% of live frames. The public
difference (−0.027, 5–95% paired bootstrap −0.133 to +0.047) is within noise. The retrained model
is not better, so the submission keeps the shipped one.

## Bench results

Leave-one-match-out: train on six degraded matches and test the whole pipeline on the seventh.

| this recipe | spellF1 | macroF1 |
|---|---|---|
| with the solution's dead-ball step | 0.535 | 0.866 |
| with the true dead ball | 0.633 | 0.930 |

Most of the remaining error is therefore in the dead ball, not the team. These experiments ran on
the previous version of the degradation, where the shipped recipe scored 0.539 and 0.635:

| experiment | result | decision |
|---|---|---|
| touch features (`touch.py`: who last changed the ball's motion) | −0.004 spellF1, better on 2 of 7 matches | dropped |
| decoder switch cost 0.5 / 1.0 / 2.0 nats | 1.0 best (0.631 / 0.635 / 0.628 with the true dead ball) | 1.0 kept |
| labels with brief touches removed ("controlled") | −0.011, better on 1 of 7 | dropped |
| a learned dead-ball step instead of the rules | 0.626 vs 0.539, better on all 7 | not adopted: an alive/dead F1 of 0.996 means the degraded DFL dead ball is far easier to see than ours, so the gain would not transfer |
| a perfect ball in play | +0.037 | an upper bound, not a method |
