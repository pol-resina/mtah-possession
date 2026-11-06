# solution_adria: Pol's dead ball + a learned team model

Labels every frame `H` / `A` / `D` from the tactical camera's tracking only. No video, no Smart
Tagging, no ground truth.

```bash
python solutions/solution_adria/run.py      # writes evaluation/results/solution_adria.csv (~4 min)
python evaluation/run_evaluation.py         # scores it
```

## What it changes

| step | from |
|---|---|
| dead ball (`D`) | `deadball.py`, copied unchanged from `solution_pol_5e9d334` (with `tracking.py`, `segments.py`, `possession.py`) |
| team in live stretches (`H` / `A`) | **new**: LightGBM P(home \| ball alive) at 5 Hz, decoded with the same two-state Viterbi (`segments.two_state`) per live stretch, switch cost 1 nat |

Pol's team step only has evidence when the ball is within 1.5 m of one team; with the ball
untracked (~40 % of frames) it just holds the last team. The learned model also reads the
players' shape and movement (team centroids, spreads, speeds, forward velocity, nearest player per
team, all over 0.4-10 s windows: `features.py`), so it still says something with the ball unseen.

## Evidence

Judged on an external bench, not on the public 5 minutes (17 possessions, too few to rank):
7 DFL Bundesliga matches (Sportec open data, Bassek et al. 2025, CC BY 4.0) with real ball-status
and possession labels, degraded to look like our ATD (ball lost in long blocks, keeper swaps,
fragmented slots), leave-one-match-out.

| | bench spellF1 | bench macroF1 | team error, ball seen / unseen | whole match: agreement with Smart Tagging team |
|---|---|---|---|---|
| `solution_pol_5e9d334 --no-broadcast` | 0.289 | 0.738 | 0.168 / 0.413 | 0.745 |
| **solution_adria** | **0.434** | **0.800** | **0.087 / 0.210** | **0.768** |

Wins on all 7 bench matches (sign test p = 0.008). Dead ball is identical by construction (dead
48.9 %, ball seen outside the lines labelled D 0.962). Public window, sanity only: spellF1 0.680,
macroF1 0.946.

Tried and left out (same bench): the restart team as the opening state (+0.000), SkillCorner
A-League matches added to the team model's training (-0.010, 1/7), the broadcast contact as in
`solution_pol_5e9d334` (no change on the whole match), switch costs 2 and 4 (-0.004, -0.023).

## Requirements

On top of `requirements.txt`: `lightgbm` (tested 4.6.0) and `scipy` (1.14.1).

## The model file

`team_model.txt` (LightGBM text, 400 trees) and `team_model_features.txt` (feature order) are
trained by `bench/export_team.py` in Adrià's workspace (`FCB_challenges/bench/`) on all 7 degraded
DFL matches, training seeds 100/101, each also H/A-swapped and mirrored. The DFL data is not
in this repository. `features.py` is a copy of `bench/features.py`: keep the two in step, or the
model reads the wrong columns.
