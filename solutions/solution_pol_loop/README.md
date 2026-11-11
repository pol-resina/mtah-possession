# Ball possession from tracking: a learned team model and a camera-aware dead ball

Labels every frame `H` / `A` / `D` (and `X` outside the two graded halves, as in the task) from the
tactical and broadcast cameras' tracking. No video, no Smart Tagging, no ground truth.

```bash
python solutions/solution_pol_loop/run.py   # writes evaluation/results/solution_pol_loop.csv (~25 s)
python evaluation/run_evaluation.py         # scores it
```

## How it works

1. **Load the tracking** (`tracking.py`). The tactical camera's tracking is read as one row per
   video frame. The tracker sometimes swaps the two goalkeepers between their tracks, so a
   goalkeeper track is given the team that defends the half it is standing in.
2. **Find the dead ball** (`deadball.py`, `broadcast.py`). Each frame gets a dead-ball score from
   the tracking (how fast the fastest players move, whether the ball is tracked or beyond a line,
   how many players are tracked) and from the broadcast camera. A minimum-cost labelling turns the
   score into stoppages, and a stoppage ends at a restart from a spot the Laws of the Game allow.
   From the broadcast camera's tracking only two flags per frame are used: is it on the game shot
   (a valid homography), and does it have the ball.
3. **Choose the team** (`features.py`, `run.py`). A LightGBM model (gradient-boosted decision trees)
   gives P(home has the ball) while the ball is alive. It is evaluated at 5 Hz (5 samples per
   second, one every 5th frame of the 25 fps tracking), and each sample's probability is copied
   to the nearest frames. The ball is untracked on about 41 % of the graded frames, so the model
   also reads the players' shape and movement (team centroids, spreads, speeds, forward velocity,
   nearest player per team, over windows of 0.4-10 s) and still says something with the ball
   unseen.
4. **Smooth the team labels** (`segments.py`). Each live stretch is labelled home or away on its
   own by a minimum-cost labelling (Viterbi decoding) in which every change of team costs 1.0 nat
   (the unit of information for the natural logarithm; a probability `p` costs `-log p`). This
   stops the label flickering on noisy frames, and because each live stretch is decoded
   separately, possession can change freely across a stoppage.

## The three dead-ball additions

The dead-ball step has three rules that use the broadcast camera or the Laws of the Game.

| addition | rule | source of its numbers |
|---|---|---|
| cut-aways | while the ball is quiet, a broadcast frame cut away from the game shot (replay, close-up) adds 1.54 log-odds to the dead-ball score | measured on the whole match: 61.3 % of quiet dead frames are cut away, 13.1 % of quiet live ones; a video audit of the changes found 5 right and 1 wrong |
| goals | a stoppage that ends with a kickoff starts when the ball was last tracked, not when the players slow down | Laws of the Game 8 and 10 (start and restart of play; determining the outcome of a match). Kickoff = at the restart, at least 0.9 of the tracked players are in their own half; over the 115 restarts that share is 0.4-0.8 for 110 and 0.95-1.00 for 5, nothing in between |
| false restarts | a restart kick is not believed when the dead-ball evidence goes on for 4 s or more after it and no camera has the ball on most of those frames | 4 s is the longest gap the tracker bridges when it loses the ball; "most" is 0.5. Both fixed before the result was seen |

## Evidence

Nothing was chosen with the public 5-minute ground truth; it was opened once per change,
afterwards, and none of the three changes a single frame inside it (public `spellF1` 0.680,
`macroF1` 0.946, the same as without the additions). Smart Tagging is not an input, so it can check
the output.

| whole match | no additions | + cut-aways | + goals | + false restarts |
|---|---|---|---|---|
| dead-ball share of graded frames | 0.489 | 0.497 | 0.500 | 0.508 |
| frames moved from live to dead | - | 1,307 | 451 | 1,264 |
| stoppages | 118 | 115 | 115 | 115 |
| possessions | 286 | 275 | 273 | 271 |
| tagged set pieces with a stoppage ending in their window (of 59) | 54 | 55 | 55 | 55 |
| restart team agrees with Smart Tagging | 20/24 | 21/25 | 21/25 | 21/25 |
| Smart Tagging phase agreement, live frames | 0.840 | 0.841 | 0.845 | 0.848 |

- **cut-aways**: the 61.3 % / 13.1 % shares above are the whole-match measurement; the video audit
  found 5 right and 1 wrong.
- **goals**: 3 stoppages start 4.7-8.0 s earlier (the fourth goal's already started on time). The
  detector finds 5 kickoffs; Smart Tagging tags 6 = the first-half start, which is not a restart
  here, plus the second-half start and 4 after a goal. The model had given those gaps to a team, most
  often the one that conceded.
- **false restarts**: 5 stoppage ends move 5.6-16.4 s later. For one tagged corner the stoppage end
  goes from 16.3 s before the tag's usual restart time to 0.2 s after it. Stills of the five (looked
  at to understand them, not as labels): a keeper kicking the ball out of the net after a goal, the
  ball placed for a corner 16 s before it is taken, a ball rolled to the keeper during a replay, the
  scuffle at the final whistle; and one that may be wrong, a short goal kick the keeper then holds at
  his feet for 10 s.

Smart Tagging's phase agreement is reported but proves little: it tags coarse phases and disagrees
with short possessions everywhere, so it rewards any change that removes one.

## The team model

The public 5 minutes hold only 17 possessions, too few to rank anything, so the team model was
judged on an external bench instead: 7 matches of the Deutsche Fußball Liga (DFL) Bundesliga
(Sportec open data, Bassek et al. 2025, CC BY 4.0) with real ball-status and possession labels,
degraded to look like this tracking (ball lost in long blocks, goalkeeper swaps, fragmented
slots), leave-one-match-out. On that bench it reaches `spellF1` 0.434 and `macroF1` 0.800, with a
team error of 0.087 when the ball is seen and 0.210 when it is not, and it wins on all 7 matches
(sign test p = 0.008). The 1.0-nat change penalty was the best value in every held-out fold; 2 and
4 were worse (-0.004, -0.023 `spellF1`).

`team_model.txt` (LightGBM text format, 400 trees) and `team_model_features.txt` (the feature
order) were trained outside this repository on all 7 degraded DFL matches, training seeds 100 and
101, each match also home/away-swapped and mirrored. The DFL data is not in this repository.
`features.py` must build the columns in the order of `team_model_features.txt`, or the model reads
the wrong columns.

## Tried on top and left out

Decided by checks fixed in advance: averaging the model over both cameras' balls; filling tactical
ball gaps with the broadcast ball; holding the last team while no camera has the ball; "a restart
needs a ball" as a blind-run rule; shirt-colour slot teams under this model; a last-touch-before-a-
stoppage rule (the public ground truth itself has possessions under 1.5 s, so "under 2 s is not
control" is not how it was annotated). On the external bench: using the restart team as the
opening state (+0.000), and adding SkillCorner A-League matches to the team model's training data
(-0.010 `spellF1`, better on 1 of 7 matches).

## Requirements

`lightgbm` (tested 4.6.0) and `scipy` (1.14.1), on top of `requirements.txt`.
