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

## The dead ball

`deadball.py` decides, half by half from the kickoff, which graded frames are stoppages (graded
frames before a half's kickoff are dead). It works in six steps:

1. **Score.** Every frame gets a dead-ball score in log-odds (the logarithm of how much likelier a
   stoppage is than live play; positive means stoppage). It adds up the players' speed (the mean
   of the five fastest, neutral at 2.6 m/s: walking below, running above), the ball being
   untracked or more than 0.5 m beyond a line, fewer than 18 players tracked, and a broadcast
   cut-away while the ball is quiet.
2. **Smooth.** A two-state minimum-cost labelling, the same kind as for the team, turns the score
   into stoppages: opening or closing one needs a couple of seconds of consistent evidence.
3. **Latch.** Once the ball crosses a line, play stays dead until a restart from a spot the Laws
   of the Game allow for that line: a throw-in from the touchline, a goal kick from the goal area,
   a corner from the corner arc, or a kickoff from the centre spot after a goal.
4. **Tidy.** Live gaps shorter than 1 s between two stoppages are merged, and stoppages shorter
   than 2 s are dropped.
5. **Restart.** A restart kick is the ball going from quiet (untracked, out, or slower than
   3 m/s) to played (inside the pitch at 5 m/s or more). Each stoppage ends 10 frames after its
   kick; one shorter than 6 s with no line crossing and no kick is slow build-up play, and is
   dropped.
6. **Goals.** A stoppage that ends in a kickoff (at least 90 % of the tracked players in their own
   half) starts when the ball was last tracked, not when the players slowed down.

Steps 1, 5 and 6 hold the three dead-ball additions described below.

## The broadcast camera

`broadcast.py` puts the broadcast camera's tracking on the tactical frames, in two steps:

1. **Align.** The two recordings start at different moments, but a live broadcast runs in real
   time, replays included, so one constant offset per half maps every broadcast frame onto its
   tactical frame. It is the offset at which the two cameras' balls are within 2 m of each other on
   the most seconds, refined to the single frame; if even the best one leaves them 2 m or more
   apart, that half gets no broadcast data. For this match: 4,014 frames in the first half and
   26,920 in the second.
2. **Cut.** A replay, close-up or crowd shot has no valid homography, so a tactical frame whose
   broadcast frame has none is marked as cut away: a sign, not proof, that play is stopped.

`run.py` passes two things on to the dead-ball step: the cut-aways (step 1) and the broadcast's
ball, to tell when neither camera has the ball (the false restarts of step 5).

## The three dead-ball additions

The dead-ball step has three rules that use the broadcast camera or the Laws of the Game.

| addition | rule | source of its numbers |
|---|---|---|
| cut-aways | while the ball is quiet, a broadcast frame cut away from the game shot (replay, close-up) adds 1.54 log-odds to the dead-ball score | measured on the whole match: 61.3 % of quiet dead frames are cut away, 13.1 % of quiet live ones; 1.54 = ln(0.613 / 0.131) |
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

- **cut-aways**: the weight is the log of the ratio of the two whole-match shares above. Stoppages,
  set pieces and restart team are the same at half and at double the weight.
- **goals**: 3 stoppages start 4.7-8.0 s earlier (the fourth goal's already started on time). The
  detector finds 5 kickoffs; Smart Tagging tags 6 = the first-half start, which is not a restart
  here, plus the second-half start and 4 after a goal. The model had given those gaps to a team, most
  often the one that conceded.
- **false restarts**: 5 stoppage ends move 5.6-16.4 s later. For one tagged corner the stoppage end
  goes from 16.3 s before the tag's usual restart time to 0.2 s after it. The rule targets a ball
  that is moved but not played, such as one kicked out of the net after a goal or placed for a
  corner well before it is taken. It can misfire on a short goal kick that the keeper then holds at
  his feet.

Smart Tagging's phase agreement is reported but proves little: it tags coarse phases and disagrees
with short possessions everywhere, so it rewards any change that removes one.

## The team features

`features.py` builds 530 columns per frame from the tactical tracking. None follows a single
player: every feature is about a team, the whole pitch or the ball, so the tracker's identity
switches barely matter. Positions are measured towards the goal the home team attacks, so a
feature means the same in both halves, and goalkeeper tracks get the team defending their half.

| group | features |
|---|---|
| ball | tracked or not, position, distance to the lines, speed, change of direction, time since and until it is tracked, still or not; while it is lost, its last known position |
| ball and teams | the nearest player of each team and their difference, players of each team within 3 m and 5 m, which team is closer when a player is within 1.5 m, the ball's distance ahead of each team's centroid |
| without the ball | per team: centroid, spread, front and back line, mean and 90th-percentile speed, forward velocity, players tracked; the speed of the five fastest players overall |
| time | seconds since the half's kickoff |
| context | the mean, minimum and maximum of 16 of the above over windows of 0.4, 1, 2, 5 and 10 s, before and after the frame (the match is labelled offline, so looking ahead is allowed) |

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
order) were trained by Adrià on all 7 degraded DFL matches, with training seeds 100 and 101 and
each match also home/away-swapped and mirrored. The DFL data is not in this repository.
[`training/`](training/README.md) holds the training code: download, degradation, training, and
the leave-one-match-out bench. It re-implements the same recipe. Retraining with it gives a model
that is not byte-identical and is not better on our match, so the shipped model is kept;
`training/README.md` has the comparison.
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

## Reproduction

Python 3.14 (tested 3.14.4 on Linux x86-64). The pinned versions in `requirements.txt` next to
this file reproduce the submitted file byte for byte, from a fresh clone:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solutions/solution_pol_loop/requirements.txt
# unpack the match data bundle into data/ (see data/README.md), or point MTAH_DATA_DIR at it
python solutions/solution_pol_loop/run.py      # -> evaluation/results/solution_pol_loop.csv
python evaluation/validate_submission.py evaluation/results/solution_pol_loop.csv
sha256sum evaluation/results/solution_pol_loop.csv
# 54aa51b1861922d2139e977b4ddf347c798f41900060722c3c09136a11d5da9d
```

`run.py --name <team_name>` writes the same labels as `evaluation/results/<team_name>.csv`.
