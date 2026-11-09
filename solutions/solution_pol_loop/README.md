# solution_pol_loop: Adrià's team model on a sharper dead ball

Labels every frame `H` / `A` / `D` from the two cameras' tracking. No video, no Smart Tagging, no
ground truth.

```bash
python solutions/solution_pol_loop/run.py   # writes evaluation/results/solution_pol_loop.csv (~25 s)
python evaluation/run_evaluation.py         # scores it
```

## What it changes

The team step is `solution_adria` unchanged (LightGBM P(home), the same Viterbi). The dead-ball
step is Pol's `deadball.py` with three additions. From the broadcast camera's tracking only two flags
per frame are used: is it on the game shot (a valid homography), and does it have the ball.

| addition | rule | source of its numbers |
|---|---|---|
| cut-aways | while the ball is quiet, a broadcast frame cut away from the game shot (replay, close-up) adds 1.54 log-odds to the dead-ball score | measured on the whole match: 61.3 % of quiet dead frames are cut away, 13.1 % of quiet live ones. Research exp 003 (branch `autoresearch_pol`), accepted there |
| goals | a stoppage that ends with a kickoff starts when the ball was last tracked, not when the players slow down | Laws 8 and 10. Kickoff = at the restart, at least 0.9 of the tracked players are in their own half; over the 115 restarts that share is 0.4-0.8 for 110 and 0.95-1.00 for 5, nothing in between |
| false restarts | a restart kick is not believed when the dead-ball evidence goes on for 4 s or more after it and no camera has the ball on most of those frames | 4 s is the tracker's own gap limit; "most" is 0.5. Both fixed before the result was seen |

## Evidence

Nothing was chosen with the public 5 minutes; they were opened once per change, afterwards, and none
of the three changes a single frame inside them (public spellF1 0.680, macroF1 0.946, as
`solution_adria`). Smart Tagging is not an input, so it can check the output.

| whole match | solution_adria | + cut-aways | + goals | + false restarts |
|---|---|---|---|---|
| dead-ball share of graded frames | 0.489 | 0.497 | 0.500 | 0.508 |
| frames moved from live to dead | - | 1,307 | 451 | 1,264 |
| stoppages | 118 | 115 | 115 | 115 |
| possessions | 286 | 275 | 273 | 271 |
| tagged set pieces with a stoppage ending in their window (of 59) | 54 | 55 | 55 | 55 |
| restart team agrees with Smart Tagging | 20/24 | 21/25 | 21/25 | 21/25 |
| Smart Tagging phase agreement, live frames | 0.840 | 0.841 | 0.845 | 0.848 |

- **cut-aways**: reproduces exp 003's whole-match numbers exactly; that experiment also had a video
  audit (5 right, 1 wrong).
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

## Tried on top and left out

Decided by checks fixed in advance: averaging the model over both
cameras' balls; filling tactical ball gaps with the broadcast ball; holding the last team while no
camera has the ball; "a restart needs a ball" as a blind-run rule; shirt-colour slot teams (research
exp 008) under this model; a last-touch-before-a-stoppage rule (the public ground truth itself has
possessions under 1.5 s, so "under 2 s is not control" is not how it was annotated).

## Requirements

As `solution_adria`: `lightgbm` (4.6.0) and `scipy` on top of `requirements.txt`.
