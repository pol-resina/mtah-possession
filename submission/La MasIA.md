# Ball possession from tracking: a learned team model and a camera-aware dead ball

**Team La MasIA:** Adrià Capdevila, Jordi Rogerson, Martina Massana, Adrián Quirante, Pol Resina

**Code:** https://github.com/pol-resina/mtah-possession (`solution/`) ·
**Submitted file:** `La MasIA.csv`, sha256
`54aa51b1861922d2139e977b4ddf347c798f41900060722c3c09136a11d5da9d`

## 1. Approach

The submission labels every frame `H` / `A` / `D` / `X` from the provided **tracking alone**. We did
not use the video pixels, and Smart Tagging is not an input. The task splits into two questions,
and we answer them separately:

1. **Is the ball alive?** A rule-based dead-ball detector (`deadball.py`) built from the Laws of the
   Game and the physics of a stoppage: players slow down, the ball goes untracked or crosses a
   line, and play restarts with a kick from a legal spot.
2. **Who has it, while it is alive?** A LightGBM model (`team_model.txt`, 400 trees, 530 features)
   gives P(home has the ball) at 5 Hz. A Viterbi decoder then turns those probabilities into
   possessions, charging 1.0 nat for each change of team. Each live stretch between two stoppages
   is decoded on its own, so possession can change freely across a stoppage.

**Why this design.** The ball is untracked on 41% of the graded frames, so a nearest-player rule
has nothing to go on for long stretches. The team model therefore also reads the shape and
movement of both teams (centroids, spreads, lines, speeds, forward velocity) over windows of
0.4–10 s, before and after the frame. The labelling is offline, so looking ahead is allowed.
Every feature describes a team, the pitch or the ball, never a single track, so the tracker's
identity switches barely affect it. We kept the dead ball rule-based because it is where the
Laws of the Game give the most structure, and because on our external bench the dead ball, not
the team, was the larger source of error.

We could not choose between designs on the public 5 minutes: they hold only 17 possessions, and
the spellF1 bootstrap band of our own file there runs from 0.48 to 0.84. We chose with:

- an external bench (§3) for the team model and the decoder's switch cost;
- whole-match checks that need no ground truth, for the dead ball: dead-ball share, number and
  length of possessions, and agreement with Smart Tagging's automatic set pieces and phases (used
  as a check, never as an input);
- the public window, only to confirm that the pipeline is aligned and well-formed.

| file | public spellF1 | public macroF1 |
|---|---|---|
| organisers' best baseline (`baseline_smooth125`) | 0.424 | 0.581 |
| our rules-only predecessor (earlier version, commit `5e9d334`) | 0.581 | 0.926 |
| **submitted (`solution`)** | **0.680** (band 0.48–0.84) | **0.946** |

## 2. Assets: used and dropped

| asset | use |
|---|---|
| Tactical tracking (players, ball) | **used**: every feature, and the dead-ball score |
| Broadcast tracking + homography | **used**, two flags per frame only: is the broadcast on the game shot (valid homography), and does it have the ball. It is aligned to the tactical timeline with one constant offset per half, found automatically where the two cameras' balls coincide most |
| Kickoff frames of the two halves | **used**: from the organisers' starter kit and FAQ |
| Smart Tagging | **not an input**. Used only as a whole-match check of the output |
| Tactical / broadcast video pixels | **dropped** (see below) |

Tried and dropped (each judged by checks fixed in advance, not by the public window):

- **Video**:
  - a ball detector on the tactical video, both as a white-blob detector and as a CNN trained on
    frames where both trackers agree. Neither beat a plain smoothing control: aerial balls were
    projected onto defenders, and the ball was confused with white socks.
  - shirt colour as each track's team. It fixed a few identity switches in our rule-based
    predecessor and brought nothing under the learned team model.
- **Smart Tagging as an input** for the restart team: dropped, because it would have made our own
  checks circular.
- **Team model**:
  - touch features (−0.004 spellF1 on the bench);
  - adding SkillCorner A-League open data to training (−0.010, better on 1 of 7 matches);
  - the restart team as the decoder's opening state (+0.000);
  - averaging the model over both cameras' balls;
  - filling tactical ball gaps with the broadcast ball.
- **Decoding**:
  - holding the last team while no camera has the ball;
  - a minimum-control rule ("a possession sandwiched between two others needs 2 players or
    ≥ 2 s"). The public ground truth itself contains won tackles shorter than 1.5 s, so it does
    not follow that convention.
  - contact-quality weighting, duel evidence and a foul-window rule.

## 3. Training data

The only learned component is the team model. **None of its training data comes from the
provided game, and we hand-labelled nothing, on any game.**

| dataset | source and licence | labels, and who made them | how much |
|---|---|---|---|
| DFL open data (used) | Sportec Solutions / DFL, published with Bassek et al. 2025, CC BY 4.0 | per-frame ball status (alive/dead) and ball-possession team, from the official DFL position data produced by Sportec Solutions | 7 full matches, 25 Hz tracking, sampled at 5 Hz |
| SkillCorner A-League open data (tried, dropped) | SkillCorner open data | SkillCorner's own data | not in the final model |

**How the DFL data was used.** The DFL tracking is far cleaner than the provided tracking, so each
match was degraded to resemble it: the ball was lost in long blocks, the goalkeepers were swapped,
and player tracks were fragmented. Two degraded copies were made per match (seeds 100 and 101),
and each copy was also home/away-swapped and mirrored. We evaluated leave-one-match-out on the 7
matches. On that bench the model reaches spellF1 0.434 and macroF1 0.800, with a team error of
0.087 when the ball is seen and 0.210 when it is not. Our rule-based predecessor (nearest-player
contact, tactical camera only) reaches 0.289 and 0.738 on the same bench, and the model beats it
on all 7 matches (sign test p = 0.008). The 1.0-nat switch cost was the best value in every
held-out fold. The shipped model is trained on all 7 matches.

**Training code.** `solution/training/` contains the whole training pipeline:
downloading the open data, degrading it, training, and the leave-one-match-out bench. It is a
re-implementation of the recipe the shipped model was trained with, using the same data,
features, LightGBM settings and seeds. Retraining with it gives a model that is not byte-identical.
Used in the submission, that model scores public spellF1 0.653 against 0.680 (a paired
difference within noise) with the same whole-match checks, so we kept the shipped model.
`training/README.md` has the steps and the comparison.

## 4. Possession rules

- **Ball out of play (`D`).** Play is dead from the moment the evidence says so (players slowing
  below a 2.6 m/s neutral point, ball untracked, or more than 0.5 m beyond a line) until the
  restart kick, plus a 10-frame lag.
  - Once the ball has crossed a line, play stays dead until a restart from a spot the Laws allow
    for that line: a throw-in near the touchline, a goal kick from the goal area, a corner from
    the corner arc, a kickoff from the centre spot.
  - Stoppages shorter than 2 s are dropped, and live gaps shorter than 1 s between two stoppages
    are merged.
- **Restarts.** A restart is the ball going from quiet (untracked, out, or slower than 3 m/s) to
  played (inside the pitch at ≥ 5 m/s).
  - A restart is not believed if the dead-ball evidence continues for ≥ 4 s after it while neither
    camera has the ball (a ball set down for a corner, or kicked out of the net).
  - A stoppage that ends in a kickoff (≥ 90% of tracked players in their own half) is a goal, and
    starts when the ball was last tracked, not when the players slowed down.
  - The team that restarts is whatever the team model says on the first live frames: we do not
    take it from the set-piece type.
- **Loose balls.** There is no "loose" state. Possession stays with the last team the model
  favoured until the evidence for the other team outweighs the 1-nat switch cost, so a brief
  deflection or a 50/50 ball does not change possession.
- **Challenges.** A won tackle changes possession, however short the new spell, once the model sees
  the other team on the ball. There is no minimum possession length.
- **`X`.** Frames before the first-half kickoff, at half-time and after the final whistle are
  `X`. Inside the two halves we never abstain.
- **Goalkeepers.** The trackers swap the two goalkeepers' tracks, so a goalkeeper track is given
  the team that defends the half it stands in.

## 5. Limitations

- **Changes with no trace in the tracking.** Most of the remaining error on the public window is
  in-play changes of possession that leave little trace in the tracking. In those cases the
  tracker's interpolated ball sits in the wrong place, a touch without control is counted, or a
  short won duel is missed. With the true dead ball and restart team substituted, our public
  spellF1 does not move, so the team decisions inside live play are the bottleneck.
- **Dead-ball errors.**
  - A goalkeeper holding the ball, or slow build-up during a broadcast cut-away, can be mistaken
    for a stoppage.
  - Long runs where both trackers lose the ball are usually real play (corners, scrambles in the
    box) and are left live, so a few real stoppages inside such runs are missed.
- **Domain shift.** The team model learned on degraded German-league tracking. Our degradation
  imitates this tracker's failure modes but cannot reproduce them exactly.
- **Validation.** The public window is too small to rank changes, and Smart Tagging tags coarse
  phases, so it rewards any change that merges short possessions. Our evidence is therefore the
  external bench plus a set of whole-match checks.
- **With more time:**
  - a ball detector good enough to overrule the tracker's interpolation in the 41% of frames
    without a ball;
  - a dead-ball model trained on external matches with a degradation closer to this tracker;
  - hand-labelled possessions on other games, which the rules allow, to train the decoder's
    notion of control.

## 6. Reproduction

Python 3.14 (tested 3.14.4, Linux x86-64). From a fresh clone, with the match bundle unpacked into
`data/` (or `MTAH_DATA_DIR` pointing at it):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.txt   # numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, lightgbm 4.6.0
python solution/run.py --name "La MasIA"   # -> evaluation/results/La MasIA.csv, ~25 s
python evaluation/validate_submission.py "evaluation/results/La MasIA.csv"
sha256sum "evaluation/results/La MasIA.csv"   # 54aa51b1861922d2139e977b4ddf347c798f41900060722c3c09136a11d5da9d
```

The pipeline is deterministic. We checked this with a fresh clone in a new environment, which
regenerated the submitted file byte for byte.
