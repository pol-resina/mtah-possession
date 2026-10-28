# Ball possession from tracking data

Labels every frame `H` / `A` / `D` from the two cameras' tracking data (ATD) alone: no video
pixels, no Smart Tagging, no ground truth.

| public 5 minutes | spellF1 | macroF1 |
|---|---|---|
| **this solution** | **0.581** | **0.926** |
| same, tactical camera only (`--no-broadcast`) | 0.520 | 0.913 |
| best baseline here (`baseline_smooth125`) | 0.424 | 0.581 |

17 possessions are too few to rank versions: the spellF1 bootstrap band is 0.39–0.76.

```bash
python solution/run.py                  # writes evaluation/results/solution.csv
python evaluation/run_evaluation.py     # scores it
python solution/diagnostics.py --sweep  # checks it (see below)
```

## The one idea used twice

Both questions -- *is the ball dead?* and *which team has it?* -- are answered the same way
(`segments.two_state`): every frame gets a cost for each of two answers, and a Viterbi picks the
labelling that minimises

    sum of per-frame costs  +  a fixed penalty for every change

so the answer only changes when enough evidence piles up. That is what turns a flickering
per-frame signal into whole stoppages and whole possessions.

## Pipeline

| file | does |
|---|---|
| `tracking.py` | loads the tactical tracking in metres; fixes the goalkeeper tracks, which the tracker swaps between the two keepers (a keeper track gets the team defending the half it is in) |
| `deadball.py` | finds the stoppages → `D` |
| `possession.py` | decides the team in between → `H` / `A` |
| `broadcast.py` | adds the broadcast camera's tracking as a second witness |
| `run.py` | runs it all and writes the CSV |

### 1. Dead ball (`deadball.py`)

1. **Score** each frame: players walking rather than running (speed of the five fastest),
   ball not tracked, ball beyond a line, few players tracked.
2. **Smooth** it with the Viterbi: a stoppage needs a couple of seconds of evidence.
3. **Latch**: once the ball crosses a line, play is dead until a restart from a spot the Laws of
   the Game allow for that line -- throw-in at the touchline, goal kick from the goal area,
   corner from the corner, kickoff from the centre.
4. **Tidy**: drop blips shorter than a second or two.
5. **Restart**: end each stoppage 10 frames after its restart (the ball going from still to
   moving). A short stoppage with no line crossing and no restart is slow build-up play, and is
   dropped.

### 2. Possession (`possession.py`)

- **Evidence**: a frame counts only when the ball is within 1.5 m of a player of exactly one team.
  That names the right team 92% of the time; "nearest player", the baseline's signal, 79%.
- **Decoding**: the Viterbi again. A change of possession must outweigh 8 frames of contact, so
  single touches and deflections are ignored; between contacts the ball stays with the last team.
- **Timing**: every change is moved 20 frames later -- the ground truth changes when the new team
  has *control*, not at its first touch.
- **Restarts**: the team restarting play starts each live stretch -- by the Laws at a corner
  (attacking side) or goal kick (defending side), else the player standing at the ball, else the
  side that did not touch it last.

### 3. Second camera (`broadcast.py`)

The broadcast camera follows the ball up close and sees duels the wide tactical camera blurs.

1. **Align**: a live broadcast runs in real time, so one offset per half maps it onto the tactical
   frames (found from the ball tracks: 4,014 and 26,920 frames).
2. **Gate**: use a broadcast frame only when its players stand where the tactical players are --
   that rules out replays and close-ups.
3. **Contact**: read exactly as above, and count it (at 0.72 of a tactical frame) only where the
   tactical camera sees no clear contact.

## Where the numbers come from

Nothing was chosen by trying values against the public score. Four numbers were *measured* on the
public window; everything else is physics, the Laws of the Game, or the full match itself.

| number | value | source |
|---|---|---|
| change delay (`CONTROL_LAG`) | 20 frames | median gap between the new team's first contact and the labelled change |
| restart delay (`RESTART_LAG`) | 10 frames | median gap between detected restart and first live frame (0, 9, 11, 25) |
| ball missing (`MISSING_WEIGHT`) | 1.5 | log of how much likelier an untracked ball is in stoppages (34% vs 6.5% of frames) |
| broadcast weight | 0.72 | ratio of the cameras' log-odds of naming the right team (85% vs 92%) |
| walking/running threshold | 2.6 m/s | valley of the full-match histogram of player speed |
| restart spots | goal area, corners... | Laws of the Game |
| "false restart" cut | 20 s | gap in the full-match distribution: 95 of 99 restarts under 10 s, 4 over 20 s |

## Checking it without overfitting (`diagnostics.py`)

- **Whole-match checks, no labels needed**: share of dead ball (real matches 40–50%; ours 49%),
  number of possessions (303), share of Smart Tagging's tagged restarts we find (92%), restart
  team agreement (79%), team agreement inside Smart Tagging's attacking phases (80%).
- **`--sweep`**: every parameter halved and doubled, one at a time. Most rows stay within the
  noise; the most sensitive number is the walking/running threshold (3.0 m/s instead of 2.6 drops
  public spellF1 to 0.35 and pushes dead ball to 53%).
- **`paired_spell_difference(old, new)`**: compares two versions on the same bootstrap resamples of
  the public possessions -- the honest way to ask whether a change helped. The broadcast camera:
  +0.061 (5–95%: −0.033 to +0.175).

## Tried, and left out

Lower switch costs, weighting contact by ball control, evidence from duels, ignoring a change just
before a foul, a self-supervised team-shape classifier: none helped. With perfect dead-ball
stretches and restart teams put in by hand, public spellF1 does not move at all -- what remains is
short possessions inside open play that the tracking barely shows.
