"""Dead ball: which graded frames are stoppages, and where each one ends.

Each half goes through five steps (segment_half):

  1. score    every frame gets a dead-ball score in log-odds, from four tracking signals: how fast
              the five fastest players move, whether the ball is tracked, whether it is beyond a
              line, and whether few players are tracked
  2. smooth   a two-state Viterbi turns the score into stoppages: one needs a couple of seconds of
              evidence to open, and again to close
  3. latch    once the ball crosses a line, play is dead until a restart from a spot the Laws of
              the Game allow for that line (throw-in, goal kick, corner, kickoff)
  4. tidy     very short live gaps between stoppages are absorbed, very short stoppages dropped
  5. restart  each stoppage ends RESTART_LAG frames after its restart kick; a short one that no line
              crossing explains and nobody restarts is slow build-up play, and is dropped

Where each number comes from is in solution/README.md.
"""

from collections import namedtuple

import numpy as np

from .segments import rolling_mean, runs, two_state

# [start, stop) in video frames, the restart kick (or None), and the frame the ball crossed a line
# if that is how the stoppage began (or None).
Stoppage = namedtuple("Stoppage", "start stop kick crossed")

# Params
# 1. score
ACTIVITY_TOP_K = 5            # speed of the five fastest players...
ACTIVITY_HALF_WIDTH = 12      # ...averaged over a 1 s centred window
ACTIVITY_MID = 2.6            # m/s where the score is neutral: walking below, running above
ACTIVITY_SCALE = 0.5          # m/s per unit of log-odds
ACTIVITY_CLIP = 4.0           # no single frame counts for more than this, either way
MISSING_WEIGHT = 1.5          # ball not tracked
OUT_MARGIN = 0.5              # metres beyond a line before the ball counts as out
OUT_WEIGHT = 4.0              # ball out
FEW_PLAYERS = 18              # fewer tracked players than this: players bunched together
FEW_PLAYERS_WEIGHT = 1.0
# 2. smooth
SWITCH_COST = 40.0            # evidence a stoppage needs to open, and again to close
KICKOFF_GUARD = 50            # frames after a kickoff that stay live whatever the score says
# 3. latch: where a restart may be taken, widened by SPOT_MARGIN because a restart is detected a
# few frames after the kick, a few metres down the ball's path
OUT_MIN_FRAMES = 3            # frames beyond a line before it counts as a crossing
OUT_RESTART_REACH = 60        # seconds a crossing waits for its restart
THROW_IN_BAND = 2.0           # metres inside the touchline
GOAL_AREA = (5.5, 9.16)       # depth and half-width, metres
CORNER_REACH = 1.0            # metres from the corner flag
CENTRE_REACH = 1.0            # metres from the centre spot (a goal, then a kickoff)
SPOT_MARGIN = 3.0
# 4. tidy
MIN_ALIVE = 25                # frames; shorter live gaps between stoppages are absorbed
MIN_DEAD = 50                 # frames; shorter stoppages are dropped
# 5. restart: the ball going from quiet to played
RESTART_SPEED = 5.0           # m/s; at least this fast inside the pitch, the ball is being played
QUIET_SPEED = 3.0             # m/s; slower than this (or untracked, or out), it is not
QUIET_LOOKBACK = 50           # frames before a restart that must be mostly quiet...
QUIET_FRACTION = 0.6          # ...this share of them
RESTART_NOT_BEFORE = 25       # frames into a stoppage before its restart can come
RESTART_SEARCH_AFTER = 25     # frames past a stoppage's scored end still searched for its restart
RESTART_TRUST = 500           # a restart followed by this many more frames of dead score is false
RESTART_LAG = 10              # frames from the detected restart to the first live frame
UNCONFIRMED_MIN = 150         # frames; a shorter stoppage needs a line crossing or a restart


# -- signals ------------------------------------------------------------------------------------

def activity(match):
    """Mean speed of the ACTIVITY_TOP_K fastest players on each frame, 1 s centred mean, m/s."""
    speed = np.where(np.isnan(match.speed), -1.0, match.speed)
    fastest = np.sort(speed, axis=1)[:, -ACTIVITY_TOP_K:]
    counted = fastest >= 0
    n = counted.sum(axis=1)
    per_frame = np.where(n > 0, np.where(counted, fastest, 0.0).sum(axis=1) / np.maximum(n, 1),
                         np.nan)
    return rolling_mean(per_frame, ACTIVITY_HALF_WIDTH)


def ball_out(match):
    """True where the ball is tracked and more than OUT_MARGIN beyond a line."""
    with np.errstate(invalid="ignore"):
        return match.ball_seen & ((match.bx < -OUT_MARGIN) | (match.bx > match.length + OUT_MARGIN)
                                  | (match.by < -OUT_MARGIN) | (match.by > match.width + OUT_MARGIN))


def dead_score(match):
    """Step 1: per-frame log-odds that the ball is dead. Positive means stoppage."""
    score = np.clip((ACTIVITY_MID - np.nan_to_num(activity(match), nan=0.0)) / ACTIVITY_SCALE,
                    -ACTIVITY_CLIP, ACTIVITY_CLIP)
    tracked = (~np.isnan(match.px)).sum(axis=1)
    return (score + MISSING_WEIGHT * ~match.ball_seen + OUT_WEIGHT * ball_out(match)
            + FEW_PLAYERS_WEIGHT * (tracked < FEW_PLAYERS))


def played_and_quiet(match):
    """(played, quiet) per frame: the ball moving fast inside the pitch / not being played."""
    out = ball_out(match)
    with np.errstate(invalid="ignore"):
        played = match.ball_seen & ~out & (match.bspeed >= RESTART_SPEED)
        quiet = ~match.ball_seen | out | (match.bspeed < QUIET_SPEED)
    return played, quiet


def restart_candidates(played, quiet, start, stop):
    """Frames in [start, stop) where the ball goes from quiet to played, oldest first.

    The first frame of a run of played frames whose QUIET_LOOKBACK frames before are mostly
    quiet: a pass is preceded by play, a restart by a stoppage.
    """
    out = []
    for a, _ in runs(played, start, stop):
        before = quiet[max(0, a - QUIET_LOOKBACK):a]
        if len(before) and before.mean() >= QUIET_FRACTION:
            out.append(a)
    return out


# -- where a restart can be taken (Laws of the Game) --------------------------------------------

def restart_spot(match, frame):
    """'corner', 'goal_area', 'centre', 'touchline' or None: where the ball is on this frame."""
    if frame is None or not match.ball_seen[frame]:
        return None
    x, y = match.bx[frame], match.by[frame]
    from_goal_line = min(x, match.length - x)
    from_touchline = min(y, match.width - y)
    if from_goal_line <= CORNER_REACH + SPOT_MARGIN and from_touchline <= CORNER_REACH + SPOT_MARGIN:
        return "corner"
    if (from_goal_line <= GOAL_AREA[0] + SPOT_MARGIN
            and abs(y - match.width / 2.0) <= GOAL_AREA[1] + SPOT_MARGIN):
        return "goal_area"
    if np.hypot(x - match.length / 2.0, y - match.width / 2.0) <= CENTRE_REACH + SPOT_MARGIN:
        return "centre"
    if from_touchline <= THROW_IN_BAND + SPOT_MARGIN:
        return "touchline"
    return None


def crossed_goal_line(match, frame):
    """True if the ball, out on this frame, is beyond a goal line rather than a touchline."""
    beyond_x = max(-match.bx[frame], match.bx[frame] - match.length)
    beyond_y = max(-match.by[frame], match.by[frame] - match.width)
    return beyond_x > beyond_y


def legal_restart(match, frame, goal_line):
    """Could the restart after the ball crossed a goal line (or a touchline) be taken from here?"""
    allowed = ("corner", "goal_area", "centre") if goal_line else ("corner", "touchline")
    return restart_spot(match, frame) in allowed


# -- the five steps -----------------------------------------------------------------------------

def smooth(score):
    """Step 2: dead-ball mask for one half, from its score starting at the kickoff."""
    guard = np.zeros(len(score))
    guard[:KICKOFF_GUARD] = 1e6  # nobody stops play at the kickoff
    return two_state(np.maximum(score, 0.0), np.maximum(-score, 0.0) + guard,
                     SWITCH_COST, SWITCH_COST, start_cost=(0.0, 1e6)).astype(bool)


def latch(match, dead, played, quiet, out, first, stop):
    """Step 3: mark dead from every line crossing to its legal restart. [(crossing, kick)]."""
    crossings = []
    for a, b in runs(out, first, stop):
        if b - a < OUT_MIN_FRAMES:
            continue
        goal_line = crossed_goal_line(match, a)
        after = restart_candidates(played, quiet, b, min(stop, b + OUT_RESTART_REACH * match.fps))
        legal = [c for c in after if legal_restart(match, c, goal_line)]
        kick = legal[0] if legal else (after[0] if after else None)
        dead[a - first:(b if kick is None else kick) - first] = True
        crossings.append((a, kick))
    return crossings


def tidy(dead):
    """Step 4: absorb live gaps shorter than MIN_ALIVE, then drop stoppages shorter than MIN_DEAD."""
    for a, b in runs(~dead):
        if 0 < a and b < len(dead) and b - a < MIN_ALIVE:
            dead[a:b] = True
    for a, b in runs(dead):
        if b - a < MIN_DEAD:
            dead[a:b] = False


def find_restart(played, quiet, crossings, start, stop, half_stop):
    """Step 5 for the stoppage [start, stop): (kick or None, crossing or None, began at a line?).

    A stoppage that began with a line crossing ends at that crossing's restart; any other ends at
    the last quiet-to-played moment inside it (or just after its scored end). A restart followed by
    RESTART_TRUST more frames of dead score is not believed.
    """
    inside = [(a, kick) for a, kick in crossings if start <= a < stop]
    answered = [(kick, a) for a, kick in inside if kick is not None]
    candidates = [c for c in restart_candidates(played, quiet, start,
                                                min(half_stop, stop + RESTART_SEARCH_AFTER))
                  if c - start >= RESTART_NOT_BEFORE]
    if answered:
        kick, crossed = max(answered)
    else:
        kick = candidates[-1] if candidates else None
        crossed = inside[-1][0] if inside else None
    if kick is not None and stop - kick >= RESTART_TRUST:
        later = [c for c in candidates if c > kick and stop - c < RESTART_TRUST]
        kick = later[-1] if later else None
    return kick, crossed, bool(inside)


def segment_half(match, score, played, quiet, out, first, stop):
    """Steps 2-5 for one half, kickoff (first) to end (stop): (dead mask, [Stoppage])."""
    dead = smooth(score[first:stop])
    crossings = latch(match, dead, played, quiet, out, first, stop)
    tidy(dead)
    final = np.zeros_like(dead)
    stoppages = []
    for a, b in runs(dead):
        kick, crossed, at_line = find_restart(played, quiet, crossings, first + a, first + b, stop)
        if kick is None and not at_line and b - a < UNCONFIRMED_MIN:
            continue  # no line crossed, nobody restarted it, and short: slow play
        end = first + b if kick is None else min(stop, kick + RESTART_LAG)
        final[a:end - first] = True
        stoppages.append(Stoppage(first + a, end, kick, crossed))
    return final, stoppages


def detect(match, kickoffs):
    """(dead mask over every frame, [Stoppage]).

    kickoffs: {half: kickoff frame}. Graded frames before a kickoff are dead: the second half is
    graded from 110,808 but kicks off at 113,742 (FAQ.md).
    """
    score = dead_score(match)
    played, quiet = played_and_quiet(match)
    out = ball_out(match)
    dead = np.zeros(match.n, dtype=bool)
    stoppages = []
    for half, first, last in match.halves:
        kickoff = kickoffs.get(half, first)
        if kickoff > first:
            dead[first:kickoff] = True
            stoppages.append(Stoppage(first, kickoff, kickoff, None))
        dead[kickoff:last + 1], found = segment_half(match, score, played, quiet, out,
                                                     kickoff, last + 1)
        stoppages.extend(found)
    return dead, stoppages
