"""Dead ball: which graded frames are stoppages, and where each one starts and ends.

Role in the project
-------------------
First stage of the pipeline. run.py's dead_ball() calls detect(), and every frame marked dead is
labelled D; the team model then labels H / A only inside the live stretches between stoppages.

The six steps (per half, from its kickoff)
------------------------------------------
  1. score    A per-frame dead-ball score (log-odds) from the players' speed, the ball missing or
              out, few players tracked, and broadcast cuts while the ball is quiet.
  2. smooth   A two-state Viterbi turns the score into stoppages: a couple of seconds of evidence
              to open one, and again to close it.
  3. latch    A ball over a line keeps play dead until a restart from a spot the Laws allow.
  4. tidy     Very short live gaps are merged, very short stoppages dropped.
  5. restart  Each stoppage ends RESTART_LAG frames after its restart kick; a short one with no
              line crossing and no restart is slow play, and is dropped.
  6. goals    A stoppage that ends in a kickoff starts when the ball was last tracked (the goal).
"""

from collections import namedtuple

import numpy as np

from segments import rolling_mean, runs, two_state

# One detected stoppage: [start, stop) in video frames, and its restart kick (or None).
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
FEW_PLAYERS_WEIGHT = 1.0      # few players tracked
CUT_AWAY_WEIGHT = 1.54        # broadcast cut away while the ball is quiet
# 2. smooth
SWITCH_COST = 40.0            # evidence a stoppage needs to open, and again to close
# 3. latch
OUT_MIN_FRAMES = 3            # frames beyond a line before it counts as a crossing
OUT_RESTART_REACH = 60        # seconds a crossing waits for its restart
THROW_IN_BAND = 2.0           # metres inside the touchline
GOAL_AREA = (5.5, 9.16)       # depth and half-width, metres
CORNER_REACH = 1.0            # metres from the corner flag
CENTRE_REACH = 1.0            # metres from the centre spot (a goal, then a kickoff)
SPOT_MARGIN = 3.0             # metres of slack on each spot: a kick is detected a few frames late
# 4. tidy
MIN_ALIVE = 25                # frames; shorter live gaps between stoppages are absorbed
MIN_DEAD = 50                 # frames; shorter stoppages are dropped
# 5. restart
RESTART_SPEED = 5.0           # m/s; at least this fast inside the pitch, the ball is being played
QUIET_SPEED = 3.0             # m/s; slower than this (or untracked, or out), it is not
QUIET_LOOKBACK = 50           # frames before a restart that must be mostly quiet...
QUIET_FRACTION = 0.6          # ...this share of them
RESTART_NOT_BEFORE = 25       # frames into a stoppage before its restart can come
RESTART_SEARCH_AFTER = 25     # frames past a stoppage's scored end still searched for its restart
RESTART_TRUST = 500           # a restart followed by this many more frames of dead score is false
RESTART_BLIND = 100           # ...or by this many (4 s, the tracker's own gap limit) when, on
RESTART_BLIND_SHARE = 0.5     # most of them, no camera has the ball: it was put down, not played
RESTART_LAG = 10              # frames from the detected restart to the first live frame
UNCONFIRMED_MIN = 150         # frames; a shorter stoppage needs a line crossing or a restart
# 6. goals
KICKOFF_OWN_HALF = 0.9        # share of tracked players in their own half that makes a kickoff
OWN_HALF_SLACK = 1.0          # metres over the halfway line still counted as the own half
KICKOFF_LOOKBACK = 5          # frames before the restart on which the players are looked at


# -- signals ------------------------------------------------------------------------------------

def activity(match):
    """How fast the players move, the main stoppage signal (score step).

    Returns per frame the mean speed (m/s) of the ACTIVITY_TOP_K fastest players, 1 s centred mean.
    """
    speed = np.where(np.isnan(match.speed), -1.0, match.speed)
    fastest = np.sort(speed, axis=1)[:, -ACTIVITY_TOP_K:]
    counted = fastest >= 0
    n = counted.sum(axis=1)
    per_frame = np.where(n > 0, np.where(counted, fastest, 0.0).sum(axis=1) / np.maximum(n, 1),
                         np.nan)
    return rolling_mean(per_frame, ACTIVITY_HALF_WIDTH)


def ball_out(match):
    """Ball out of play (score and latch steps).

    Returns per frame True where the ball is tracked and more than OUT_MARGIN beyond a line.
    """
    with np.errstate(invalid="ignore"):
        return match.ball_seen & ((match.bx < -OUT_MARGIN) | (match.bx > match.length + OUT_MARGIN)
                                  | (match.by < -OUT_MARGIN) | (match.by > match.width + OUT_MARGIN))


def dead_score(match):
    """Combine the tracking signals into one dead-ball score (score step).

    Returns per frame the log-odds that the ball is dead (positive: stoppage). detect() then adds
    the broadcast cuts.
    """
    score = np.clip((ACTIVITY_MID - np.nan_to_num(activity(match), nan=0.0)) / ACTIVITY_SCALE,
                    -ACTIVITY_CLIP, ACTIVITY_CLIP)
    tracked = (~np.isnan(match.px)).sum(axis=1)
    return (score + MISSING_WEIGHT * ~match.ball_seen + OUT_WEIGHT * ball_out(match)
            + FEW_PLAYERS_WEIGHT * (tracked < FEW_PLAYERS))


def played_and_quiet(match):
    """Two flags on the ball: to find restart kicks (latch, restart) and to gate the cuts (score).

    Returns (played, quiet) per frame. played: tracked, inside the pitch and at RESTART_SPEED or
    faster, so just kicked. quiet: untracked, out, or slower than QUIET_SPEED, so not in play. A
    restart kick is a played frame after mostly quiet ones.
    """
    out = ball_out(match)
    with np.errstate(invalid="ignore"):
        played = match.ball_seen & ~out & (match.bspeed >= RESTART_SPEED)
        quiet = ~match.ball_seen | out | (match.bspeed < QUIET_SPEED)
    return played, quiet


def restart_candidates(played, quiet, start, stop):
    """Candidate restart kicks (latch and restart steps).

    Returns the frames in [start, stop) where the ball goes from quiet to played, oldest first: a
    pass is preceded by play, a restart by a stoppage.
    """
    out = []
    for a, _ in runs(played, start, stop):
        before = quiet[max(0, a - QUIET_LOOKBACK):a]
        if len(before) and before.mean() >= QUIET_FRACTION:
            out.append(a)
    return out


# -- where a restart can be taken (Laws of the Game) --------------------------------------------

def restart_spot(match, frame):
    """Which restart spot the ball is on (latch step).

    Returns 'corner', 'goal_area', 'centre', 'touchline' or None for this frame.
    """
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
    """Which line the ball went over, which decides its legal restarts (latch step).

    Returns True if the ball, out on this frame, is beyond a goal line rather than a touchline.
    """
    beyond_x = max(-match.bx[frame], match.bx[frame] - match.length)
    beyond_y = max(-match.by[frame], match.by[frame] - match.width)
    return beyond_x > beyond_y


def legal_restart(match, frame, goal_line):
    """Check a candidate kick against the Laws of the Game (latch step).

    Returns True if the restart after a goal-line (or touchline) crossing can be taken from where
    the ball is on this frame.
    """
    allowed = ("corner", "goal_area", "centre") if goal_line else ("corner", "touchline")
    return restart_spot(match, frame) in allowed


# -- steps 2 to 6 -------------------------------------------------------------------------------

def smooth(score):
    """Turn one half's score into stoppages (smooth step).

    Returns the half's dead-ball mask (True = dead), from its kickoff.
    """
    # State 1 is dead. Being live costs the positive part of the score and being dead the negative
    # part, so the decoder follows the score's sign except where a switch (SWITCH_COST each way)
    # would not pay for itself. start_cost: a half begins live, at its kickoff.
    return two_state(np.maximum(score, 0.0), np.maximum(-score, 0.0),
                     SWITCH_COST, SWITCH_COST, start_cost=(0.0, 1e6)).astype(bool)


def latch(match, dead, played, quiet, out, first, stop):
    """Mark dead from every line crossing to its legal restart, in `dead` itself (latch step).

    Returns [(crossing frame, restart kick or None)], for find_restart.
    """
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
    """Merge very short live gaps, then drop very short stoppages (tidy step).

    Edits `dead` itself (thresholds MIN_ALIVE and MIN_DEAD) and returns nothing.
    """
    for a, b in runs(~dead):
        if 0 < a and b < len(dead) and b - a < MIN_ALIVE:
            dead[a:b] = True
    for a, b in runs(dead):
        if b - a < MIN_DEAD:
            dead[a:b] = False


def find_restart(played, quiet, crossings, start, stop, half_stop, blind=None):
    """Find the restart kick that ends the stoppage [start, stop) (restart step).

    Returns (kick or None, crossing frame or None, whether it began at a line). The kick is the
    crossing's legal restart if there is one, else the last quiet-to-played moment. It is rejected
    if the dead score then goes on for RESTART_TRUST frames, or for RESTART_BLIND frames with no
    camera on the ball.
    """
    def false(c):
        left = stop - c
        return left >= RESTART_TRUST or (blind is not None and left >= RESTART_BLIND
                                         and blind[c:stop].mean() > RESTART_BLIND_SHARE)

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
    if kick is not None and false(kick):
        later = [c for c in candidates if c > kick and not false(c)]
        kick = later[-1] if later else None
    return kick, crossed, bool(inside)


def is_kickoff(match, frame):
    """Tell a kickoff from other restarts (goals step).

    Returns True if at least KICKOFF_OWN_HALF of the tracked players stand in their own half on
    this frame (Law 8).
    """
    x, team = match.px[frame], match.team_at[frame]
    seen = ~np.isnan(x)
    if not seen.any():
        return False
    middle = match.length / 2.0
    with np.errstate(invalid="ignore"):
        left, right = x <= middle + OWN_HALF_SLACK, x >= middle - OWN_HALF_SLACK
    home_left = match.attacks_right[int(match.half[frame])]  # home defends the x = 0 goal
    home_own, away_own = (left, right) if home_left else (right, left)
    own = np.where(team == 0, home_own, away_own)
    return own[seen].mean() >= KICKOFF_OWN_HALF


def from_the_goal(match, final, stoppages, first, stop):
    """Move the start of each stoppage that ends in a kickoff back to the goal (goals step).

    Marks those frames dead in `final` and returns the stoppages with their new starts.
    """
    out = []
    for s in stoppages:
        restart = s.stop if s.kick is None else s.kick
        look = min(max(restart - KICKOFF_LOOKBACK, first), stop - 1)
        start = s.start
        if start > first and is_kickoff(match, look):
            while start > first and not match.ball_seen[start - 1] and not final[start - 1 - first]:
                start -= 1
            final[start - first:s.start - first] = True
        out.append(s._replace(start=start))
    return out


def segment_half(match, score, played, quiet, out, first, stop, blind=None):
    """Run steps 2-6 on one half, from its kickoff (first) to its end (stop).

    Returns (the half's dead-ball mask, [Stoppage]).
    """
    dead = smooth(score[first:stop])
    crossings = latch(match, dead, played, quiet, out, first, stop)
    tidy(dead)
    final = np.zeros_like(dead)
    stoppages = []
    for a, b in runs(dead):
        kick, crossed, at_line = find_restart(played, quiet, crossings, first + a, first + b, stop,
                                              blind)
        if kick is None and not at_line and b - a < UNCONFIRMED_MIN:
            continue  # no line crossed, nobody restarted it, and short: slow play
        end = first + b if kick is None else min(stop, kick + RESTART_LAG)
        final[a:end - first] = True
        stoppages.append(Stoppage(first + a, end, kick, crossed))
    return final, from_the_goal(match, final, stoppages, first, stop)


def detect(match, kickoffs, cut=None, blind=None):
    """Entry point, called by run.py's dead_ball(): steps 1-6 over the whole match.

    Returns (dead-ball mask over every frame, [Stoppage]).
      kickoffs  {half: kickoff frame}; graded frames before a kickoff are dead
      cut       per frame, the broadcast is cut away; it only counts while the ball is quiet
      blind     per frame, no camera has the ball; used to reject false restarts
    """
    score = dead_score(match)
    played, quiet = played_and_quiet(match)
    if cut is not None:
        score = score + CUT_AWAY_WEIGHT * (cut & quiet)
    out = ball_out(match)
    dead = np.zeros(match.n, dtype=bool)
    stoppages = []
    for half, first, last in match.halves:
        kickoff = kickoffs.get(half, first)
        if kickoff > first:
            dead[first:kickoff] = True
            stoppages.append(Stoppage(first, kickoff, kickoff, None))
        dead[kickoff:last + 1], found = segment_half(match, score, played, quiet, out,
                                                     kickoff, last + 1, blind)
        stoppages.extend(found)
    return dead, stoppages
