"""Possession inside the live stretches: whose ball it is, and when it changes hands.

  evidence   a frame counts only when the ball is within CONTROL_RADIUS of a player of exactly one
             team (a "clear contact"); a duel, or a ball in flight, says nothing. Where the
             tactical camera sees no clear contact, the broadcast camera's (broadcast.py) counts
             BROADCAST_WEIGHT of a frame.
  decoding   a two-state Viterbi per live stretch: a change of possession has to outweigh
             SWITCH_COST frames of contact, so single touches and deflections are ignored, and
             between contacts the ball stays with the last team that had it.
  timing     every change is moved CONTROL_LAG frames later: the ground truth changes when the new
             team has control, not at its first touch.
  restarts   the team restarting play opens each live stretch: by the Laws of the Game at a corner
             or a goal kick, otherwise the player standing at the resting ball, otherwise the side
             that did not touch the ball last before it went out.
"""

import numpy as np

import deadball
from segments import runs, two_state

# Params
CONTROL_RADIUS = 1.5     # metres: contact range, allowing for ~0.5 m of tracking noise
SWITCH_COST = 8.0        # frames of clear contact a change of possession must outweigh
CONTROL_LAG = 20         # frames from the new team's first contact to the labelled change
BROADCAST_WEIGHT = 0.72  # a broadcast contact frame, relative to a tactical one
KICKER_RADIUS = 2.0      # metres around the resting ball in which the restart taker stands
KICKER_WINDOW = 50       # frames before a restart in which to look for the taker
LAST_TOUCH_REACH = 75    # frames before a line crossing in which to look for the last touch


def team_distances(match):
    """(nearest home player, nearest away player) distance to the ball on every frame, inf if none."""
    dist = np.hypot(match.px - match.bx[:, None], match.py - match.by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = match.team_at == 0
    return (np.where(home, dist, np.inf).min(axis=1),
            np.where(home, np.inf, dist).min(axis=1))


def contact(d_home, d_away, radius=None):
    """Per frame: 0 home has clear contact, 1 away has, -1 nobody clearly has."""
    radius = CONTROL_RADIUS if radius is None else radius
    out = np.full(len(d_home), -1, dtype=np.int8)
    out[(d_home < radius) & (d_away >= radius)] = 0
    out[(d_away < radius) & (d_home >= radius)] = 1
    return out


# -- who restarts play -----------------------------------------------------------------------------

def end_teams(match, frame):
    """(side attacking the goal nearer the ball, side defending it) on this frame."""
    home_attacks_right = match.attacks_right[match.half[frame]]
    attacking = 0 if home_attacks_right == (match.bx[frame] > match.length / 2.0) else 1
    return attacking, 1 - attacking


def kicker_team(match, d_home, d_away, kick):
    """Team of the player standing at the ball while it rests before a restart, or None.

    Only frames with the ball tracked and slower than deadball.QUIET_SPEED count: once it moves,
    the players near it are the ones it is passing, not the one who kicked it.
    """
    lo = max(0, kick - KICKER_WINDOW)
    with np.errstate(invalid="ignore"):
        resting = match.ball_seen[lo:kick] & (match.bspeed[lo:kick] < deadball.QUIET_SPEED)
    near = contact(d_home[lo:kick], d_away[lo:kick], KICKER_RADIUS)[resting]
    home, away = int(np.sum(near == 0)), int(np.sum(near == 1))
    if home == away:
        return None
    return 0 if home > away else 1


def last_touch(evidence, frame):
    """Team of the last clear contact in the LAST_TOUCH_REACH frames before frame, or None."""
    window = evidence[max(0, frame - LAST_TOUCH_REACH):frame]
    touched = np.flatnonzero(window >= 0)
    return int(window[touched[-1]]) if len(touched) else None


def opening_team(match, d_home, d_away, evidence, stoppage):
    """The side restarting play after a stoppage, or None if nothing says."""
    kick = stoppage.kick
    spot = deadball.restart_spot(match, kick)
    if spot == "corner":
        return end_teams(match, kick)[0]
    if spot == "goal_area":
        return end_teams(match, kick)[1]
    team = kicker_team(match, d_home, d_away, kick) if kick is not None else None
    if team is None and stoppage.crossed is not None:
        last = last_touch(evidence, stoppage.crossed)
        team = None if last is None else 1 - last
    return team


# -- decoding ----------------------------------------------------------------------------------

def contact_costs(evidence, second=None):
    """(cost of saying home, cost of saying away) per frame: each contact of the other team.

    second: the broadcast camera's contact evidence on the same frames, used only where the
    tactical camera's is silent.
    """
    cost_home = (evidence == 1).astype(np.float64)
    cost_away = (evidence == 0).astype(np.float64)
    if second is not None:
        silent = evidence < 0
        cost_home += BROADCAST_WEIGHT * (silent & (second == 1))
        cost_away += BROADCAST_WEIGHT * (silent & (second == 0))
    return cost_home, cost_away


def decode_live(cost_home, cost_away, opening=None):
    """0/1 team per frame of one live stretch, every change moved CONTROL_LAG frames later.

    opening: the team restarting play, if known; the other team then needs one change to take over.
    """
    start = (0.0, 0.0)
    if opening is not None:
        start = (0.0, SWITCH_COST) if opening == 0 else (SWITCH_COST, 0.0)
    state = two_state(cost_home, cost_away, SWITCH_COST, SWITCH_COST, start_cost=start)
    if CONTROL_LAG and len(state) > CONTROL_LAG:
        state = np.concatenate((np.repeat(state[:1], CONTROL_LAG), state[:-CONTROL_LAG]))
    return state


def label(match, dead, stoppages, second=None):
    """Letters H/A/D for every graded frame, X elsewhere. second: broadcast contact, or None."""
    d_home, d_away = team_distances(match)
    evidence = contact(d_home, d_away)
    openings = {s.stop: opening_team(match, d_home, d_away, evidence, s) for s in stoppages}
    cost_home, cost_away = contact_costs(evidence, second)

    out = np.full(match.n, "X", dtype="<U1")
    for _, first, last in match.halves:
        out[first:last + 1] = "D"
        for a, b in runs(~dead, first, last + 1):
            state = decode_live(cost_home[a:b], cost_away[a:b], openings.get(a))
            out[a:b] = np.where(state == 0, "H", "A")
    return out
