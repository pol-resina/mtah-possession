"""Small shared tools: run-length helpers, a centred rolling mean, and a two-state Viterbi.

WHAT IT IS FOR
    Both stages (live/dead in deadball.py, home/away in run.py) are noisy per-frame evidence for one
    of two states plus a strong prior that the state changes rarely. This is the shared machinery.

THE IMPORTANT PARTS
    - two_state() picks the labelling that minimises

          sum of per-frame costs  +  a fixed penalty for every change of state

      so a change happens only when the evidence outweighs the penalty. Costs are in nats (the unit
      of information for the natural logarithm; a probability p costs -log p). Ties go to the later
      change.
    - Spans from runs() and label_runs() are half-open, [a, b); graded halves elsewhere end on an
      inclusive last frame.
"""

import numpy as np


def runs(mask, start=0, stop=None):
    """[(a, b)] half-open spans where mask is True, inside [start, stop)."""
    stop = len(mask) if stop is None else stop
    window = np.asarray(mask[start:stop], dtype=np.int8)
    # over the window, each frame is given a value 0 (dead) or 1 (~dead), in this binary vector:
    # [diff = v{i} - v{i-1}]
    #  1. diff == +1 is a ~dead span start
    #  2. diff == -1 is a ~dead span end (exclusive). The zero padding on both
    # 0-padding marks the window's edges -> ensures zip matches lengths
    edges = np.diff(np.concatenate(([0], window, [0])))
    return [(start + int(a), start + int(b))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def label_runs(labels, start, stop):
    """[(a, b, label)] maximal runs of one label inside [start, stop)."""
    window = np.asarray(labels[start:stop])
    bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
    return [(start + int(a), start + int(b), window[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def rolling_mean(values, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window is all NaN."""
    values = np.asarray(values, dtype=np.float64)
    seen = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0)))) # O(n)
    count = np.concatenate(([0], np.cumsum(seen)))                          # O(n)
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)
                                                      # divisor is the number of non-NaN samples in the window


def two_state(cost0, cost1, switch01, switch10, start_cost=(0.0, 0.0)):
    """Minimum-cost 0/1 labelling of a sequence (int8 array).

    cost0[t], cost1[t]: cost of being in state 0 or 1 on frame t. switch01, switch10: penalty paid
    on each 0 -> 1 and 1 -> 0 change. start_cost: extra cost of each state on the first frame
    (optional; deadball.py uses it to force the first frame into state 0).

    Ties go to the later change: across frames with no evidence, a change lands on the first frame
    of evidence for the new state -- an intercepted pass stays with the passing side until the
    interceptor has the ball.
    """
    n = len(cost0)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    c0, c1 = list(map(float, cost0)), list(map(float, cost1))
    # a0 / a1: cheapest total cost of any labelling of frames 0..t that ends in state 0 / 1 on t.
    a0, a1 = start_cost[0] + c0[0], start_cost[1] + c1[0]
    from1 = bytearray(n) # from1[t] = 1 -> cheapest path into state 0 at t came from state 1 at t-1 (a switch)
    from0 = bytearray(n)
    for t in range(1, n):
        # cost of arriving in state 0 / 1 on t by switching from the other state.
        switch_in0 = a1 + switch10
        switch_in1 = a0 + switch01
        if switch_in0 <= a0:  # <= : on a tie, take the change (moves it to the latest frame)
            new0, from1[t] = switch_in0, 1
        else:
            new0 = a0
        if switch_in1 <= a1:
            new1, from0[t] = switch_in1, 1
        else:
            new1 = a1
        a0, a1 = new0 + c0[t], new1 + c1[t]
    # backtrack from the cheaper end state (ties pick state 0), following the backpointers (from0/from1).
    state = 0 if a0 <= a1 else 1
    for t in range(n - 1, -1, -1):
        out[t] = state
        if state == 0 and from1[t]:
            state = 1
        elif state == 1 and from0[t]:
            state = 0
    return out
