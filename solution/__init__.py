"""Ball possession from tracking alone: dead ball first (deadball.py), then possession
(possession.py) from ball contact seen by the tactical camera and, where it is silent, by the
broadcast camera's own tracking (broadcast.py), all decoded with the switch-cost Viterbi in
segments.py. run.py writes the submission; diagnostics.py checks it without tuning against the
public window."""
