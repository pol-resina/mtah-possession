"""Stand-in for bench/models.py of Adrià's workspace, which holds his experiment models and is not
needed to train the team model. features.py only uses _pol_paths, to import team_per_frame from the
solution's tracking.py (unchanged since the model was trained)."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pol_paths():
    for path in (REPO, os.path.join(REPO, "solution")):
        if path not in sys.path:
            sys.path.insert(0, path)
