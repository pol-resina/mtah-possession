"""Download the open DFL dataset the bench is built on.

    python solution/training/download.py            # 21 files, 2.6 GB, into cache/dfl/raw/ (git-ignored)

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data. Figshare article 28196177, CC BY 4.0: seven matches of
the German Bundesliga (1st and 2nd division), with tracking at 25 Hz, and per frame which team has
the ball and whether it is in play. Nothing from it is committed to this repository.
"""

import json
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "cache", "dfl", "raw")
ARTICLE = "https://api.figshare.com/v2/articles/28196177"
AGENT = {"User-Agent": "dfl-open-data-downloader"}


def fetch(url, path, size):
    if os.path.isfile(path) and os.path.getsize(path) == size:
        return False
    request = urllib.request.Request(url, headers=AGENT)
    with urllib.request.urlopen(request) as response, open(path + ".part", "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    os.replace(path + ".part", path)
    return True


if __name__ == '__main__':
    os.makedirs(RAW, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(ARTICLE, headers=AGENT)) as response:
        files = json.load(response)["files"]
    for item in sorted(files, key=lambda f: f["size"]):
        path = os.path.join(RAW, os.path.basename(item["name"]))
        new = fetch(item["download_url"], path, item["size"])
        print("{} {:8.1f} MB  {}".format("got " if new else "have", item["size"] / 1e6, item["name"]))
        sys.stdout.flush()
