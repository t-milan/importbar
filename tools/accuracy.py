"""Measure how well importbar predicts *your* imports.

    python tools/accuracy.py --runs 8 pandas scipy.stats sklearn.ensemble

Each run is a fresh interpreter. The first few runs only train the model, then
each measured run reports how far the predicted total time was from the actual
total, sampled across the run (ignoring the first 10%), and whether the bar
ever went backwards. Uses a throwaway cache so your real history is untouched.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

CHILD = r"""
import importlib, json, sys, time
t0 = time.perf_counter()
import importbar
mods = sys.argv[1:]
rows = []
class Rec:
    def __call__(self, p):
        e = sess.estimate()
        rows.append([p.elapsed, p.fraction, e.fraction, e.remaining])
sess = importbar.Session("accuracy", roots=[m.partition(".")[0] for m in mods],
                         renderer=Rec(), delay=0.0, fps=50)
with sess:
    for m in mods:
        importlib.import_module(m)
print(json.dumps({"actual": time.perf_counter() - t0, "rows": rows}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--train", type=int, default=3)
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, IMPORTBAR_CACHE_DIR=cache,
                   PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))

        def run():
            out = subprocess.run([sys.executable, "-c", CHILD, *args.modules], env=env,
                                 capture_output=True, text=True, check=True)
            return json.loads(out.stdout.strip().splitlines()[-1])

        for i in range(args.train):
            print("training run %d: %.2fs" % (i + 1, run()["actual"]))
        for i in range(args.runs):
            r = run()
            a = r["actual"]
            live = [x for x in r["rows"] if x[2] is not None and x[0] > 0.1 * a]
            errs = [abs(x[0] + x[3] - a) / a for x in live] or [float("nan")]
            shown = [x[1] for x in r["rows"]]
            mono = all(b >= c for c, b in zip(shown, shown[1:]))
            print("run %d: %.2fs  predicted-total error mean %4.1f%%  max %4.1f%%  monotonic=%s"
                  % (i + 1, a, 100 * statistics.mean(errs), 100 * max(errs), mono))


if __name__ == "__main__":
    main()
