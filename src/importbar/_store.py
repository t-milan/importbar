"""Persistent timing history.

One small file per Python environment (keyed by ``sys.prefix`` and the
interpreter version), written atomically after a session finishes. It uses
``marshal`` rather than ``json``: marshal is built into the interpreter, whereas
``import json`` drags in ``re`` and ``enum`` (~7 ms), and this library should
cost next to nothing to import. ``python -m importbar report`` shows the
contents in readable form.

Durations are modelled as log-normal: each package keeps an exponentially
weighted mean and variance of log(seconds). That adapts when the machine or a
package version changes, and clipping each new observation to ±3σ keeps one
freak run (antivirus scan, cold disk) from wrecking the estimate.
"""

import marshal
import math
import os
import sys

FORMAT = 1
ALPHA = 0.3          # EWMA weight of the newest run (floored by 1/n for early runs)
PRIOR_VAR = 0.25     # log-variance of a single observation (~ ±50%)
EPS = 1e-5           # seconds; avoid log(0)
MAX_MISSES = 3       # drop a dependency edge after this many unexplained absences


# --- stats -----------------------------------------------------------------

def stat_update(st, seconds):
    x = math.log(max(seconds, EPS))
    if not st:
        return [x, PRIOR_VAR, 1]
    mu, var, n = st
    sd = math.sqrt(var) + 0.05
    x = min(max(x, mu - 3 * sd), mu + 3 * sd)
    a = max(ALPHA, 1.0 / (n + 1))
    d = x - mu
    mu += a * d
    var = (1 - a) * (var + a * d * d)
    return [mu, var, n + 1]


def stat_mean(st):
    """Mean of the log-normal (slightly above the median: mild pessimism)."""
    mu, var, _ = st
    return math.exp(mu + var / 2)


# --- location ----------------------------------------------------------------

def cache_dir():
    d = os.environ.get("IMPORTBAR_CACHE_DIR")
    if d:
        return d
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "importbar")


def env_key():
    s = "%s|%s|%d.%d" % (sys.prefix, sys.implementation.name, *sys.version_info[:2])
    h = 0xCBF29CE484222325  # FNV-1a: stable across runs, no hashlib import
    for b in s.encode("utf-8", "surrogateescape"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def default_path():
    return os.path.join(cache_dir(), "timings-%s.bin" % env_key())


# --- store -------------------------------------------------------------------

class Store:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.packages = {}   # name -> [mu, var, n]   (self time of the package)
        self.eager = {}      # name -> stat of self time spent while its __init__ ran
        self.edges = {}      # parent -> {child: misses}
        self.sessions = {}   # label -> {"roots": [...], "other": stat, "total": stat}

    @classmethod
    def load(cls, path=None):
        st = cls(path)
        try:
            with open(st.path, "rb") as f:
                data = marshal.load(f)
            if isinstance(data, dict) and data.get("format") == FORMAT:
                st.packages = data.get("packages", {})
                st.eager = data.get("eager", {})
                st.edges = data.get("edges", {})
                st.sessions = data.get("sessions", {})
        except (OSError, ValueError, EOFError, TypeError):
            pass  # missing or corrupt: start fresh
        return st

    def save(self):
        data = {
            "format": FORMAT,
            "python": sys.version.split()[0],
            "prefix": sys.prefix,
            "packages": self.packages,
            "eager": self.eager,
            "edges": self.edges,
            "sessions": self.sessions,
        }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = "%s.%d.tmp" % (self.path, os.getpid())
            with open(tmp, "wb") as f:
                marshal.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass  # read-only home, full disk...: the bar still works next time, just unlearned

    def expected(self, pkg):
        """(total self time, part of it spent while the package's __init__ ran)."""
        st = self.packages.get(pkg)
        if not st:
            return None
        total = stat_mean(st)
        ea = self.eager.get(pkg)
        return total, min(stat_mean(ea), total) if ea else total
