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
freak run from wrecking the estimate.

Every timing is kept twice, once per *mode*: ``warm`` (the files are in the
OS file cache) and ``cold`` (first start after a reboot, or after the cache
was evicted: the same imports wait on the disk and the virus scanner). The two
differ by 1.2x on a fast SSD and 10x or more on a laptop hard disk, and not
uniformly across packages, so they are learned separately.
"""

import marshal
import math
import os
import sys
import time

FORMAT = 2
MODES = ("warm", "cold")
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


def ratio_update(st, x):
    """Linear EWMA of a value in 0..1 (the CPU/wall ratio of a run)."""
    if not st:
        return [x, 0.0, 1]
    mean, var, n = st
    a = max(ALPHA, 1.0 / (n + 1))
    d = x - mean
    return [mean + a * d, (1 - a) * (var + a * d * d), n + 1]


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


def bundle_dir():
    """Folder holding a frozen app's bundled code (PyInstaller & co.), else None."""
    if not getattr(sys, "frozen", False):
        return None
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.executable))


def env_key():
    if getattr(sys, "frozen", False):
        # A frozen app's sys.prefix is where its files were unpacked: for a
        # PyInstaller one-file build, a new temp folder on every launch. The
        # executable's own path is what stays the same from run to run.
        where = "frozen:" + os.path.abspath(sys.executable)
    else:
        where = sys.prefix
    s = "%s|%s|%d.%d" % (where, sys.implementation.name, *sys.version_info[:2])
    h = 0xCBF29CE484222325  # FNV-1a: stable across runs, no hashlib import
    for b in s.encode("utf-8", "surrogateescape"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def default_path():
    return os.path.join(cache_dir(), "timings-%s.bin" % env_key())


def boot_time():
    """Wall-clock time the machine booted, or None. Used to spot cold starts.

    Uses a clock that keeps counting through sleep, so waking a laptop isn't
    mistaken for a reboot (on Windows that holds for GetTickCount64, which
    Python < 3.13 uses; newer versions may drift across sleep, which only makes
    the first estimate after waking more cautious).
    """
    try:
        if sys.platform.startswith("linux"):
            up = time.clock_gettime(time.CLOCK_BOOTTIME)
        elif sys.platform == "darwin":
            up = time.clock_gettime(time.CLOCK_MONOTONIC)   # includes sleep on macOS
        else:
            up = time.monotonic()
        return time.time() - up
    except (AttributeError, OSError, ValueError):
        return None


# --- store -------------------------------------------------------------------

class Store:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.times = {m: {} for m in MODES}  # mode -> pkg -> [self-time stat, eager stat]
        self.edges = {}      # parent -> {child: misses}
        self.sessions = {}   # label -> {"roots": [...], "other": {mode: stat}, "total": {mode: stat}}
        self.cpu = {}        # mode -> ratio stat: importing thread's CPU time / wall time
        self.priors = {}     # context -> [cold runs, runs]   (how often each context was cold)
        self.last = {}       # {"boot": ..., "end": ...} of the most recent session

    @classmethod
    def load(cls, path=None):
        st = cls(path)
        try:
            with open(st.path, "rb") as f:
                data = marshal.load(f)
            if isinstance(data, dict):
                fmt = data.get("format")
                if fmt == FORMAT:
                    st.times = data.get("times") or st.times
                    for m in MODES:
                        st.times.setdefault(m, {})
                    st.edges = data.get("edges", {})
                    st.sessions = data.get("sessions", {})
                    st.cpu = data.get("cpu", {})
                    st.priors = data.get("priors", {})
                    st.last = data.get("last", {})
                elif fmt == 1:
                    st._migrate_v1(data)
        except (OSError, ValueError, EOFError, TypeError):
            pass  # missing or corrupt: start fresh
        return st

    def _migrate_v1(self, data):
        # Version 1 didn't separate cold and warm runs; most runs are warm.
        eager = data.get("eager", {})
        self.times["warm"] = {p: [st, eager.get(p) or st]
                              for p, st in data.get("packages", {}).items()}
        self.edges = data.get("edges", {})
        for label, s in data.get("sessions", {}).items():
            self.sessions[label] = {
                "roots": s.get("roots", []),
                "other": {"warm": s["other"]} if s.get("other") else {},
                "total": {"warm": s["total"]} if s.get("total") else {},
            }

    def save(self):
        data = {
            "format": FORMAT,
            "python": sys.version.split()[0],
            "prefix": sys.prefix,
            "times": self.times,
            "edges": self.edges,
            "sessions": self.sessions,
            "cpu": self.cpu,
            "priors": self.priors,
            "last": self.last,
        }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = "%s.%d.tmp" % (self.path, os.getpid())
            with open(tmp, "wb") as f:
                marshal.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass  # read-only home, full disk...: the bar still works next time, just unlearned

    def expected(self, pkg, mode="warm"):
        """(total self time, part of it spent while the package's __init__ ran), or None."""
        e = self.times[mode].get(pkg)
        if not e:
            return None
        total = stat_mean(e[0])
        return total, min(stat_mean(e[1]), total)

    def known(self, pkg):
        return any(pkg in self.times[m] for m in MODES)
