"""What to expect from a session, and whether this start is cold.

Expected timings come from, in order of trust:

1. this machine's own history for that mode (warm or cold);
2. this machine's history for the *other* mode, times the profile's cold/warm
   ratio for that package (the author measured both, this machine one);
3. the shared profile, times how fast this machine is compared with the
   author's (worked out from packages both have measured);
4. the other mode times a global cold/warm ratio.

Each source comes with a ``tau``, the log-sd of how far off it might be,
which sets how much a mismatch counts as evidence when telling cold from warm.

Cold or warm? The program can't ask the OS whether its files are cached, so it
combines three clues into one probability:

* **Before the run**, how often starts like this one were cold: after a
  reboot (the file cache is empty), after hours of inactivity (it may have
  been evicted), or the first run ever (fresh install, virus scanner). Learned
  per machine, starting from sensible defaults.
* **During the run**, the importing thread's CPU time divided by wall time. A
  warm import is CPU-bound, so the ratio is close to 1 on any machine, fast or
  slow. A cold one waits for the disk and the virus scanner, which costs wall
  time but no CPU time. This is the clue that tells "slow machine" from "cold
  start", and it is nearly free to measure.
* **During the run**, which set of expectations the finished packages match
  better after allowing for machine speed.
"""

import math
import sys

from ._store import MODES, stat_mean

TAU_LOCAL = 0.25     # this machine, this mode
TAU_SCALED = 0.4     # this machine's other mode x the profile's per-package ratio
TAU_PROFILE = 0.6    # the author's machine, rescaled
TAU_GUESS = 0.8      # other mode x a global cold/warm ratio
DEFAULT_COLD_RATIO = 2.0
NOISE_PKG = 0.09     # log-variance of one package's time around its expectation

# How often each kind of start is cold, before this machine has taught us.
PRIOR_COLD = {"first": 0.5, "boot": 0.85, "recent": 0.05, "hours": 0.2, "days": 0.4}
PRIOR_WEIGHT = 2.0   # pseudo-runs behind those defaults

# CPU/wall ratio of the importing thread: (mean, sd) per mode by default.
CPU_DEFAULT = {"warm": (0.95, 0.06), "cold": (0.6, 0.2)}
# Resolution of the thread CPU clock: ~15.6 ms ticks on Windows, fine elsewhere.
CPU_CLOCK_RES = 0.016 if sys.platform == "win32" else 0.002


def _other(mode):
    return "cold" if mode == "warm" else "warm"


class Model:
    def __init__(self, store, profile=None):
        self.store = store
        self.profile = profile
        self.machine = self._machine_ratio()
        self.cold_ratio = self._cold_ratio()

    # ------------------------------------------------------------ calibration

    def _machine_ratio(self):
        """How much slower this machine is than the profile's, per mode."""
        out = {m: None for m in MODES}
        if self.profile is None:
            return {m: 1.0 for m in MODES}
        for m in MODES:
            loc = sh = 0.0
            for p, (total, _) in self.profile.times[m].items():
                e = self.store.expected(p, m)
                if e:
                    loc += e[0]
                    sh += total
            if sh >= 0.05:
                out[m] = min(max(loc / sh, 0.1), 10.0)
        # Missing one mode? CPU speed carries over; disk speed may not, but
        # the live evidence corrects for that within the first packages.
        w, c = out["warm"], out["cold"]
        return {"warm": w or c or 1.0, "cold": c or w or 1.0}

    def _cold_ratio(self):
        """Typical cold/warm slowdown, for packages measured in one mode only."""
        def ratio(times):
            w = c = 0.0
            for p, tw in times["warm"].items():
                tc = times["cold"].get(p)
                if tc is not None:
                    w += tw
                    c += tc
            return c / w if w >= 0.05 else None

        local = {m: {p: stat_mean(e[0]) for p, e in self.store.times[m].items()} for m in MODES}
        r = ratio(local)
        if r is None and self.profile is not None:
            r = ratio({m: {p: e[0] for p, e in self.profile.times[m].items()} for m in MODES})
        return min(max(r or DEFAULT_COLD_RATIO, 1.0), 20.0)

    # ------------------------------------------------------------ expectations

    def expect(self, pkg):
        """{mode: (total, eager, tau, local?)} for a package, or None if nothing is known."""
        st, prof = self.store, self.profile
        loc = {m: st.expected(pkg, m) for m in MODES}
        sh = {m: (prof.times[m].get(pkg) if prof else None) for m in MODES}
        out = {}
        for m in MODES:
            o = _other(m)
            if loc[m]:
                out[m] = (loc[m][0], loc[m][1], TAU_LOCAL, True)
            elif loc[o] and sh[m] and sh[o] and sh[o][0] > 0:
                r = sh[m][0] / sh[o][0]
                re = sh[m][1] / sh[o][1] if sh[o][1] > 0 else r
                out[m] = (loc[o][0] * r, min(loc[o][1] * re, loc[o][0] * r), TAU_SCALED, False)
            elif sh[m]:
                k = self.machine[m]
                out[m] = (sh[m][0] * k, sh[m][1] * k, TAU_PROFILE, False)
        for m in MODES:
            o = _other(m)
            if m not in out and o in out:
                f = self.cold_ratio if m == "cold" else 1.0 / self.cold_ratio
                t, e = out[o][0], out[o][1]
                out[m] = (t * f, e * f, TAU_GUESS, False)
        return out or None

    def deps(self, pkg):
        if self.store.known(pkg) or self.profile is None:
            return self.store.edges.get(pkg, ())
        return self.profile.edges.get(pkg, ())

    def session(self, label, portable):
        """(roots or None, {mode: expected non-import time}, known?) for a session."""
        loc = self.store.sessions.get(label)
        sh = self.profile.sessions.get(portable) if (self.profile and portable) else None
        if loc is None and sh is None:
            return None, {m: 0.0 for m in MODES}, False
        roots = loc["roots"] if loc else sh["roots"]
        other = {}
        for m in MODES:
            if loc and loc.get("other", {}).get(m):
                other[m] = stat_mean(loc["other"][m])
            elif sh and m in sh.get("other", {}):
                other[m] = sh["other"][m] * self.machine[m]
        for m in MODES:
            if m not in other:
                o = other.get(_other(m), 0.0)
                other[m] = o * self.cold_ratio if m == "cold" else o / self.cold_ratio
        return roots, other, True

    # ------------------------------------------------------------ coldness

    def prior_cold(self, boot, now):
        """(P(cold) before any evidence, the context it was judged in)."""
        last = self.store.last
        if not last.get("end"):
            ctx = "first"
        elif boot is not None and last.get("boot") is not None and boot > last["boot"] + 120:
            ctx = "boot"
        else:
            gap = now - last["end"]
            ctx = "recent" if gap < 3600 else "hours" if gap < 12 * 3600 else "days"
        cold, n = self.store.priors.get(ctx, (0, 0))
        p = (cold + PRIOR_WEIGHT * PRIOR_COLD[ctx]) / (n + PRIOR_WEIGHT)
        return min(max(p, 0.02), 0.98), ctx

    def cpu_params(self, mode):
        st = self.store.cpu.get(mode)
        if st and st[2] >= 2:
            return st[0], max(math.sqrt(st[1]), 0.04)
        dm, dsd = CPU_DEFAULT[mode]
        if self.profile is not None and mode in self.profile.cpu:
            return self.profile.cpu[mode], dsd
        return dm, dsd


def log_lik(x, mean, var):
    return -(x - mean) ** 2 / (2 * var) - 0.5 * math.log(var)


def cold_logit(prior, fit, cpu_ratio, cpu_elapsed, model):
    """Log-odds that a run is cold, from the prior and the evidence so far.

    fit:   (observed eager time, expected under warm, expected under cold,
            tau of warm, tau of cold) over finished packages.
    """
    logit = math.log(prior / (1 - prior))
    num, dw, dc, tw, tc = fit
    if num > 0 and dw > 0.005 and dc > 0.005:
        for d, tau, sign in ((dw, tw, -1), (dc, tc, 1)):
            v = tau * tau + NOISE_PKG / (1 + d / 0.1)
            logit += sign * log_lik(math.log(num / d), 0.0, v)
    if cpu_ratio is not None and cpu_elapsed > 0.05:
        noise = CPU_CLOCK_RES / cpu_elapsed
        r = min(max(cpu_ratio, 0.0), 1.2)
        for mode, sign in (("warm", -1), ("cold", 1)):
            mean, sd = model.cpu_params(mode)
            logit += sign * log_lik(r, mean, sd * sd + noise * noise)
    return min(max(logit, -9.0), 9.0)
