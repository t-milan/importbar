"""A loading session: records what gets imported and estimates time remaining.

Accounting is by *package* (top-level name) and by *self time*: a module's
execution time minus the time spent in modules it imported. Summed over every
package plus "other" (non-import code in a ``with`` block), self times add up
exactly to wall-clock time, which is what the bar measures.

The dependency graph between packages is learned from nesting (``pandas``
imports ``numpy`` while it executes), so the plan for a session is the
transitive closure of its root imports *minus whatever is already in
``sys.modules``*. Import ``numpy`` early and the bar for ``pandas`` knows to
expect less work.

Every planned package has two expectations, warm and cold (see ``_model.py``).
The estimate is their mixture, weighted by the current probability that this
start is cold, and each is rescaled by its own speed factor fitted to the
packages that have finished. So even when the cold/warm call is wrong, the
estimate still tracks reality.
"""

import math
import os
import sys
import threading
import time
from time import perf_counter

from . import _hook
from ._model import Model, cold_logit
from ._store import MODES, Store, boot_time, ratio_update, stat_update, MAX_MISSES

SPEED_PRIOR = 0.05         # seconds of pseudo-evidence pulling the speed factor to 1
SPEED_MIN, SPEED_MAX = 0.2, 5.0
DEFAULT_ROOT = 0.25        # guess for an explicitly requested, never-seen package
CPU_SAMPLE = 0.01          # seconds between samples of the importing thread's CPU time

_thread_time = getattr(time, "thread_time", time.process_time)


def disabled():
    return os.environ.get("IMPORTBAR_DISABLE", "").strip().lower() in ("1", "true", "yes")


def assumed_mode():
    m = os.environ.get("IMPORTBAR_ASSUME", "").strip().lower()
    return m if m in MODES else None


def _left(expected, spent, running):
    """Remaining time for one phase of one package."""
    if spent < expected:
        rem = expected - spent
        return max(rem, 0.05 * expected) if running else rem
    if running:   # overrunning: a late task tends to run later still
        return 0.25 * (spent - expected) + 0.05 * expected
    return 0.0


class Estimate:
    __slots__ = ("fraction", "remaining", "elapsed", "label", "cold")

    def __init__(self, fraction, remaining, elapsed, label, cold=None):
        self.fraction = fraction    # None = no basis for an estimate yet
        self.remaining = remaining
        self.elapsed = elapsed
        self.label = label
        self.cold = cold            # probability this is a cold start (None if unknown)


class Session:
    def __init__(self, label, roots=None, display=None, desc="Importing",
                 renderer=None, store=None, delay=0.2, fps=20, leave=False,
                 profile=None, origin=None):
        self.label = label
        self.explicit_roots = list(roots) if roots else None
        self._display_opt = display
        self._renderer_opt = renderer
        self._store_opt = store
        self._profile_opt = profile
        self.origin = origin      # directory of the calling code: where to look for a profile
        self.desc = desc
        self.delay = delay
        self.fps = fps
        self.leave = leave
        self.active = False
        self.display = None
        self.profile = None
        self.mode = None          # "warm" / "cold" once the session has ended

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        if self.active or _hook.ACTIVE is not None or disabled():
            return self   # nested or disabled: behave as a plain no-op
        store = self._store_opt or Store.load()
        self.store = store
        self.profile = self._load_profile()
        self.model = model = Model(store, self.profile)
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._stacks = []
        self.preloaded = {k.partition(".")[0] for k in list(sys.modules)}
        self.started = {}         # pkg -> order of first execution this run
        self.done = set()         # pkgs whose top-level module finished executing
        self.spent_at_done = {}   # pkg -> self time when its __init__ finished
        self.skipped = set()      # planned pkgs we now believe won't be imported
        self.spent = {}           # pkg -> self time this run
        self.run_children = {}    # pkg -> pkgs first imported while it executed
        self.roots = []           # pkgs first imported with nothing else executing
        self.current = None
        self.plan = {}            # pkg -> ((warm total, eager), (cold total, eager), known?)
        self._local = {m: set() for m in MODES}   # planned pkgs with this machine's own data
        self._tau = {}            # pkg -> (tau warm, tau cold)

        portable = None
        if self.profile is not None:
            from ._profile import portable_label
            portable = portable_label(self.label, os.path.dirname(self.profile.path))
        hist_roots, self.exp_other, hist_known = model.session(self.label, portable)
        roots = self.explicit_roots if self.explicit_roots is not None else (hist_roots or [])
        for r in roots:
            self._extend_plan(r, root=self.explicit_roots is not None)
        self.known = hist_known or any(k for _, _, k in self.plan.values())

        # Cold or warm? Start from how often starts like this one were cold.
        self.forced = assumed_mode()
        self.boot = boot_time()
        self.prior_cold, self.prior_ctx = model.prior_cold(self.boot, time.time())

        self._main_stack = self._stack()
        self.t0 = perf_counter()
        self._cpu0 = _thread_time()
        self._cpu_sample = (self.t0, self._cpu0)
        self.active = True
        _hook.install(self)
        self._start_display()
        return self

    def stop(self, ok=True):
        if not self.active:
            return
        _hook.uninstall()
        t_end = perf_counter()
        self._cpu_sample = (t_end, _thread_time())
        self.active = False
        try:
            if self.display is not None:
                self.display.finish(t_end - self.t0)
        finally:
            self._learn(t_end - self.t0, ok)
        if os.environ.get("IMPORTBAR_MEASURE"):
            from ._measure import child_report
            child_report(self, t_end - self.t0)   # used by `python -m importbar measure`

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop(ok=exc_type is None)
        return False

    # ---------------------------------------------------------------- planning

    def _load_profile(self):
        if self._profile_opt is False:
            return None
        from ._profile import Profile, find   # only costs anything when used
        if hasattr(self._profile_opt, "times"):
            return self._profile_opt          # a Profile object
        path = find(self.origin, self._profile_opt)
        return Profile.load(path) if path else None

    def _extend_plan(self, pkg, root=False):
        stack = [pkg]
        model = self.model
        while stack:
            p = stack.pop()
            if p in self.plan or p in self.preloaded:
                continue
            exp = model.expect(p)
            if exp is None:
                if not (root and p == pkg):
                    continue   # unknown non-root: nothing to go on, don't guess
                self.plan[p] = ((DEFAULT_ROOT, DEFAULT_ROOT),
                                (DEFAULT_ROOT * model.cold_ratio,) * 2, False)
                self._tau[p] = (1.0, 1.0)
            else:
                w, c = exp["warm"], exp["cold"]
                self.plan[p] = ((w[0], w[1]), (c[0], c[1]), True)
                self._tau[p] = (w[2], c[2])
                for m in MODES:
                    if exp[m][3]:
                        self._local[m].add(p)
            stack.extend(model.deps(p))

    # ---------------------------------------------------------------- hook callbacks

    def _stack(self):
        st = getattr(self._tls, "stack", None)
        if st is None:
            st = self._tls.stack = []
            with self._lock:
                self._stacks.append(st)
        return st

    def _enter(self, name):
        now = perf_counter()
        pkg = name.partition(".")[0]
        st = self._stack()
        with self._lock:
            if pkg not in self.started:
                self.started[pkg] = len(self.started)
                parent = None
                for fr in reversed(st):
                    if fr[0] != pkg:
                        parent = fr[0]
                        break
                if parent is None:
                    if not st:
                        self.roots.append(pkg)
                else:
                    self.run_children.setdefault(parent, set()).add(pkg)
                if pkg not in self.plan and pkg not in self.preloaded:
                    self._extend_plan(pkg)
                self.skipped.discard(pkg)
            self.current = pkg
        st.append([pkg, now, 0.0])

    def _exit(self, name):
        now = perf_counter()
        st = self._stack()
        if not st:
            return
        if st is self._main_stack and now - self._cpu_sample[0] > CPU_SAMPLE:
            self._cpu_sample = (now, _thread_time())   # one tuple: atomic for readers
        pkg, t_start, child = st.pop()
        incl = now - t_start
        if st:
            st[-1][2] += incl
        with self._lock:
            self.spent[pkg] = self.spent.get(pkg, 0.0) + (incl - child)
            if name == pkg:
                self.done.add(pkg)
                self.spent_at_done[pkg] = self.spent[pkg]
                # If this package does (almost) all its work in __init__, planned
                # children that never started under it aren't coming from it.
                e = self.plan.get(pkg)
                if e and e[2] and e[0][1] >= 0.9 * e[0][0]:
                    for c in self.model.deps(pkg):
                        if c in self.plan and c not in self.started:
                            self.skipped.add(c)
            self.current = st[-1][0] if st else None

    # ---------------------------------------------------------------- estimation

    def _fit(self):
        """Eager time of finished packages: observed, and expected per mode."""
        num = dw = dc = tw = tc = 0.0
        plan, tau, at_done = self.plan, self._tau, self.spent_at_done
        for p in self.done:
            e = plan.get(p)
            if e and e[2]:
                num += at_done[p]
                w, c = e[0][1], e[1][1]
                dw += w
                dc += c
                t = tau[p]
                tw += w * t[0]
                tc += c * t[1]
        return (num, dw, dc, tw / dw if dw > 0 else 1.0, tc / dc if dc > 0 else 1.0)

    def cold_probability(self, fit=None):
        """Probability that this is a cold start, given everything seen so far."""
        if self.forced:
            return 1.0 if self.forced == "cold" else 0.0
        if fit is None:
            with self._lock:
                fit = self._fit()
        t, c = self._cpu_sample
        el = t - self.t0
        ratio = (c - self._cpu0) / el if el > 0 else None
        z = cold_logit(self.prior_cold, fit, ratio, el, self.model)
        return 1.0 / (1.0 + math.exp(-z))

    def estimate(self, now=None):
        """Cheap: O(planned packages), called a few times a second by the display."""
        now = perf_counter() if now is None else now
        elapsed = now - self.t0
        with self._lock:
            spent = dict(self.spent)
            on_stack = set()
            for st in self._stacks:            # time of modules executing right now
                if st:
                    pkg, t_start, child = st[-1]
                    spent[pkg] = spent.get(pkg, 0.0) + max(now - t_start - child, 0.0)
                    on_stack.update(fr[0] for fr in st)
            label = self.current
            if not self.known:
                return Estimate(None, None, elapsed, label)

            fit = self._fit()
            pc = self.cold_probability(fit)

            # Speed factor per mode: is this run slower than that mode's history?
            # Compared on the eager phase, which is complete once __init__ returns.
            num, dw, dc = fit[0] + SPEED_PRIOR, fit[1] + SPEED_PRIOR, fit[2] + SPEED_PRIOR
            sw = min(max(num / dw, SPEED_MIN), SPEED_MAX)
            sc = min(max(num / dc, SPEED_MIN), SPEED_MAX)

            rw = rc = 0.0
            done, started, at_done = self.done, self.started, self.spent_at_done
            for p, (w, c, _) in self.plan.items():
                if p in self.skipped:
                    continue
                sp = spent.get(p, 0.0)
                if p in done:   # what's left is work done later by lazily imported submodules
                    a, running = sp - at_done[p], p in on_stack
                    rw += _left(sw * (w[0] - w[1]), a, running)
                    rc += _left(sc * (c[0] - c[1]), a, running)
                elif p in started:
                    rw += _left(sw * w[1], sp, True) + sw * (w[0] - w[1])
                    rc += _left(sc * c[1], sp, True) + sc * (c[0] - c[1])
                else:
                    rw += sw * w[0]
                    rc += sc * c[0]
            other_so_far = max(elapsed - sum(spent.values()), 0.0)
            rw += max(sw * self.exp_other["warm"] - other_so_far, 0.0)
            rc += max(sc * self.exp_other["cold"] - other_so_far, 0.0)

        remaining = pc * rc + (1.0 - pc) * rw
        total = elapsed + remaining
        frac = elapsed / total if total > 0 else 1.0
        return Estimate(frac, remaining, elapsed, label, pc)

    # ---------------------------------------------------------------- learning

    def _learn(self, elapsed, ok):
        store = self.store
        with self._lock:
            fit = self._fit()
        self.cold = pc = self.cold_probability(fit)
        mode = self.mode = "cold" if pc >= 0.5 else "warm"

        # Model: duration = run speed × package base time. Learn the base time by
        # dividing out this run's overall speed, so one slow run (busy machine)
        # is recorded as "slow run", not "every package got slower". Only this
        # machine's own history counts here: dividing by a profile's numbers
        # would just copy the author's machine.
        num = den = 0.0
        local = self._local[mode]
        idx = 0 if mode == "warm" else 1
        for p in self.done:
            if p in local:
                num += self.spent_at_done[p]
                den += self.plan[p][idx][1]
        speed = min(max(num / den, SPEED_MIN), SPEED_MAX) if den > 0.02 else 1.0
        self.run_speed = speed
        times = store.times[mode]
        for p in self.done:
            if p not in self.preloaded:
                old = times.get(p) or (None, None)
                times[p] = [stat_update(old[0], self.spent.get(p, 0.0) / speed),
                            stat_update(old[1], self.spent_at_done[p] / speed)]

        for parent, kids in self.run_children.items():
            d = store.edges.setdefault(parent, {})
            for c in kids:
                d[c] = 0
        never = float("inf")
        for parent in self.done:
            d = store.edges.get(parent)
            if not d:
                continue
            seen = self.run_children.get(parent, ())
            p_order = self.started.get(parent, never)
            for c in list(d):
                if c in seen:
                    continue
                if c in self.preloaded or self.started.get(c, never) < p_order:
                    continue   # absence explained: it was already loaded
                d[c] += 1
                if d[c] >= MAX_MISSES:
                    del d[c]

        t, c = self._cpu_sample
        if t - self.t0 >= 0.3 and self.done:
            store.cpu[mode] = ratio_update(store.cpu.get(mode), (c - self._cpu0) / (t - self.t0))
        if ok and not self.forced and self.done:
            cold, n = store.priors.get(self.prior_ctx, (0, 0))
            store.priors[self.prior_ctx] = [cold + (mode == "cold"), n + 1]
        store.last = {"boot": self.boot, "end": time.time()}

        if ok:
            prev = store.sessions.get(self.label, {})
            if self.explicit_roots is not None:
                roots = list(self.explicit_roots)
            else:
                roots = list(self.roots)
                for r in prev.get("roots", ()):  # keep roots hidden by preloading
                    if r not in roots and (r in self.preloaded or r in self.started):
                        roots.append(r)
            other = max(elapsed - sum(self.spent.values()), 0.0)
            o, tot = dict(prev.get("other", {})), dict(prev.get("total", {}))
            o[mode] = stat_update(o.get(mode), other / speed)
            tot[mode] = stat_update(tot.get(mode), elapsed)
            store.sessions[self.label] = {"roots": roots, "other": o, "total": tot}
        store.save()

    # ---------------------------------------------------------------- display

    def _start_display(self):
        from . import _display
        self.display = _display.make_display(self, self._display_opt, self._renderer_opt)
        if self.display is not None:
            self.display.start()
