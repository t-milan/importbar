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
"""

import os
import sys
import threading
from time import perf_counter

from . import _hook
from ._store import Store, stat_mean, stat_update, MAX_MISSES

SPEED_PRIOR = 0.05         # seconds of pseudo-evidence pulling the speed factor to 1
SPEED_MIN, SPEED_MAX = 0.2, 5.0
DEFAULT_ROOT = 0.25        # guess for an explicitly requested, never-seen package


def disabled():
    return os.environ.get("IMPORTBAR_DISABLE", "").strip().lower() in ("1", "true", "yes")


def _left(expected, spent, running):
    """Remaining time for one phase of one package."""
    if spent < expected:
        rem = expected - spent
        return max(rem, 0.05 * expected) if running else rem
    if running:   # overrunning: a late task tends to run later still
        return 0.25 * (spent - expected) + 0.05 * expected
    return 0.0


class Estimate:
    __slots__ = ("fraction", "remaining", "elapsed", "label")

    def __init__(self, fraction, remaining, elapsed, label):
        self.fraction = fraction    # None = no basis for an estimate yet
        self.remaining = remaining
        self.elapsed = elapsed
        self.label = label


class Session:
    def __init__(self, label, roots=None, display=None, desc="Importing",
                 renderer=None, store=None, delay=0.2, fps=20, leave=False):
        self.label = label
        self.explicit_roots = list(roots) if roots else None
        self._display_opt = display
        self._renderer_opt = renderer
        self._store_opt = store
        self.desc = desc
        self.delay = delay
        self.fps = fps
        self.leave = leave
        self.active = False
        self.display = None

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        if self.active or _hook.ACTIVE is not None or disabled():
            return self   # nested or disabled: behave as a plain no-op
        store = self._store_opt or Store.load()
        self.store = store
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
        self.plan = {}            # pkg -> (expected total, expected eager, from history?)

        hist = store.sessions.get(self.label)
        roots = self.explicit_roots if self.explicit_roots is not None else (
            hist["roots"] if hist else [])
        for r in roots:
            self._extend_plan(r, root=self.explicit_roots is not None)
        self.known = hist is not None or any(k for _, _, k in self.plan.values())
        self.exp_other = stat_mean(hist["other"]) if hist and hist.get("other") else 0.0

        self.t0 = perf_counter()
        self.active = True
        _hook.install(self)
        self._start_display()
        return self

    def stop(self, ok=True):
        if not self.active:
            return
        _hook.uninstall()
        t_end = perf_counter()
        self.active = False
        try:
            if self.display is not None:
                self.display.finish(t_end - self.t0)
        finally:
            self._learn(t_end - self.t0, ok)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop(ok=exc_type is None)
        return False

    # ---------------------------------------------------------------- planning

    def _extend_plan(self, pkg, root=False):
        stack = [pkg]
        store = self.store
        while stack:
            p = stack.pop()
            if p in self.plan or p in self.preloaded:
                continue
            exp = store.expected(p)
            if exp is None:
                if not (root and p == pkg):
                    continue   # unknown non-root: nothing to go on, don't guess
                self.plan[p] = (DEFAULT_ROOT, DEFAULT_ROOT, False)
            else:
                self.plan[p] = (exp[0], exp[1], True)
            stack.extend(store.edges.get(p, ()))

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
                if e and e[2] and e[1] >= 0.9 * e[0]:
                    for c in self.store.edges.get(pkg, ()):
                        if c in self.plan and c not in self.started:
                            self.skipped.add(c)
            self.current = st[-1][0] if st else None

    # ---------------------------------------------------------------- estimation

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

            # Global speed factor: is this run slower than usual across the board?
            # Compared on the eager phase, which is complete once __init__ returns.
            num = den = SPEED_PRIOR
            for p in self.done:
                e = self.plan.get(p)
                if e and e[2]:
                    num += self.spent_at_done[p]
                    den += e[1]
            s = min(max(num / den, SPEED_MIN), SPEED_MAX)

            remaining = 0.0
            done, started, at_done = self.done, self.started, self.spent_at_done
            for p, (total, eager, _) in self.plan.items():
                if p in self.skipped:
                    continue
                sp = spent.get(p, 0.0)
                lazy = s * (total - eager)   # work done later by lazily imported submodules
                if p in done:
                    r = _left(lazy, sp - at_done[p], p in on_stack)
                elif p in started:
                    r = _left(s * eager, sp, True) + lazy
                else:
                    r = s * total
                remaining += r
            other_so_far = max(elapsed - sum(spent.values()), 0.0)
            remaining += max(s * self.exp_other - other_so_far, 0.0)

        total = elapsed + remaining
        frac = elapsed / total if total > 0 else 1.0
        return Estimate(frac, remaining, elapsed, label)

    # ---------------------------------------------------------------- learning

    def _learn(self, elapsed, ok):
        store = self.store
        # Model: duration = run speed × package base time. Learn the base time by
        # dividing out this run's overall speed, so one slow run (busy machine,
        # cold disk) is recorded as "slow run", not "every package got slower".
        num = den = 0.0
        for p in self.done:
            e = self.plan.get(p)
            if e and e[2]:
                num += self.spent_at_done[p]
                den += e[1]
        speed = min(max(num / den, SPEED_MIN), SPEED_MAX) if den > 0.02 else 1.0
        self.run_speed = speed
        for p in self.done:
            if p not in self.preloaded:
                store.packages[p] = stat_update(store.packages.get(p),
                                                self.spent.get(p, 0.0) / speed)
                store.eager[p] = stat_update(store.eager.get(p), self.spent_at_done[p] / speed)

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
            store.sessions[self.label] = {
                "roots": roots,
                "other": stat_update(prev.get("other"), other / speed),
                "total": stat_update(prev.get("total"), elapsed),
            }
        store.save()

    # ---------------------------------------------------------------- display

    def _start_display(self):
        from . import _display
        self.display = _display.make_display(self, self._display_opt, self._renderer_opt)
        if self.display is not None:
            self.display.start()
