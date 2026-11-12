"""Command line: measure startups, share them, inspect or reset learned timings.

    python -m importbar measure app.py [args]   # time warm + cold starts, write importbar-profile.txt
    python -m importbar measure -m mypkg        # same, for python -m mypkg
    python -m importbar export [DIR]            # refresh DIR's profile from what this machine learned
    python -m importbar report                  # slowest packages in this environment
    python -m importbar try numpy pandas
    python -m importbar clear
    python -m importbar path
"""

import os
import sys

from ._store import MODES, Store, stat_mean


def _report(store, top):
    names = set(store.times["warm"]) | set(store.times["cold"])
    if not names:
        print("No timings learned yet for this environment (%s)." % store.path)
        return

    def exp(p, m):
        e = store.times[m].get(p)
        return (stat_mean(e[0]), e[0][2]) if e else (None, 0)

    rows = sorted(names, key=lambda p: -max(exp(p, m)[0] or 0.0 for m in MODES))
    print("%-28s %10s %10s %6s" % ("package (self time)", "warm", "cold", "runs"))
    for name in rows[:top]:
        (w, nw), (c, nc) = exp(name, "warm"), exp(name, "cold")
        print("%-28s %10s %10s %6s" % (name, "%.3fs" % w if w is not None else "-",
                                       "%.3fs" % c if c is not None else "-",
                                       "%d/%d" % (nw, nc)))
    if len(rows) > top:
        print("... %d more" % (len(rows) - top))
    if store.sessions:
        print("\n%-44s %8s %8s  roots" % ("session", "warm", "cold"))
        for label, s in store.sessions.items():
            tot = s.get("total", {})
            cols = ["%.2fs" % stat_mean(tot[m]) if tot.get(m) else "-" for m in MODES]
            short = label if len(label) <= 44 else "…" + label[-43:]
            print("%-44s %8s %8s  %s" % (short, cols[0], cols[1], ", ".join(s.get("roots", []))))
    if store.priors:
        print("\ncold starts so far, by context: " + "  ".join(
            "%s %d/%d" % (ctx, cold, n) for ctx, (cold, n) in sorted(store.priors.items())))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "report"
    if cmd == "report":
        top = int(argv[0]) if argv else 25
        _report(Store.load(), top)
    elif cmd == "measure":
        from ._measure import measure
        return measure(argv)
    elif cmd == "export":
        from ._measure import export
        return export(argv)
    elif cmd == "path":
        print(Store().path)
    elif cmd == "clear":
        path = Store().path
        try:
            os.remove(path)
            print("Removed", path)
        except FileNotFoundError:
            print("Nothing to remove at", path)
    elif cmd == "try":
        if not argv:
            print("usage: python -m importbar try MODULE [MODULE ...]")
            return 2
        import importbar
        importbar.load(*argv, display=True, leave=True)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
