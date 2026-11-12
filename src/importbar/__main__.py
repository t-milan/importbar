"""Command line: inspect or reset learned timings, or try the bar.

    python -m importbar report          # slowest packages in this environment
    python -m importbar try numpy pandas
    python -m importbar clear
    python -m importbar path
"""

import os
import sys

from ._store import Store, stat_mean


def _report(store, top):
    if not store.packages:
        print("No timings learned yet for this environment (%s)." % store.path)
        return
    rows = sorted(((stat_mean(st), name, st[2]) for name, st in store.packages.items()),
                  reverse=True)
    print("%-28s %10s %6s" % ("package (self time)", "expected", "runs"))
    for exp, name, n in rows[:top]:
        print("%-28s %9.3fs %6d" % (name, exp, n))
    if len(rows) > top:
        print("... %d more" % (len(rows) - top))
    if store.sessions:
        print("\n%-50s %10s  roots" % ("session", "total"))
        for label, s in store.sessions.items():
            tot = stat_mean(s["total"]) if s.get("total") else float("nan")
            short = label if len(label) <= 50 else "…" + label[-49:]
            print("%-50s %9.3fs  %s" % (short, tot, ", ".join(s.get("roots", []))))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "report"
    if cmd == "report":
        top = int(argv[0]) if argv else 25
        _report(Store.load(), top)
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
