"""``python -m importbar measure``: time a program's startup and write a profile.

The program is run several times in fresh interpreters. Each run stops as soon
as its first importbar session ends (``IMPORTBAR_MEASURE`` makes ``stop()``
report and exit), so the program's real work (a GUI, a server, a long
computation) never runs. Warm runs come first. On Linux, cold runs follow:
every file the program loaded is evicted from the OS file cache beforehand
with ``posix_fadvise``, which needs no special rights. Other systems have no
unprivileged way to do this, so there importbar learns cold starts from normal
use (the first run after a reboot), and ``python -m importbar export`` adds
them to the profile later.
"""

import os
import sys

from . import _profile
from ._store import MODES, Store, cache_dir, frozen_history


# --------------------------------------------------------------------- child side

def child_report(session, elapsed):
    import json
    files = set()
    for m in list(sys.modules.values()):
        for attr in ("__file__", "__cached__"):
            f = getattr(m, attr, None)
            if isinstance(f, str):
                files.add(f)
    try:   # shared libraries, including those loaded by extension modules
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    files.add(parts[5].strip())
    except OSError:
        pass
    data = {"label": session.label, "origin": session.origin, "elapsed": elapsed,
            "bundle_dir": _bundle_rel(session.origin),
            "mode": session.mode, "cold": getattr(session, "cold", None),
            "files": sorted(files)}
    try:
        with open(os.environ["IMPORTBAR_MEASURE"], "w", encoding="utf-8") as f:
            json.dump(data, f)
    finally:
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass
        os._exit(0)


def _bundle_rel(origin):
    """Where a freezer (PyInstaller) puts the folder ``origin``: its path inside
    the bundle, "" for the bundle root, None if unknown. Bundles are laid out
    by module name, so this comes from the modules loaded from that folder."""
    if not origin:
        return None
    origin = os.path.normcase(origin)
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not isinstance(f, str) or os.path.normcase(os.path.dirname(os.path.abspath(f))) != origin:
            continue
        if name == "__main__":
            spec = getattr(m, "__spec__", None)
            if spec is None:
                return ""          # a script: the bundle's entry point, at its root
            name = spec.name       # python -m pkg: pkg.__main__
        if os.path.basename(f).startswith("__init__."):
            return name.replace(".", "/")
        return name.rpartition(".")[0].replace(".", "/")
    return None


# --------------------------------------------------------------------- parent side

def can_evict():
    return hasattr(os, "posix_fadvise") and sys.platform.startswith("linux")


def evict(paths):
    """Drop files from the OS page cache (Linux). Returns how many were evicted."""
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return n


def _run(cmd, env_extra, timeout):
    import json
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(prefix="importbar-measure-", suffix=".json")
    os.close(fd)
    env = dict(os.environ, IMPORTBAR_MEASURE=out, IMPORTBAR_NO_DISPLAY="1", **env_extra)
    env.pop("IMPORTBAR_DISABLE", None)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        try:
            with open(out, encoding="utf-8") as f:
                text = f.read()
            if text:
                return json.loads(text), proc
        except (OSError, ValueError):
            pass
        return None, proc
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def _fail(proc, cmd):
    print("\nThe program finished without starting an importbar session, so there was")
    print("nothing to measure. Wrap its slow imports in importbar.progress(),")
    print("importbar.load() or importbar.start()/stop() first.")
    print("command: %s (exit code %s)" % (" ".join(cmd), proc.returncode))
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
    if tail:
        print("last output:\n  " + "\n  ".join(tail))
    return 1


def measure(argv):
    import argparse
    import tempfile
    ap = argparse.ArgumentParser(
        prog="python -m importbar measure",
        usage="%(prog)s [options] (script.py | -m module) [args...]",
        epilog="Options go before the script; anything after it is passed to the script.",
        description="Time a program's imports, warm and cold, and write "
                    "%s next to the code that shows the bar." % _profile.FILENAME)
    ap.add_argument("--runs", type=int, default=5, help="warm runs (default 5)")
    ap.add_argument("--cold", type=int, default=None,
                    help="cold runs (default 3 on Linux; other systems can't simulate them)")
    ap.add_argument("-o", "--output", help="profile file or directory to write "
                    "(default: next to the file that calls importbar)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("-m", dest="module", help="run a module, like python -m")
    ap.add_argument("target", nargs=argparse.REMAINDER, help="script.py [args...]")
    args = ap.parse_args(argv)
    if args.module:
        cmd = [sys.executable, "-m", args.module] + args.target
    elif args.target:
        cmd = [sys.executable] + args.target
    else:
        ap.error("give a script to run, or -m module")
    n_cold = args.cold if args.cold is not None else (3 if can_evict() else 0)
    if n_cold and not can_evict():
        print("Cold starts can only be simulated on Linux; skipping the cold runs.")
        n_cold = 0

    # Warm-up: fills the OS cache and finds out what the program loads, without
    # touching the real history (it may itself be a cold start).
    with tempfile.TemporaryDirectory() as tmp:
        print("warm-up run ...", end=" ", flush=True)
        first, proc = _run(cmd, {"IMPORTBAR_CACHE_DIR": tmp}, args.timeout)
    if first is None:
        return _fail(proc, cmd)
    print("%.2fs" % first["elapsed"])
    files = set(first["files"])
    label, origin = first["label"], first["origin"]

    results = {m: [] for m in MODES}
    plan = [("warm", i) for i in range(args.runs)] + [("cold", i) for i in range(n_cold)]
    for mode, i in plan:
        n = args.runs if mode == "warm" else n_cold
        print("%s run %d/%d ..." % (mode, i + 1, n), end=" ", flush=True)
        if mode == "cold":
            evict(files)
        res, proc = _run(cmd, {"IMPORTBAR_ASSUME": mode}, args.timeout)
        if res is None:
            return _fail(proc, cmd)
        files.update(res["files"])
        results[mode].append(res["elapsed"])
        print("%.2fs" % res["elapsed"])

    path = write_profile(args.output or origin or os.getcwd(), [label])
    print()
    for m in MODES:
        if results[m]:
            xs = sorted(results[m])
            print("%s start: %.2fs (median of %d)" % (m, xs[len(xs) // 2], len(xs)))
    if results["warm"] and results["cold"]:
        w, c = sorted(results["warm"]), sorted(results["cold"])
        ratio = c[len(c) // 2] / max(w[len(w) // 2], 1e-6)
        print("cold starts are %.1fx slower than warm ones on this machine" % ratio)
    elif not n_cold:
        print("cold: not measured. importbar learns them on its own from the first run")
        print("after each reboot; `python -m importbar export %s` adds them to the"
              % _rel(os.path.dirname(path)))
        print("profile later.")
    print("\nwrote %s" % _rel(path))
    print("Ship it next to your code; importbar finds it there automatically.")
    dest = _bundle_dest(first.get("bundle_dir"), origin, os.path.dirname(path))
    if dest is not None:
        src = _rel(path)
        print("\nTo bundle it with PyInstaller, add:  --add-data \"%s:%s\"" % (src, dest))
        print("(in a .spec file:  datas=[(%r, %r)])" % (src, dest))
    return 0


def _bundle_dest(code_dir, origin, profile_dir):
    """Folder inside a PyInstaller bundle where the profile must go, or None.

    At run time the profile is looked for in the calling file's folder and the
    one above, and session names are relative to the profile, so it goes in the
    same place relative to the code as it has in the source tree.
    """
    if code_dir is None or not origin:
        return None
    try:
        up = os.path.relpath(profile_dir, origin)
    except ValueError:
        return None
    if up not in (os.curdir, os.pardir):
        return None   # not where importbar looks; IMPORTBAR_PROFILE or profile= needed
    dest = code_dir if up == os.curdir else code_dir.rpartition("/")[0]
    if up == os.pardir and not code_dir:
        return None   # would be above the bundle root
    return dest or "."


def _rel(p):
    try:
        r = os.path.relpath(p)
        return r if not r.startswith(".." + os.sep + "..") else p
    except ValueError:
        return p


def _labels_for(store, directory, existing):
    """Sessions in the local history that belong to code in ``directory``."""
    directory = os.path.abspath(directory)
    out = []
    for label in store.sessions:
        path, sep, _ = label.rpartition("::")
        if sep and os.path.isabs(path):
            try:
                inside = os.path.commonpath([directory, os.path.abspath(path)]) == directory
            except ValueError:
                inside = False
            if inside:
                out.append(label)
        elif existing is not None and label in existing.sessions:
            out.append(label)   # load:... or custom labels already in the profile
    return out


def _frozen_labels(store, directory, existing):
    """Sessions in a frozen app's history that belong to code in ``directory``.

    A frozen app labels sessions by the calling file's path inside the bundle
    (``myapp/cli.py::main``). Returns {label: name in the profile}: that path
    made relative to ``directory``, found by matching it against the source
    files there (or against the sessions the profile already has).
    """
    out = {}
    for label in store.sessions:
        path, sep, func = label.rpartition("::")
        if not sep or os.path.isabs(path):
            if existing is not None and label in existing.sessions:
                out[label] = label   # custom labels already in the profile
            continue
        parts = path.split("/")
        for k in range(len(parts)):   # longest match first
            name = "/".join(parts[k:]) + "::" + func
            if ((existing is not None and name in existing.sessions)
                    or os.path.isfile(os.path.join(directory, *parts[k:]))):
                out[label] = name
                break
    return out


def write_profile(where, labels=(), store=None, frozen=False):
    """Write (or refresh) the profile for code in ``where`` from local history.

    ``frozen``: ``store`` is a frozen app's history (see ``_frozen_labels``).
    """
    store = store or Store.load()
    where = os.path.abspath(where)
    if where.endswith(".txt") or (os.path.exists(where) and not os.path.isdir(where)):
        path = where
    else:
        path = os.path.join(where, _profile.FILENAME)
    base = os.path.dirname(path)
    existing = _profile.Profile.load(path) if os.path.exists(path) else None
    names = None
    if frozen:
        names = _frozen_labels(store, base, existing)
        wanted = list(names)
    else:
        wanted = list(labels) + [l for l in _labels_for(store, base, existing) if l not in labels]
    prof = _profile.build(store, wanted, base, names)
    if frozen:
        prof.info += " source=frozen-app"
    if existing is not None:
        # Keep sessions this machine hasn't run, rather than silently dropping them.
        for label, s in existing.sessions.items():
            if label not in prof.sessions:
                prof.sessions[label] = s
                for p in _closure(existing, s["roots"]):
                    for m in MODES:
                        if p in existing.times[m] and p not in prof.times[m]:
                            prof.times[m][p] = existing.times[m][p]
                    if p in existing.edges and p not in prof.edges:
                        prof.edges[p] = existing.edges[p]
    prof.save(path)
    return path


def _closure(prof, roots):
    seen, stack = set(), list(roots)
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(prof.edges.get(p, ()))
    return seen


def export(argv):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m importbar export",
        description="Refresh %s from the timings this machine has learned." % _profile.FILENAME)
    ap.add_argument("where", nargs="?", default=os.getcwd(),
                    help="the code's folder, a script in it, or the profile (default: .)")
    ap.add_argument("--from", dest="app", metavar="APP",
                    help="use what a frozen (PyInstaller) build learned instead of this "
                         "Python environment: its executable, its folder, or a .app")
    args = ap.parse_args(argv)
    where = args.where
    if os.path.isfile(where) and where.endswith(".py"):
        where = os.path.dirname(os.path.abspath(where))
    if args.app:
        hist = frozen_history(args.app)
        if hist is None:
            print("No history found for %s in %s." % (args.app, cache_dir()))
            print("Run the built app at least once on this machine, as this user, first.")
            return 1
        store = Store.load(hist)
    else:
        store = Store.load()
    path = write_profile(where, store=store, frozen=bool(args.app))
    prof = _profile.Profile.load(path)
    n = len(prof.sessions) if prof else 0
    print("wrote %s (%d session%s)" % (_rel(path), n, "" if n == 1 else "s"))
    if prof:
        for label, s in sorted(prof.sessions.items()):
            tot = s.get("total", {})
            print("  %-40s warm %s  cold %s" % (
                label[-40:], *("%.2fs" % tot[m] if m in tot else "-" for m in MODES)))
    if not n:
        if args.app:
            print("None of the app's sessions match code in that directory. Give the folder")
            print("of the source file that calls importbar (or the one above it).")
        else:
            print("No sessions from code in that directory are in this machine's history yet.")
    return 0
