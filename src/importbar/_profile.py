"""Shareable timing profiles: ``importbar-profile.txt`` next to your code.

The local history (``_store.py``) lives in the user's cache directory and
starts empty on every new machine. A profile carries the author's measured
timings, warm and cold, to the people who run the code, so their very first
run already has a sensible bar. It is only a starting point: as soon as a
machine has its own measurements for a package, those win.

What transfers across machines is the *shape*: which packages a session pulls
in, how they depend on each other, how long each takes compared with the
others, and how much slower each gets when the start is cold. Absolute speed
does not transfer and doesn't need to: the session rescales everything by how
fast the first packages actually load on the new machine.

The file is plain text, so it is readable, diffable and easy to review before
committing, and it parses in well under a millisecond without ``import json``
(~9 ms). Times are seconds; ``-`` means not measured::

    importbar-profile 1
    cpu      warm=0.98 cold=0.55
    time     pandas   warm=0.4120,0.3800 cold=1.9000,1.7000
    deps     pandas   numpy dateutil pytz
    session  main.py::<module>
      roots  pandas scipy
      other  warm=0.0120
      total  warm=1.2300 cold=4.1000

``time`` gives each package's self time and the part of it spent while its
``__init__`` ran (see ``_session.py``).
"""

import os
import sys

from ._store import MODES, bundle_dir, stat_mean

FILENAME = "importbar-profile.txt"
HEADER = "importbar-profile 1"


class Profile:
    def __init__(self, path=None):
        self.path = path
        self.times = {m: {} for m in MODES}   # mode -> pkg -> (total, eager)
        self.edges = {}                       # parent -> [children]
        self.sessions = {}                    # label -> {"roots", "other", "total"}
        self.cpu = {}                         # mode -> typical CPU/wall ratio
        self.info = ""

    # ------------------------------------------------------------------ reading

    @classmethod
    def load(cls, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            return None
        prof = cls(path)
        try:
            prof._parse(text)
        except (ValueError, IndexError):
            return None   # damaged file: behave as if there were none
        return prof

    def _parse(self, text):
        lines = text.splitlines()
        body = [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]
        if not body or body[0] != HEADER:
            raise ValueError("not an importbar profile")
        sess = None
        for line in body[1:]:
            kind, _, rest = line.partition(" ")
            rest = rest.strip()
            if kind == "time":
                name, *fields = rest.split()
                for m, val in _pairs(fields):
                    if val != "-":
                        total, _, eager = val.partition(",")
                        total = float(total)
                        self.times[m][name] = (total, min(float(eager or total), total))
            elif kind == "deps":
                name, *kids = rest.split()
                self.edges[name] = kids
            elif kind == "cpu":
                for m, val in _pairs(rest.split()):
                    if val != "-":
                        self.cpu[m] = float(val)
            elif kind == "session":
                sess = self.sessions[rest] = {"roots": [], "other": {}, "total": {}}
            elif kind == "roots" and sess is not None:
                sess["roots"] = rest.split()
            elif kind in ("other", "total") and sess is not None:
                for m, val in _pairs(rest.split()):
                    if val != "-":
                        sess[kind][m] = float(val)
            elif kind == "measured":
                self.info = rest
            # unknown keywords are ignored, so newer files still load

    # ------------------------------------------------------------------ writing

    def dumps(self):
        out = [
            "# importbar timing profile: how long this program's imports took on the",
            "# machine it was measured on, for warm starts and for cold starts (first",
            "# run after a reboot). Keep it next to your code. importbar uses it on",
            "# machines that haven't learned their own timings yet, and adjusts it",
            "# for their speed. Regenerate with:  python -m importbar measure <script>",
            "# Times are seconds; '-' means not measured.",
            HEADER,
        ]
        if self.info:
            out.append("measured %s" % self.info)
        if self.cpu:
            out.append("cpu      " + _fmt_modes(self.cpu, "%.2f"))
        for label in sorted(self.sessions):
            s = self.sessions[label]
            out.append("")
            out.append("session  %s" % label)
            out.append("  roots  %s" % " ".join(s["roots"]))
            if s.get("other"):
                out.append("  other  " + _fmt_modes(s["other"]))
            if s.get("total"):
                out.append("  total  " + _fmt_modes(s["total"]))
        names = sorted(set(self.times["warm"]) | set(self.times["cold"]),
                       key=lambda p: -max(self.times[m].get(p, (0, 0))[0] for m in MODES))
        if names:
            out.append("")
            out.append("#        package  mode=self-time,during-__init__")
        width = max([len(n) for n in names] + [8])
        for p in names:
            cols = []
            for m in MODES:
                e = self.times[m].get(p)
                cols.append("%s=%s" % (m, "%.4f,%.4f" % e if e else "-"))
            out.append("time     %-*s %s" % (width, p, " ".join(cols)))
        deps = [p for p in sorted(self.edges) if self.edges[p]]
        if deps:
            out.append("")
            out.append("#        package  packages it imports")
        for p in deps:
            out.append("deps     %-*s %s" % (width, p, " ".join(sorted(self.edges[p]))))
        return "\n".join(out) + "\n"

    def save(self, path=None):
        path = path or self.path
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.dumps())
        os.replace(tmp, path)
        self.path = path


def _pairs(fields):
    for f in fields:
        m, eq, val = f.partition("=")
        if eq and m in MODES:
            yield m, val


def _fmt_modes(d, fmt="%.4f"):
    return " ".join("%s=%s" % (m, fmt % d[m] if m in d else "-") for m in MODES)


# ---------------------------------------------------------------------- locating

def find(origin_dir, explicit=None):
    """Path of the profile to use for code in ``origin_dir``, or None.

    ``explicit`` is the ``profile=`` argument: False disables profiles, a path
    names a file or the directory holding one. Otherwise ``IMPORTBAR_PROFILE``,
    then ``origin_dir`` and its parent are searched (so a profile at the root
    of a ``src/`` layout is found too). In a frozen app the search stays inside
    the bundle.
    """
    if explicit is False:
        return None
    cand = explicit or os.environ.get("IMPORTBAR_PROFILE")
    if cand:
        cand = os.fspath(cand)
        if os.path.isdir(cand):
            cand = os.path.join(cand, FILENAME)
        return cand if os.path.isfile(cand) else None
    if not origin_dir:
        return None
    d = origin_dir
    bundle = bundle_dir()
    for _ in range(2):
        p = os.path.join(d, FILENAME)
        if os.path.isfile(p):
            return p
        if bundle and os.path.normcase(d) == os.path.normcase(bundle):
            break   # a one-file build's parent is the system temp folder
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def portable_label(label, base_dir):
    """Session label as stored in a profile: file paths relative to the profile."""
    path, sep, func = label.rpartition("::")
    if not sep or not base_dir:
        return label
    if not os.path.isabs(path):
        bundle = bundle_dir()   # frozen app: labels are relative to the bundle
        if not bundle:
            return label
        path = os.path.join(bundle, path)
    try:
        rel = os.path.relpath(path, base_dir)
    except ValueError:   # different drive on Windows
        rel = os.path.basename(path)
    return "%s::%s" % (rel.replace(os.sep, "/"), func)


# ---------------------------------------------------------------------- building

def build(store, labels, base_dir):
    """A profile with the given sessions from a local store, and everything they import."""
    prof = Profile()
    prof.info = "python=%s platform=%s" % (sys.version.split()[0], sys.platform)
    for m in MODES:
        if store.cpu.get(m):
            prof.cpu[m] = store.cpu[m][0]
    wanted = set()
    for label in labels:
        s = store.sessions.get(label)
        if not s:
            continue
        prof.sessions[portable_label(label, base_dir)] = {
            "roots": list(s.get("roots", [])),
            "other": {m: stat_mean(st) for m, st in s.get("other", {}).items()},
            "total": {m: stat_mean(st) for m, st in s.get("total", {}).items()},
        }
        stack = list(s.get("roots", []))
        while stack:
            p = stack.pop()
            if p in wanted:
                continue
            wanted.add(p)
            stack.extend(store.edges.get(p, ()))
    for p in wanted:
        for m in MODES:
            e = store.expected(p, m)
            if e:
                prof.times[m][p] = e
        kids = [c for c in store.edges.get(p, ()) if c in wanted]
        if kids:
            prof.edges[p] = kids
    return prof
