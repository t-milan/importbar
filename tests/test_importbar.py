import io
import json
import marshal
import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
sys.path.insert(0, os.path.abspath(SRC))

from importbar._display import Smoother, TerminalRenderer, Progress, CAP  # noqa: E402
from importbar._session import Estimate  # noqa: E402

BUSY = """
import os as _os, time as _t
_slow = float(_os.environ.get("FAKE_SLOW") or 1)   # a slower machine
_cold = float(_os.environ.get("FAKE_COLD") or 0)   # a cold start: extra waiting, no CPU
def _busy(sec):
    sec *= _slow
    if _cold:
        _t.sleep(sec * _cold)        # like waiting for the disk or the virus scanner
    end = _t.perf_counter() + sec
    while _t.perf_counter() < end:   # holds the GIL, like real import work
        pass
"""


@pytest.fixture
def env(tmp_path):
    pkgs = tmp_path / "pkgs"
    pkgs.mkdir()

    def pkg(name, init, files=None):
        d = pkgs / name
        d.mkdir()
        (d / "__init__.py").write_text(BUSY + textwrap.dedent(init))
        for fname, body in (files or {}).items():
            (d / fname).write_text(BUSY + textwrap.dedent(body))

    # fa: 4 submodules of 50ms each + 100ms of its own => ~0.3s self time
    pkg("fa", "from . import m1, m2, m3, m4\n_busy(0.10)\n",
        {"m%d.py" % i: "_busy(0.05)\n" for i in range(1, 5)})
    pkg("fb", "import fa\n_busy(0.20)\n")                       # depends on fa
    pkg("fc", "_busy(0.30)\n")
    pkg("fd", "import os\nif os.environ.get('FD_USE_FA'):\n    import fa\n_busy(0.10)\n")
    pkg("fbad", "_busy(0.05)\nraise RuntimeError('boom')\n")

    cache = tmp_path / "cache"
    base_env = dict(os.environ, IMPORTBAR_CACHE_DIR=str(cache),
                    PYTHONPATH=os.pathsep.join([os.path.abspath(SRC), str(pkgs)]))
    base_env.pop("IMPORTBAR_DISABLE", None)

    def run(code, extra_env=None, args=None, cwd=None):
        e = dict(base_env, **(extra_env or {}))
        argv = args if args is not None else ["-c", textwrap.dedent(code)]
        out = subprocess.run([sys.executable] + argv, env=e, cwd=cwd,
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise AssertionError(out.stdout + out.stderr)
        run.last_stdout = out.stdout
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else None

    def store(cache_dir=None):
        c = cache_dir or cache
        files = list(c.glob("timings-*.bin")) if c.exists() else []
        return marshal.loads(files[0].read_bytes()) if files else None

    run.store = store
    run.tmp = tmp_path
    return run


RECORDER = """
import json, time, sys, importbar
class Rec:
    def __init__(self): self.rows = []; self.sess = None
    def __call__(self, p):
        e = self.sess.estimate()
        self.rows.append([p.elapsed, p.fraction, e.fraction, e.remaining, p.done])
rec = Rec()
"""


def test_imports_are_transparent(env):
    res = env("""
        import sys, inspect, importlib.resources, importbar
        before = list(sys.meta_path)
        with importbar.progress(display=False):
            import fa
        print(json.dumps({
            "loader": type(fa.__loader__).__name__,
            "patched": "exec_module" in vars(fa.__loader__),
            "sub_patched": any("exec_module" in vars(sys.modules[m].__loader__)
                               for m in ("fa.m1", "fa.m2")),
            "source": "_busy" in inspect.getsource(fa),
            "resources": importlib.resources.files(fa).joinpath("m1.py").is_file(),
            "meta_path_restored": sys.meta_path == before,
            "attr": hasattr(fa, "m4"),
        }))
    """.replace("import sys,", "import json, sys,"))
    assert res == {"loader": "SourceFileLoader", "patched": False, "sub_patched": False,
                   "source": True, "resources": True, "meta_path_restored": True, "attr": True}


def test_learns_self_times_and_edges(env):
    env("import importbar; importbar.load('fb', 'fc', display=False)")
    st = env.store()
    p = {k: v for k, v in st["times"]["warm"].items() if k in ("fa", "fb", "fc")}
    import math
    exp = {k: math.exp(v[0][0]) for k, v in p.items()}
    assert exp["fa"] == pytest.approx(0.30, rel=0.25)
    assert exp["fb"] == pytest.approx(0.20, rel=0.25)   # self time excludes fa
    assert exp["fc"] == pytest.approx(0.30, rel=0.25)
    assert "fa" in st["edges"]["fb"]


def test_prediction_accuracy_and_smoothness(env):
    code = RECORDER + """
sess = importbar.Session("acc", roots=["fb", "fc"], renderer=rec, delay=0.0, fps=50)
rec.sess = sess
t0 = time.perf_counter()
with sess:
    import fb, fc
actual = time.perf_counter() - t0
print(json.dumps({"rows": rec.rows, "actual": actual, "known": sess.known}))
"""
    env(code)
    env(code)
    res = env(code)
    assert res["known"]
    actual = res["actual"]
    live = [r for r in res["rows"] if not r[4] and r[2] is not None and r[0] > 0.1 * actual]
    assert len(live) > 10
    errs = [abs((r[0] + r[3]) - actual) / actual for r in live]
    assert max(errs) < 0.15, max(errs)
    shown = [r[1] for r in res["rows"]]
    assert all(b >= a for a, b in zip(shown, shown[1:])), "bar went backwards"
    assert shown[-1] == 1.0 and max(shown[:-1]) <= CAP


def test_preloaded_dependency_shrinks_plan(env):
    env("import importbar; importbar.load('fb', display=False)")
    res = env("""
        import json, importbar, fa
        s = importbar.Session("load:fb", roots=["fb"], display=False).start()
        plan = sorted(s.plan)
        e = s.estimate()
        s.stop()
        print(json.dumps({"plan": plan, "pred": e.elapsed + e.remaining}))
    """)
    assert res["plan"] == ["fb"]
    assert res["pred"] == pytest.approx(0.20, rel=0.3)


def test_context_manager_learns_roots(env):
    code = """
        import json, importbar
        with importbar.progress(label="app", display=False) as s:
            import fb
            import fc
        print(json.dumps({"known": s.known, "plan": sorted(s.plan)}))
    """
    first = env(code)
    second = env(code)
    assert first["known"] is False                 # first run: indeterminate
    assert second["known"] is True
    assert {"fa", "fb", "fc"} <= set(second["plan"])
    assert env.store()["sessions"]["app"]["roots"][:2] == ["fb", "fc"]


def test_unused_dependency_is_skipped_then_forgotten(env):
    env("import importbar; importbar.load('fd', display=False)", {"FD_USE_FA": "1"})
    assert "fa" in env.store()["edges"]["fd"]
    res = env("""
        import json, importbar
        s = importbar.Session("load:fd", roots=["fd"], display=False).start()
        import fd
        skipped = sorted(s.skipped)
        s.stop()
        print(json.dumps({"skipped": skipped}))
    """)
    assert res["skipped"] == ["fa"]
    for _ in range(3):
        env("import importbar; importbar.load('fd', display=False)")
    assert "fa" not in env.store()["edges"].get("fd", {})


def test_exception_propagates_and_cleans_up(env):
    res = env("""
        import json, sys, importbar
        try:
            with importbar.progress(display=False):
                import fc
                import fbad
        except RuntimeError as e:
            msg = str(e)
        print(json.dumps({"msg": msg, "clean": not any("importbar" in repr(f) for f in sys.meta_path)}))
    """)
    assert res == {"msg": "boom", "clean": True}
    assert "fc" in env.store()["times"]["warm"]


def test_disabled_is_plain_import(env):
    res = env("""
        import json, sys, importbar
        with importbar.progress(display=True) as s:
            import fa
        print(json.dumps({"active": s.active, "display": s.display is None}))
    """, {"IMPORTBAR_DISABLE": "1"})
    assert res == {"active": False, "display": True}
    assert env.store() is None


def test_nested_sessions_are_noops(env):
    res = env("""
        import json, importbar
        with importbar.progress(display=False) as outer:
            fb = importbar.load("fb", display=False)
        print(json.dumps({"outer_saw_fb": "fb" in outer.done}))
    """)
    assert res["outer_saw_fb"]


# ------------------------------------------------------------------ unit tests

def test_smoother_monotonic_and_capped():
    sm = Smoother()
    seq = [0.1, 0.3, 0.2, 0.25, 0.9, 0.5, 1.0, 1.0, 1.0]
    out = []
    for i, f in enumerate(seq * 20):
        out.append(sm.step(0.05, Estimate(f, 1.0, i * 0.05, None)))
    assert all(b >= a for a, b in zip(out, out[1:]))
    assert max(out) <= CAP


def test_smoother_unknown_crawls():
    sm = Smoother()
    vals = [sm.step(0.05, Estimate(None, None, t * 0.05, None)) for t in range(400)]
    assert 0 < vals[20] < vals[100] < vals[-1] <= CAP


def test_terminal_renderer_line():
    buf = io.StringIO()
    r = TerminalRenderer(stream=buf, width=80)
    r(Progress(0.5, "pandas", 1.0, 1.0, False, "Importing", False))
    r(Progress(1.0, None, 2.0, 0.0, True, "Importing", False))
    r.close(True)
    text = buf.getvalue()
    assert "50%" in text and "pandas" in text and "done in 2.0s" in text
    assert all(len(seg) <= 80 for seg in text.split("\r"))


# ------------------------------------------------------------------ cold vs warm

LIVE = RECORDER.replace("p.done]", "p.done, e.cold]") + """
sess = importbar.Session("live", roots=["fb", "fc"], renderer=rec, delay=0.0, fps=50)
rec.sess = sess
t0 = time.perf_counter()
with sess:
    import fb, fc
actual = time.perf_counter() - t0
print(json.dumps({"rows": rec.rows, "actual": actual, "mode": sess.mode, "cold": sess.cold}))
"""


def _errors(res, start=0.3):
    a = res["actual"]
    live = [r for r in res["rows"] if not r[4] and r[2] is not None and r[0] > start * a]
    assert len(live) > 5
    return [abs((r[0] + r[3]) - a) / a for r in live]


def test_cold_runs_are_detected_and_learned_separately(env):
    cold = {"FAKE_COLD": "2"}       # 3x slower, two thirds of it spent waiting
    for _ in range(2):
        env(LIVE)
    res = env(LIVE, cold)       # never seen a cold run, yet recognised as one
    assert res["mode"] == "cold" and res["cold"] > 0.9
    env(LIVE, cold)
    st = env.store()
    import math
    w, c = st["times"]["warm"]["fc"][0], st["times"]["cold"]["fc"][0]
    assert (w[2], c[2]) == (2, 2)
    assert math.exp(c[0] - w[0]) == pytest.approx(3.0, rel=0.25)
    assert st["cpu"]["warm"][0] > 0.85 and st["cpu"]["cold"][0] < 0.5

    # Now both kinds of start are predicted well, and told apart early on.
    res = env(LIVE, cold)
    assert res["mode"] == "cold"
    assert max(_errors(res)) < 0.15
    mid = [r[5] for r in res["rows"] if 0.2 * res["actual"] < r[0] < 0.5 * res["actual"]]
    assert mid and min(mid) > 0.8
    res = env(LIVE)
    assert res["mode"] == "warm"
    assert max(_errors(res)) < 0.15


def test_prior_contexts():
    from importbar._model import Model
    from importbar._store import Store
    st = Store(path=os.devnull)
    m = Model(st)
    assert m.prior_cold(1000.0, 5000.0) == (0.5, "first")
    st.last = {"boot": 1000.0, "end": 4000.0}
    p, ctx = m.prior_cold(1000.0, 4100.0)
    assert ctx == "recent" and p < 0.1
    p, ctx = m.prior_cold(3000.0, 9000.0)          # rebooted since the last run
    assert ctx == "boot" and p > 0.8
    assert m.prior_cold(1000.0, 4000.0 + 86400)[1] == "days"
    st.priors["boot"] = [0, 10]                    # this machine: reboots aren't cold
    assert m.prior_cold(3000.0, 9000.0)[0] < 0.3


def test_v1_store_is_migrated(tmp_path):
    from importbar._store import Store
    path = tmp_path / "t.bin"
    st = [0.0, 0.1, 3]
    path.write_bytes(marshal.dumps({
        "format": 1, "packages": {"a": st}, "eager": {"a": st}, "edges": {"a": {"b": 0}},
        "sessions": {"x": {"roots": ["a"], "other": st, "total": st}}}))
    s = Store.load(str(path))
    assert s.times["warm"]["a"] == [st, st] and s.times["cold"] == {}
    assert s.sessions["x"]["total"] == {"warm": st}
    assert s.edges == {"a": {"b": 0}}


# ------------------------------------------------------------------ shared profiles

APP = """
import json, sys, time
t0 = time.perf_counter()
import importbar
class Rec:
    def __init__(self): self.rows = []
    def __call__(self, p):
        e = sess.estimate()
        self.rows.append([p.elapsed, p.fraction, e.fraction, e.remaining, p.done, e.cold])
rec = Rec()
sess = importbar.progress(renderer=rec, delay=0.0, fps=50)
with sess:
    import fb
    import fc
actual = time.perf_counter() - t0
print(json.dumps({"rows": rec.rows, "actual": actual, "known": sess.known,
                  "mode": sess.mode, "profile": bool(sess.profile)}))
"""


def _app(env):
    d = env.tmp / "app"
    d.mkdir(exist_ok=True)
    (d / "app.py").write_text(APP)
    return d


def test_measure_writes_profile(env):
    d = _app(env)
    env(None, args=["-m", "importbar", "measure", "--runs", "2", "--cold", "1", "app.py"], cwd=d)
    out = env.last_stdout
    assert "wrote importbar-profile.txt" in out and "cold start" in out
    text = (d / "importbar-profile.txt").read_text()
    assert "session  app.py::<module>" in text
    from importbar._profile import Profile
    prof = Profile.load(str(d / "importbar-profile.txt"))
    assert prof.sessions["app.py::<module>"]["roots"][:2] == ["fb", "fc"]
    assert {"fa", "fb", "fc"} <= set(prof.times["warm"]) and "fc" in prof.times["cold"]
    assert prof.edges["fb"] == ["fa"]
    assert prof.times["warm"]["fc"][0] == pytest.approx(0.30, rel=0.3)
    # the program's real work never ran during measuring
    st = env.store()
    assert st["times"]["warm"]["fc"][0][2] == 2 and st["times"]["cold"]["fc"][0][2] == 1


def test_measure_without_a_session_explains(env):
    d = env.tmp / "plain"
    d.mkdir()
    (d / "plain.py").write_text("import fc\nprint('hi')\n")
    with pytest.raises(AssertionError) as e:
        env(None, args=["-m", "importbar", "measure", "--runs", "1", "plain.py"], cwd=d)
    assert "without starting an importbar session" in str(e.value)


def test_profile_gives_other_machines_a_head_start(env):
    d = _app(env)
    # Author: measures warm starts, and has met cold starts in normal use.
    env(None, args=["-m", "importbar", "measure", "--runs", "3", "--cold", "0", "app.py"], cwd=d)
    for _ in range(2):
        assert env(None, {"FAKE_COLD": "2"}, args=["app.py"], cwd=d)["mode"] == "cold"
    env(None, args=["-m", "importbar", "export", "."], cwd=d)
    prof_text = (d / "importbar-profile.txt").read_text()
    assert "cold=0.9" in prof_text or "cold=" in prof_text

    # Someone else: fresh cache, a machine 1.5x slower, code moved elsewhere.
    other = env.tmp / "elsewhere"
    (d).rename(other)
    fresh = {"IMPORTBAR_CACHE_DIR": str(env.tmp / "cache2"), "FAKE_SLOW": "1.5"}
    res = env(None, fresh, args=["app.py"], cwd=other)
    assert res["profile"] and res["known"]                  # no "learning..." on run one
    assert max(_errors(res, 0.4)) < 0.25
    # Their first cold start (after a reboot): recognised, and predicted from the
    # author's cold/warm ratios plus their machine's speed.
    s2 = env.store(env.tmp / "cache2")
    assert s2["times"]["cold"] == {}
    res = env(None, dict(fresh, FAKE_COLD="2"), args=["app.py"], cwd=other)
    assert res["mode"] == "cold"
    assert max(_errors(res, 0.4)) < 0.25
    # Ignored when asked to.
    assert env(None, dict(fresh, IMPORTBAR_PROFILE=str(env.tmp / "nope")), args=["app.py"],
               cwd=other)["profile"] is False


def test_profile_text_roundtrip_and_damage(tmp_path):
    from importbar._profile import Profile, find, FILENAME
    p = Profile()
    p.times["warm"]["pandas"] = (0.4, 0.3)
    p.times["cold"]["pandas"] = (1.9, 1.7)
    p.times["warm"]["numpy"] = (0.1, 0.1)
    p.edges["pandas"] = ["numpy"]
    p.cpu = {"warm": 0.98, "cold": 0.5}
    p.sessions["my app.py::<module>"] = {"roots": ["pandas"], "other": {"warm": 0.01},
                                         "total": {"warm": 0.5, "cold": 2.0}}
    path = tmp_path / FILENAME
    p.save(str(path))
    q = Profile.load(str(path))
    assert q.times == p.times and q.edges == p.edges and q.cpu == p.cpu
    assert q.sessions == p.sessions
    sub = tmp_path / "src"
    sub.mkdir()
    assert find(str(sub)) == str(path)              # found one directory up
    assert find(str(sub), explicit=False) is None
    path.write_text("importbar-profile 1\ntime pandas warm=abc\n")
    assert Profile.load(str(path)) is None
    path.write_text("something else\n")
    assert Profile.load(str(path)) is None
