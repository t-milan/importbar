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
import time as _t
def _busy(sec):
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

    def run(code, extra_env=None):
        e = dict(base_env, **(extra_env or {}))
        out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=e,
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise AssertionError(out.stderr)
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else None

    def store():
        files = list(cache.glob("timings-*.bin")) if cache.exists() else []
        return marshal.loads(files[0].read_bytes()) if files else None

    run.store = store
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
    p = {k: v for k, v in st["packages"].items() if k in ("fa", "fb", "fc")}
    import math
    exp = {k: math.exp(v[0]) for k, v in p.items()}
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
    assert "fc" in env.store()["packages"]


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
