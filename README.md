# importbar

A progress bar for slow Python imports that **learns** how long your imports
take, so it moves smoothly and finishes when the imports do. It has no
dependencies, costs about 2.5 ms to import, and adds no measurable time to the
imports it watches.

```
Loading  57% |█████████████████             | 0.7s, ~0.6s left  scipy
```

## Drop-in usage

```python
import importbar

# 1. Wrap the import statements you already have
with importbar.progress("Loading"):
    import pandas as pd
    import scipy.stats
    import sklearn.ensemble
    import matplotlib.pyplot as plt

# 2. Or name the modules
pd, plt = importbar.load("pandas", "matplotlib.pyplot")

# 3. Or bracket a whole startup whose imports are spread across modules
bar = importbar.start("Starting")
from myapp.cli import main        # everything this pulls in is covered
bar.stop()
main()
```

Put it in your **entry point** (`__main__.py`, a CLI `main()`, a notebook's
first cell), not inside a library: code that merely imports your library
shouldn't get a progress bar.

The first run shows an indeterminate "learning…" bar. Every later run uses
the learned timings.

Options for `progress()` and `load()`:

| option | default | meaning |
|---|---|---|
| `desc` | `"Importing"` | text before the bar |
| `display` | `None` | `None`: draw only on a terminal or in Jupyter. `True`: always. `False`: never (still learns) |
| `renderer` | built-in | any callable taking a `Progress`; may define `close(rendered)` (see `examples/rich_renderer.py`) |
| `delay` | `0.2` | seconds before drawing, so fast imports show nothing at all |
| `leave` | `False` | keep the final "done in 1.3s" line instead of clearing it |
| `label` | caller's file + function | which history to use for a `with` block |

Environment variables: `IMPORTBAR_DISABLE=1` (plain imports, no hook, nothing
written), `IMPORTBAR_NO_DISPLAY=1`, `IMPORTBAR_CACHE_DIR`.

CLI: `python -m importbar report` lists your slowest packages (handy in its
own right). `try numpy pandas` previews the bar, `clear` resets the history,
and `path` shows where the history is stored.

## How it works

The package has three layers, each in its own module:

```
src/importbar/
  _hook.py      measurement: times each module's execution, nearly free
  _store.py     history: learned per-package timings, one cache file per environment
  _session.py   estimate: plans the session, predicts remaining time
  _display.py   display: smoothing plus the terminal renderer, on a daemon thread
  __init__.py   public API: progress(), load(), start()
  __main__.py   CLI
```

**Measurement (`_hook.py`).** While a session is active, a finder sits at
the front of `sys.meta_path`. It doesn't find anything itself; it asks the
real finders and then patches `exec_module` on the returned loader *instance*,
so the module's execution gets timed. The patch removes itself the first
time it runs. Loaders keep their class and identity, so `isinstance` checks,
`importlib.resources` and `inspect.getsource` all keep working. When the
session ends the finder is removed, so outside a session there is no
overhead at all.

**What gets counted.** Each package is charged its *self time*: the time its
modules spent executing, minus time spent in other packages they imported.
Add up the self times of every package, plus non-import code in the block,
and you get the exact wall-clock time, which is what the bar represents. From
the nesting it also learns a dependency graph (`seaborn` → `matplotlib` →
`numpy` …).

**History (`_store.py`).** Durations are treated as log-normal and tracked
with an exponentially weighted mean and variance in log space. Each new
observation is clipped to ±3σ, so one freak run can't wreck the averages. Each
package's time is split into an *eager* part (while its `__init__` runs) and a
*lazy* part (submodules imported later). The split matters: `scipy`'s
`__init__` is quick, and `scipy.stats` does the real work afterwards. Timings
are stored after dividing by that run's overall speed, so a run on a busy
machine is recorded as "slow run" rather than "every package got slower". The
cache uses `marshal`, because `import json` would pull in `re` and `enum` and
cost about 7 ms.

**Estimate (`_session.py`).** At the start, the plan is the dependency
closure of the session's root imports, minus anything already in
`sys.modules`. Import `numpy` earlier and the bar for `pandas` expects less
work. During the run, a global **speed factor** compares finished packages
with their history and rescales everything that's left. This one correction
fixes most of the error on a busy machine or a cold disk. A package that
overruns its estimate has its remaining time grow, rather than drop to zero.
Each estimate is O(planned packages).

**Display (`_display.py`).** The smoother never goes backwards, never stalls,
stays at or below 97% until the imports really finish, then jumps to 100%. It
moves at the predicted rate (1/total per second) plus a correction toward the
estimate. A plain exponential chase would trail a moving target by about 20%.

## Measured (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn; 1,848 modules)

| | result |
|---|---|
| hook overhead | 1.219 s without the hook vs 1.223 s with it (medians of 6 runs; within noise) |
| fixed cost | `import importbar` 2.6 ms, session start 0.4 ms, stop + save 1.3 ms |
| predicted total vs actual, normal runs | 2–6 % mean error during the run |
| same, 3× CPU contention | 5–15 % mean (the speed factor adapts during the run) |
| first normal run after contention | 7 % (the history wasn't skewed) |

Check it on your own imports: `python tools/accuracy.py pandas torch ...`

## Limitations

- **A single long C call freezes the bar.** The display thread needs the GIL
  to redraw, and a long C-level initialisation that holds it (some CUDA or
  driver setup) blocks redraws until it returns. The estimate is still right
  afterwards.
- **The first run is a guess.** Nothing is known yet, so the bar is
  indeterminate. You could ship default timings with your app to avoid this.
- **Imports in parallel threads** are timed per thread, but the remaining
  time assumes they run one after another, so the estimate is pessimistic.
  (The GIL and the import lock make parallel imports rarely worth it.)
- **History is per environment** (`sys.prefix` and Python version). A
  package upgrade that changes timing is learned over the next few runs.

## Tests

```
pip install pytest && python -m pytest
```

The tests build fake packages that do timed CPU work, with dependencies
between them, and run each scenario in a fresh interpreter. They cover
transparency, learning, accuracy and smoothness, preloaded dependencies,
dependencies that stop being imported, exceptions inside imports, disabling,
and nested sessions.
