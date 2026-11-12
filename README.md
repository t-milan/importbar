# importbar

A progress bar for slow Python imports that **learns** how long your imports
take, so it moves smoothly and finishes when the imports do. It tells cold
starts (first run after a reboot) from warm ones, and you can ship your
measured timings with your code so other people's first run is already
accurate. It has no dependencies, costs about 3 ms to import, and adds no
measurable time to the imports it watches.

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

Without a profile (see below), the first run shows an indeterminate
"learning…" bar. Every later run uses the learned timings.

Options for `progress()` and `load()`:

| option | default | meaning |
|---|---|---|
| `desc` | `"Importing"` | text before the bar |
| `display` | `None` | `None`: draw only on a terminal or in Jupyter. `True`: always. `False`: never (still learns) |
| `renderer` | built-in | any callable taking a `Progress`; may define `close(rendered)` (see `examples/rich_renderer.py`) |
| `delay` | `0.2` | seconds before drawing, so fast imports show nothing at all |
| `leave` | `False` | keep the final "done in 1.3s" line instead of clearing it |
| `label` | caller's file + function | which history to use for a `with` block |
| `profile` | `None` | `None`: use `importbar-profile.txt` next to the calling file (or one folder up). A path: that file or folder. `False`: no profile |

Environment variables: `IMPORTBAR_DISABLE=1` (plain imports, no hook, nothing
written), `IMPORTBAR_NO_DISPLAY=1`, `IMPORTBAR_CACHE_DIR`, `IMPORTBAR_PROFILE`
(path to a profile), `IMPORTBAR_ASSUME=cold|warm` (skip detection, treat
every run as that kind).

CLI: `python -m importbar report` lists your slowest packages, warm and cold
(handy in its own right). `measure` and `export` create profiles (below).
`try numpy pandas` previews the bar, `clear` resets the history, and `path`
shows where the history is stored.

## Cold and warm starts

The first start after a reboot can take several times longer than the next
one, because nothing is in the OS file cache yet and the virus scanner checks
every DLL again. On a fast SSD the difference is about 1.2×. On a laptop with
a hard disk or aggressive antivirus it can be 10× or more, and it isn't the
same for every package: libraries made of many files or large DLLs suffer
most. So importbar keeps two sets of timings, **warm** and **cold**, and
works out which kind of start it is in. The OS won't tell a program whether
its files are cached, so it combines three clues:

1. **Before the run:** what kind of start this is. A reboot since the last
   run, hours of inactivity, or the very first run in this environment each
   start with a different chance of being cold. The chances are learned per
   machine (`report` shows the counts).
2. **During the run:** the importing thread's **CPU time ÷ wall time**. A
   warm import is CPU-bound, so this is close to 1 on any machine, fast or
   slow. A cold import spends much of its time waiting for the disk, which
   costs wall time but no CPU time. This is the clue that tells "slower
   machine" apart from "cold start", and it costs a clock read every 10 ms.
3. **During the run:** whether the packages that have finished match the
   warm or the cold timings better, allowing for machine speed.

The bar shows a mix of the warm and cold estimates, weighted by how likely a
cold start is, and that weight sharpens within the first packages. Each
estimate keeps its own live speed correction, so even when the cold/warm call
is wrong the estimate still follows reality. After the run, the timings are
learned into whichever set they belong to. When a cold start is likely, the
terminal bar says `(cold start)`.

## Sharing your timings: `importbar-profile.txt`

Your machine's history lives in your cache folder, so on anyone else's
machine importbar starts from nothing. To fix that, measure once and ship the
result with your code:

```
python -m importbar measure app.py            # or: measure -m mypackage
```

This runs your program several times in fresh interpreters, stopping each
run as soon as its importbar session ends. Your program's actual work (a GUI,
a server) never runs. It records warm starts. On Linux it also records cold
starts, by evicting the files the program loaded from the OS cache first
(`posix_fadvise`, no admin rights needed). Then it writes
**`importbar-profile.txt` next to the file that calls importbar**. Options go
before the script (`measure --runs 8 --cold 3 app.py`); anything after the
script is passed to it.

Send that file along with your code (commit it, or add it to your package
data) and you're done. Nothing else needs configuring on the other machine:
`importbar.progress()` looks for the file next to the code that calls it. A
recipient's very first run then shows a real estimate instead of
"learning…", for warm and for cold starts:

- **What transfers** is the shape: which packages your program pulls in, how
  they depend on each other, how long each takes compared with the others,
  and how much slower each gets when cold.
- **Machine speed** is corrected for automatically. Once the first packages
  have loaded, everything left is rescaled by how much faster or slower this
  machine is than yours. Once the recipient has their own numbers for some
  packages, the ratio also carries over to the rest.
- **Their own history wins.** As soon as their machine has measured a
  package, those numbers replace yours. The profile is read-only: nothing is
  ever written next to the code. The profile is just a plain text file
  (diffable, reviewable) that parses in under a millisecond.

Windows and macOS have no unprivileged way to empty the file cache, so
`measure` can't simulate cold starts there. importbar learns them anyway: the
first run after each reboot is detected as cold. Once you've rebooted and run
your program a couple of times, `python -m importbar export .` refreshes the
profile with everything this machine has learned, cold starts included.

## How it works

The package has three layers, each in its own module:

```
src/importbar/
  _hook.py      measurement: times each module's execution, nearly free
  _store.py     history: learned per-package timings, warm and cold, one cache file per environment
  _profile.py   sharing: importbar-profile.txt, read next to the code
  _model.py     expectations from history + profile; is this a cold start?
  _session.py   estimate: plans the session, predicts remaining time
  _display.py   display: smoothing plus the terminal renderer, on a daemon thread
  _measure.py   `measure` / `export`: profile your startup in fresh interpreters
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
machine is recorded as "slow run" rather than "every package got slower".
Everything is kept separately for warm and cold starts. The cache uses
`marshal`, because `import json` would pull in `re` and `enum` and cost about
7 ms.

**Expectations (`_model.py`).** For each package and mode, the most
trustworthy source wins: this machine's own history; this machine's other
mode times the profile's cold/warm ratio for that package; the profile times
this machine's measured speed relative to the author's; and finally the other
mode times a typical cold/warm ratio. Each source carries an uncertainty that
weights it as evidence when telling cold from warm.

**Estimate (`_session.py`).** At the start, the plan is the dependency
closure of the session's root imports, minus anything already in
`sys.modules`. Import `numpy` earlier and the bar for `pandas` expects less
work. During the run, a **speed factor** per mode compares finished packages
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
(add `--cold` on Linux to evict the file cache before every run).

On the VM used for development, a cold start of pandas + scipy + matplotlib
was only about 1.1× slower than a warm one, so the cold-start logic is tested
with fake packages that wait the way a slow disk does (3× slower, two thirds
of the time spent waiting). In those tests a cold run is recognised the first
time one occurs, before any cold timings exist. Once both kinds are known,
both are predicted within 15%, and the cold/warm call is settled about a tenth
of the way into the run. A fresh machine that is 1.5× slower and has only a profile
is predicted within 25%, for warm starts and for its first cold one.

## Limitations

- **A single long C call freezes the bar.** The display thread needs the GIL
  to redraw, and a long C-level initialisation that holds it (some CUDA or
  driver setup) blocks redraws until it returns. The estimate is still right
  afterwards.
- **Without a profile, the first run is a guess.** Nothing is known yet, so
  the bar is indeterminate. Ship an `importbar-profile.txt` to avoid this.
- **Cold detection is inferred, not observed.** CPU contention (another
  program hogging every core) also lowers the CPU/wall ratio and can look
  like a cold start. The estimate still adapts through the speed factor. On
  Windows, "Fast Startup" shutdowns don't reset the uptime clock, so those
  starts are caught by the CPU/wall clue rather than the reboot check. On
  Python 3.13+ on Windows the uptime clock may also skip time asleep. Waking
  from a long sleep can then look like a reboot, which only makes the first
  estimate more cautious.
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
nested sessions, cold-start detection and learning (simulated with waits that
use no CPU), `measure` and `export` end to end, and a profile used on a
fresh, slower machine.
