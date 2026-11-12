"""importbar: a learned, smooth progress bar for slow Python imports.

    import importbar

    # 1. Explicit list of imports
    np, pd, torch = importbar.load("numpy", "pandas", "torch")

    # 2. Wrap existing import statements
    with importbar.progress():
        import numpy as np
        import pandas as pd
        import torch

    # 3. Imperative, e.g. around an app's whole startup
    bar = importbar.start("Starting")
    ...
    bar.stop()

Timings are learned per environment, separately for warm and cold starts,
and stored in a small cache file. To give other machines a head start, run
``python -m importbar measure your_script.py``: it writes
``importbar-profile.txt`` next to your code, and importbar picks it up
automatically wherever the code runs.
Set IMPORTBAR_DISABLE=1 to turn everything into a plain import.
"""

import importlib
import os
import sys

from ._display import Progress, TerminalRenderer
from ._session import Session
from ._store import Store, bundle_dir

__all__ = ["load", "progress", "start", "Progress", "TerminalRenderer", "Session", "Store"]
__version__ = "0.2.0"


def _caller(depth=2):
    """(label, directory) of the code calling into importbar."""
    try:
        f = sys._getframe(depth)
        code = f.f_code
        bundle = bundle_dir()
        if bundle and not os.path.isabs(code.co_filename):
            # Frozen apps (PyInstaller) name bundled code relative to the bundle,
            # which for a one-file build is a new temp folder on every launch.
            # Label by that relative path, so the history and a shipped profile
            # keep matching, and look for the profile inside the bundle.
            rel = code.co_filename.replace("\\", "/")
            return "%s::%s" % (rel, code.co_name), os.path.join(bundle, os.path.dirname(rel))
        path = os.path.abspath(code.co_filename)
        # Notebooks and `python -c` have no file: look for a profile in the cwd.
        d = os.path.dirname(path) if os.path.isfile(path) else os.getcwd()
        return "%s::%s" % (path, code.co_name), d
    except (AttributeError, ValueError):
        return "default", None


def progress(desc="Importing", *, label=None, display=None, renderer=None,
             delay=0.2, fps=20, leave=False, profile=None, _depth=2):
    """Context manager showing a bar for whatever is imported inside the block.

    The block's imports are learned from previous runs, keyed by ``label``
    (default: calling file and function).

    display:  None = auto (only on a terminal / Jupyter), True = always, False = never.
    renderer: callable receiving a :class:`Progress` each frame; may define
              ``close(rendered)``. Use it to drive rich, tqdm, or a GUI splash.
    delay:    seconds before anything is drawn (fast imports show nothing).
    profile:  shared timings to start from. None = look for importbar-profile.txt
              next to the calling file (or one directory up); a path = that
              file or directory; False = don't use one.
    """
    auto_label, origin = _caller(_depth)
    return Session(label or auto_label, display=display, desc=desc,
                   renderer=renderer, delay=delay, fps=fps, leave=leave,
                   profile=profile, origin=origin)


def start(desc="Importing", **kwargs):
    """Start a session now; call ``.stop()`` on the result when loading is done."""
    return progress(desc, _depth=3, **kwargs).start()


def load(*names, desc="Importing", label=None, display=None, renderer=None,
         delay=0.2, fps=20, leave=False, profile=None):
    """Import ``names`` with a progress bar; returns the module (or a tuple of them)."""
    if len(names) == 1 and isinstance(names[0], (list, tuple)):
        names = tuple(names[0])
    roots = [n.partition(".")[0] for n in names]
    _, origin = _caller()
    sess = Session(label or "load:" + ",".join(names), roots=roots, display=display,
                   desc=desc, renderer=renderer, delay=delay, fps=fps, leave=leave,
                   profile=profile, origin=origin)
    with sess:
        mods = tuple(importlib.import_module(n) for n in names)
    return mods[0] if len(mods) == 1 else mods
