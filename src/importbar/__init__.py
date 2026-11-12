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

Timings are learned per environment and stored in a small JSON cache, so the
first run shows an indeterminate "learning" bar and later runs are accurate.
Set IMPORTBAR_DISABLE=1 to turn everything into a plain import.
"""

import importlib
import os
import sys

from ._display import Progress, TerminalRenderer
from ._session import Session
from ._store import Store

__all__ = ["load", "progress", "start", "Progress", "TerminalRenderer", "Session", "Store"]
__version__ = "0.1.0"


def _caller_label(depth=2):
    try:
        f = sys._getframe(depth)
        code = f.f_code
        return "%s::%s" % (os.path.abspath(code.co_filename), code.co_name)
    except (AttributeError, ValueError):
        return "default"


def progress(desc="Importing", *, label=None, display=None, renderer=None,
             delay=0.2, fps=20, leave=False):
    """Context manager showing a bar for whatever is imported inside the block.

    The block's imports are learned from previous runs, keyed by ``label``
    (default: calling file and function).

    display:  None = auto (only on a terminal / Jupyter), True = always, False = never.
    renderer: callable receiving a :class:`Progress` each frame; may define
              ``close(rendered)``. Use it to drive rich, tqdm, or a GUI splash.
    delay:    seconds before anything is drawn (fast imports show nothing).
    """
    return Session(label or _caller_label(), display=display, desc=desc,
                   renderer=renderer, delay=delay, fps=fps, leave=leave)


def start(desc="Importing", **kwargs):
    """Start a session now; call ``.stop()`` on the result when loading is done."""
    kwargs.setdefault("label", _caller_label())
    return progress(desc, **kwargs).start()


def load(*names, desc="Importing", label=None, display=None, renderer=None,
         delay=0.2, fps=20, leave=False):
    """Import ``names`` with a progress bar; returns the module (or a tuple of them)."""
    if len(names) == 1 and isinstance(names[0], (list, tuple)):
        names = tuple(names[0])
    roots = [n.partition(".")[0] for n in names]
    sess = Session(label or "load:" + ",".join(names), roots=roots, display=display,
                   desc=desc, renderer=renderer, delay=delay, fps=fps, leave=leave)
    with sess:
        mods = tuple(importlib.import_module(n) for n in names)
    return mods[0] if len(mods) == 1 else mods
