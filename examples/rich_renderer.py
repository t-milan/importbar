"""Drive a `rich` progress bar (or anything else) from importbar.

A renderer is any callable taking an ``importbar.Progress``; an optional
``close(rendered)`` method is called once at the end. It runs on importbar's
display thread, so for GUI toolkits (Tk, Qt) hand the value to the UI thread
instead of touching widgets directly.

Note: importing rich costs ~30-50 ms itself; the built-in renderer costs ~0.
"""

import importbar
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class RichRenderer:
    def __init__(self):
        self.bar = Progress(TextColumn("{task.description}"), BarColumn(),
                            TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn(),
                            transient=True, auto_refresh=False)
        self.task = None

    def __call__(self, p):
        if self.task is None:
            self.bar.start()
            self.task = self.bar.add_task(p.desc, total=1.0)
        label = "%s [dim]%s" % (p.desc, p.label or "")
        self.bar.update(self.task, completed=p.fraction, description=label)
        self.bar.refresh()

    def close(self, rendered):
        if self.task is not None:
            self.bar.stop()


if __name__ == "__main__":
    with importbar.progress("Loading", renderer=RichRenderer()):
        import pandas  # noqa: F401
        import scipy.stats  # noqa: F401
        import pylinac
