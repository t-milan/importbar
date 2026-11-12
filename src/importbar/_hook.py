"""Import instrumentation.

A meta-path finder is placed at the front of ``sys.meta_path`` *only while a
session is active*. It never finds anything itself: it asks the other finders,
and when one returns a spec it temporarily patches ``exec_module`` on that
spec's loader *instance* so the module's execution can be timed. The patch
removes itself on first use, so loaders keep their identity and class
(``isinstance`` checks, ``importlib.resources``, ``inspect.getsource`` etc. are
unaffected) and nothing is left behind once the session ends.
"""

import sys
import threading

ACTIVE = None  # the running Session, or None
_tls = threading.local()


def _make_wrapper(loader_dict, orig):
    def exec_module(module):
        # Self-removal: restore the loader to its pristine state before running.
        if loader_dict.get("exec_module") is exec_module:
            loader_dict.pop("exec_module", None)
        session = ACTIVE
        if session is None:
            return orig(module)
        name = module.__name__
        session._enter(name)
        try:
            return orig(module)
        finally:
            session._exit(name)

    exec_module.__importbar__ = True
    return exec_module


def _instrument(loader):
    if loader is None or isinstance(loader, type):
        # BuiltinImporter / FrozenImporter are classes: patching them would be
        # global, and those modules are fast anyway.
        return
    try:
        d = loader.__dict__
    except AttributeError:  # __slots__ loaders
        return
    current = d.get("exec_module")
    if current is not None and getattr(current, "__importbar__", False):
        return
    orig = getattr(loader, "exec_module", None)
    if orig is None:
        return
    try:
        loader.exec_module = _make_wrapper(d, orig)
    except Exception:
        pass


class _TimingFinder:
    """Delegating finder; see module docstring."""

    def find_spec(self, fullname, path=None, target=None):
        if getattr(_tls, "busy", False):
            return None
        _tls.busy = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                fs = getattr(finder, "find_spec", None)
                if fs is None:
                    continue
                spec = fs(fullname, path, target)
                if spec is not None:
                    break
        finally:
            _tls.busy = False
        if spec is not None:
            _instrument(spec.loader)
        return spec

    # Make the finder recognisable in sys.meta_path dumps.
    def __repr__(self):
        return "<importbar timing finder>"


_FINDER = _TimingFinder()


def install(session):
    global ACTIVE
    ACTIVE = session
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def uninstall():
    global ACTIVE
    ACTIVE = None
    try:
        while True:
            sys.meta_path.remove(_FINDER)
    except ValueError:
        pass
