"""Turning a jumpy estimate into a bar that moves smoothly.

The estimator says where the bar *should* be; the smoother decides where it
*is*. Rules: never move backwards, never visibly stall, never claim to be done
before the work is (cap below 100% until the session ends), then jump to 100%.
"""

import math
import os
import sys
import threading
from time import perf_counter

CAP = 0.97              # highest the bar goes before the session actually ends
CHASE_TAU = 0.25        # seconds; how quickly the bar closes on the estimate
UNKNOWN_TAU = 3.0       # seconds; asymptotic crawl when there's no history yet


class Progress:
    """What a renderer receives on every frame."""
    __slots__ = ("fraction", "label", "elapsed", "remaining", "done", "desc", "learning",
                 "cold")

    def __init__(self, fraction, label, elapsed, remaining, done, desc, learning, cold=None):
        self.fraction = fraction    # smoothed, 0..1, monotonic
        self.label = label          # package currently importing (or None)
        self.elapsed = elapsed
        self.remaining = remaining  # estimated seconds left (None if unknown)
        self.done = done
        self.desc = desc
        self.learning = learning    # True on a first run with no history
        self.cold = cold            # probability this is a cold start (None if unknown)

    def __repr__(self):
        return "Progress(%.3f, %r, elapsed=%.2f, done=%s)" % (
            self.fraction, self.label, self.elapsed, self.done)


class Smoother:
    def __init__(self):
        self.shown = 0.0
        self.first = True

    def step(self, dt, est):
        shown = self.shown
        if self.first and est.fraction is not None:
            # First frame (drawn after the initial delay): start where we are.
            self.first = False
            self.shown = min(est.fraction, CAP)
            return self.shown
        self.first = False
        if est.fraction is None:
            goal = CAP * (1.0 - math.exp(-est.elapsed / UNKNOWN_TAU))
            shown = max(shown, goal)
        else:
            target = min(est.fraction, CAP)
            if target > shown:
                # Feed-forward: the target itself advances at ~1/total per second, so
                # move at that rate plus a correction; a pure chase would always lag.
                total = est.elapsed + (est.remaining or 0.0)
                base = dt / total if total > 0 else 0.0
                shown += base + (target - shown) * (1.0 - math.exp(-dt / CHASE_TAU))
            else:
                # Estimate fell behind the bar: keep creeping toward the cap,
                # slower the more work we think is left.
                tau = max(3.0, 3.0 * (est.remaining or 0.0))
                shown += (CAP - shown) * (1.0 - math.exp(-dt / tau))
        self.shown = min(max(shown, self.shown), CAP)
        return self.shown


class Display(threading.Thread):
    def __init__(self, session, renderer):
        super().__init__(name="importbar-display", daemon=True)
        self.session = session
        self.renderer = renderer
        self.smoother = Smoother()
        self._stop_evt = threading.Event()
        self.rendered = False
        self.last = None

    def run(self):
        s = self.session
        interval = 1.0 / max(s.fps, 1)
        if self._stop_evt.wait(s.delay):     # quick session: never draw anything
            return
        last_t = perf_counter()
        while not self._stop_evt.is_set():
            now = perf_counter()
            est = s.estimate(now)
            frac = self.smoother.step(now - last_t, est)
            last_t = now
            self.last = Progress(frac, est.label, est.elapsed, est.remaining,
                                 False, s.desc, est.fraction is None, est.cold)
            try:
                self.renderer(self.last)
                self.rendered = True
            except Exception:
                return   # a broken renderer must never break the import
            self._stop_evt.wait(interval)

    def finish(self, elapsed):
        self._stop_evt.set()
        self.join()
        if self.rendered:
            final = Progress(1.0, None, elapsed, 0.0, True, self.session.desc, False,
                             getattr(self.session, "cold", None))
            try:
                self.renderer(final)
            except Exception:
                pass
        close = getattr(self.renderer, "close", None)
        if close is not None:
            try:
                close(self.rendered)
            except Exception:
                pass


# --------------------------------------------------------------------- renderers

class TerminalRenderer:
    """Single-line bar on stderr using carriage returns (works in terminals and Jupyter)."""

    def __init__(self, stream=None, leave=False, width=None):
        self.stream = stream or sys.stderr
        self.leave = leave
        self.width = width
        self._last_len = 0
        enc = getattr(self.stream, "encoding", None) or "ascii"
        try:
            "█▏▕…".encode(enc)
            self.blocks = " ▏▎▍▌▋▊▉█"
            self.ell = "…"
        except (UnicodeEncodeError, LookupError):
            self.blocks = " #"
            self.ell = "..."

    def _cols(self):
        if self.width:
            return self.width
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (OSError, AttributeError, ValueError):
            return 80

    def _bar(self, frac, n):
        b = self.blocks
        k = len(b) - 1
        cells = frac * n
        full = int(cells)
        part = int((cells - full) * k)
        s = b[-1] * full
        if full < n:
            s += b[part] + b[0] * (n - full - 1)
        return s

    def __call__(self, p):
        cols = self._cols()
        pct = "%3d%%" % int(p.fraction * 100)
        if p.done:
            tail = "done in %.1fs" % p.elapsed
        elif p.learning:
            tail = "%.1fs (learning%s)" % (p.elapsed, self.ell)
        else:
            tail = "%.1fs, ~%.1fs left" % (p.elapsed, max(p.remaining or 0.0, 0.0))
            if p.cold is not None and p.cold >= 0.7:
                tail += " (cold start)"
        label = p.label or ""
        bar_w = max(10, min(30, cols // 3))
        head = "%s %s |%s| " % (p.desc, pct, self._bar(p.fraction, bar_w))
        room = cols - 1 - len(head) - len(tail) - 2
        if label and room > 4:
            if len(label) > room:
                label = label[: room - len(self.ell)] + self.ell
            line = "%s%s  %s" % (head, tail, label)
        else:
            line = head + tail
        line = line[: cols - 1]
        pad = " " * max(self._last_len - len(line), 0)
        self._last_len = len(line)
        try:
            self.stream.write("\r" + line + pad)
            self.stream.flush()
        except Exception:
            pass

    def close(self, rendered):
        if not rendered:
            return
        try:
            if self.leave:
                self.stream.write("\n")
            else:
                self.stream.write("\r" + " " * self._last_len + "\r")
            self.stream.flush()
        except Exception:
            pass


class WindowsRenderer:
    """Small native Windows progress window, using only ctypes.

    Use with importbar.progress(renderer=WindowsRenderer()).
    Created lazily on the first frame, so short sessions never open a window.
    Closing the window only dismisses it; it does not cancel the work. With
    leave=True it stays open until dismissed or the Python process exits.
    Each instance is intended for one display session. finish_delay holds the
    completed bar briefly before close() returns (seconds; zero disables it).
    """

    def __init__(self, leave=False, finish_delay=0.15):
        if sys.platform != "win32":
            raise OSError("WindowsRenderer requires Windows")
        self.leave = leave
        self.finish_delay = float(finish_delay)
        if not math.isfinite(self.finish_delay) or self.finish_delay < 0:
            raise ValueError("finish_delay must be a finite, non-negative number")
        self._lock = threading.Lock()
        self._thread = None
        self._latest = None
        self._closed = False

    def __call__(self, p):
        with self._lock:
            if self._closed:
                return
            self._latest = p
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="importbar-window", daemon=True)
                self._thread.start()

    def close(self, rendered):
        with self._lock:
            if self.leave and rendered:
                return
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _run(self):
        # All Win32 calls, including destruction, belong to this thread.
        hwnd = None
        user32 = None
        registered = False
        try:
            import ctypes as ct
            from ctypes import wintypes as wt

            user32 = ct.WinDLL("user32", use_last_error=True)
            comctl32 = ct.WinDLL("comctl32", use_last_error=True)
            kernel32 = ct.WinDLL("kernel32", use_last_error=True)
            gdi32 = ct.WinDLL("gdi32", use_last_error=True)
            uxtheme = ct.WinDLL("uxtheme", use_last_error=True)

            def bind(dll, name, result, *args):
                fn = getattr(dll, name)
                fn.restype = result
                fn.argtypes = args
                return fn

            # Pointer-sized message values and handles also work on 64-bit Python.
            create = bind(user32, "CreateWindowExW", wt.HWND,
                          wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                          ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                          wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID)
            send = bind(user32, "SendMessageW", ct.c_ssize_t,
                        wt.HWND, wt.UINT, ct.c_size_t, ct.c_ssize_t)
            set_text = bind(user32, "SetWindowTextW", wt.BOOL,
                            wt.HWND, wt.LPCWSTR)
            show = bind(user32, "ShowWindow", wt.BOOL, wt.HWND, ct.c_int)
            is_window = bind(user32, "IsWindow", wt.BOOL, wt.HWND)
            destroy = bind(user32, "DestroyWindow", wt.BOOL, wt.HWND)
            get_message = bind(user32, "GetMessageW", ct.c_int,
                               ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT)
            translate = bind(user32, "TranslateMessage", wt.BOOL,
                             ct.POINTER(wt.MSG))
            dispatch = bind(user32, "DispatchMessageW", ct.c_ssize_t,
                            ct.POINTER(wt.MSG))
            adjust = bind(user32, "AdjustWindowRectEx", wt.BOOL,
                          ct.POINTER(wt.RECT), wt.DWORD, wt.BOOL, wt.DWORD)
            default_proc = bind(user32, "DefWindowProcW", ct.c_ssize_t,
                                wt.HWND, wt.UINT, ct.c_size_t, ct.c_ssize_t)
            post_quit = bind(user32, "PostQuitMessage", None, ct.c_int)
            set_timer = bind(user32, "SetTimer", ct.c_size_t,
                             wt.HWND, ct.c_size_t, wt.UINT, wt.LPVOID)
            update_window = bind(user32, "UpdateWindow", wt.BOOL, wt.HWND)
            module_handle = bind(kernel32, "GetModuleHandleW", wt.HINSTANCE,
                                 wt.LPCWSTR)
            stock_object = bind(gdi32, "GetStockObject", wt.HANDLE, ct.c_int)
            set_theme = bind(uxtheme, "SetWindowTheme", ct.c_long,
                             wt.HWND, wt.LPCWSTR, wt.LPCWSTR)
            load_cursor = bind(user32, "LoadCursorW", wt.HANDLE,
                               wt.HINSTANCE, wt.LPVOID)

            wndproc_type = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT,
                                         ct.c_size_t, ct.c_ssize_t)

            class WindowClass(ct.Structure):
                _fields_ = [("style", wt.UINT), ("proc", wndproc_type),
                            ("class_extra", ct.c_int), ("window_extra", ct.c_int),
                            ("instance", wt.HINSTANCE), ("icon", wt.HANDLE),
                            ("cursor", wt.HANDLE), ("background", wt.HBRUSH),
                            ("menu", wt.LPCWSTR), ("name", wt.LPCWSTR)]

            register = bind(user32, "RegisterClassW", wt.WORD,
                            ct.POINTER(WindowClass))
            unregister = bind(user32, "UnregisterClassW", wt.BOOL,
                              wt.LPCWSTR, wt.HINSTANCE)

            class InitControls(ct.Structure):
                _fields_ = [("size", wt.DWORD), ("classes", wt.DWORD)]

            init = bind(comctl32, "InitCommonControlsEx", wt.BOOL,
                        ct.POINTER(InitControls))
            if not init(ct.byref(InitControls(ct.sizeof(InitControls), 0x20))):
                raise ct.WinError(ct.get_last_error())  # ICC_PROGRESS_CLASS

            previous = None
            last_title = last_label = None
            last_position = None
            completed_at = None
            label = bar = None

            def draw_frame(window):
                nonlocal previous, last_title, last_label, last_position, completed_at
                with self._lock:
                    p, closed = self._latest, self._closed
                if p is not None and p is not previous:
                    fraction = 1.0 if p.done else min(1.0, max(0.0, p.fraction))
                    # Keep the title stable; repaint only genuinely changed text.
                    title = p.desc or "Importing"
                    text = ("Done in %.1fs" % p.elapsed) if p.done else (
                        p.label or p.desc or "Importing...")
                    if p.learning and not p.done:
                        text += " (learning...)"
                    if title != last_title:
                        set_text(window, title)
                        last_title = title
                    if text != last_label:
                        set_text(label, text)
                        last_label = text
                    position = round(fraction * 1000)
                    if position != last_position:
                        send(bar, 0x0402, position, 0)  # PBM_SETPOS
                        last_position = position
                    if p.done and completed_at is None:
                        # Paint the final state before starting the short hold.
                        update_window(bar)
                        update_window(label)
                        completed_at = perf_counter()
                    previous = p
                if closed and (completed_at is None or
                               perf_counter() - completed_at >= self.finish_delay):
                    destroy(window)

            @wndproc_type
            def window_proc(window, message, wparam, lparam):
                if message == 0x0113 and wparam == 1:  # WM_TIMER
                    try:
                        draw_frame(window)
                    except Exception:
                        destroy(window)
                    return 0
                if message == 0x0002:  # WM_DESTROY
                    post_quit(0)
                    return 0
                # Standard hit testing and caption dragging belong to Windows.
                return default_proc(window, message, wparam, lparam)

            instance = module_handle(None)
            class_name = "ImportbarWindow_%x" % id(self)
            wc = WindowClass()
            wc.proc = window_proc  # Keep callback and class alive until unregister.
            wc.instance = instance
            wc.cursor = load_cursor(None, 32512)  # IDC_ARROW
            wc.background = 16                   # COLOR_BTNFACE + 1
            wc.name = class_name
            if not register(ct.byref(wc)):
                raise ct.WinError(ct.get_last_error())
            registered = True

            # WS_CAPTION | WS_SYSMENU | WS_CLIPCHILDREN. The parent must not
            # erase over the label and progress control when it repaints.
            style = 0x00C00000 | 0x00080000 | 0x02000000
            ex_style = 0x00000080            # WS_EX_TOOLWINDOW
            rect = wt.RECT(0, 0, 360, 76)
            if not adjust(ct.byref(rect), style, False, ex_style):
                raise ct.WinError(ct.get_last_error())
            hwnd = create(ex_style, class_name, "Importing", style,
                          -2147483648, -2147483648,
                          rect.right - rect.left, rect.bottom - rect.top,
                          None, None, instance, None)
            if not hwnd:
                raise ct.WinError(ct.get_last_error())
            child_style = 0x40000000 | 0x10000000  # WS_CHILD | WS_VISIBLE
            label = create(0, "STATIC", "", child_style | 0x4000 | 0x80,
                           12, 10, 336, 20, hwnd, None, instance, None)
            bar = create(0, "msctls_progress32", "", child_style | 0x01,
                         12, 38, 336, 20, hwnd, None, instance, None)
            if not label or not bar:
                raise ct.WinError(ct.get_last_error())
            send(label, 0x0030, stock_object(17), 0)  # WM_SETFONT / DEFAULT_GUI_FONT
            # The smoother already animates progress. Use the plain native bar
            # so themed Windows animations don't add a second layer of lag.
            if set_theme(bar, "", "") < 0:
                raise OSError("Could not disable progress-bar theme animation")
            send(bar, 0x0406, 0, 1000)  # PBM_SETRANGE32
            show(hwnd, 4)              # SW_SHOWNOACTIVATE
            draw_frame(hwnd)
            # A window timer keeps frames and shutdown working inside Windows'
            # nested message loop while the user is dragging the title bar.
            if is_window(hwnd) and not set_timer(hwnd, 1, 30, None):
                raise ct.WinError(ct.get_last_error())
            msg = wt.MSG()
            while is_window(hwnd):
                status = get_message(ct.byref(msg), None, 0, 0)
                if status <= 0:  # WM_QUIT or message retrieval failure
                    break
                translate(ct.byref(msg))
                dispatch(ct.byref(msg))
        except Exception:
            pass  # GUI failures must never interrupt the import session.
        finally:
            if hwnd and user32 is not None and user32.IsWindow(hwnd):
                user32.DestroyWindow(hwnd)
            if registered:
                unregister(class_name, instance)
            with self._lock:
                self._closed = True


def _interactive_stderr():
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except Exception:
        return False


def make_display(session, display, renderer):
    if display is False:
        return None
    if renderer is None:
        if display is None and not _interactive_stderr():
            return None
        if os.environ.get("IMPORTBAR_NO_DISPLAY"):
            return None
        renderer = TerminalRenderer(leave=session.leave)
    return Display(session, renderer)
