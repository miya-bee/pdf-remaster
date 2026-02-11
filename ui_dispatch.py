# -*- coding: utf-8 -*-
from __future__ import annotations

"""ui_dispatch.py

Tkinter is NOT thread-safe.
Any widget method call (including .after/.winfo_exists/.configure/.update/.destroy)
MUST happen on the UI (Tk) thread.

This module provides a single, root-attached dispatcher and small helpers so the
rest of the codebase can safely request UI work from worker threads.

Key design goals
- One source of truth: UI work is routed through the default UiDispatcher.
- Safety first: when the dispatcher is not available, we DO NOT fall back to
  calling Tk from a worker thread (that would reintroduce thread-safety bugs).
  Instead we log a throttled warning once and drop the request.
- Minimal behavior change on the UI thread.

"""

import queue
import threading
from typing import Callable, Optional, Any

from log_utils import _log_exception_once


_DEFAULT_DISPATCHER: Optional["UiDispatcher"] = None

# "Strict" mode: when True, helpers never call Tk from worker threads.
# If a dispatcher is missing and the caller is not on the UI thread, we log once
# (and optionally raise) rather than doing an unsafe fallback.
_STRICT_MODE: bool = True
_RAISE_ON_VIOLATION: bool = False

# Tk apps run on the main thread in this project; keep as a conservative fallback
# for is_ui_thread() when the dispatcher is not set.
_MAIN_TID: Optional[int] = getattr(threading.main_thread(), 'ident', None)


def configure_ui_dispatch(*, strict: bool = True, raise_on_violation: bool = False) -> None:
    """Configure safety behavior.

    Args:
        strict: if True, never perform unsafe Tk calls from worker threads.
        raise_on_violation: if True, raise RuntimeError on violations (debug aid).
    """
    global _STRICT_MODE, _RAISE_ON_VIOLATION
    _STRICT_MODE = bool(strict)
    _RAISE_ON_VIOLATION = bool(raise_on_violation)


def set_default_dispatcher(d: Optional["UiDispatcher"]) -> None:
    global _DEFAULT_DISPATCHER
    _DEFAULT_DISPATCHER = d


def get_default_dispatcher() -> Optional["UiDispatcher"]:
    return _DEFAULT_DISPATCHER


def _is_main_thread() -> bool:
    try:
        return (_MAIN_TID is not None) and (threading.get_ident() == _MAIN_TID)
    except Exception:
        return False


def is_ui_thread() -> bool:
    d = get_default_dispatcher()
    if d is not None:
        return bool(d.is_ui_thread())
    # Fallback: treat main thread as UI thread.
    return _is_main_thread()


def _violation(key: str, message: str, *, context: Optional[dict] = None) -> None:
    """Handle a thread-safety violation in a throttled way."""
    exc = RuntimeError(message)
    try:
        _log_exception_once(key, exc, context=context)
    except Exception:
        # best effort
        pass
    if _RAISE_ON_VIOLATION:
        raise exc


class UiDispatcher:
    """A single dispatcher attached to the root Tk instance.

    Worker threads call .post(fn) (thread-safe).
    The UI thread pumps the queue periodically via root.after.
    """

    def __init__(self, root, interval_ms: int = 30):
        self.root = root
        self.interval_ms = int(max(5, interval_ms))
        self._q: "queue.Queue[Callable[[], Any]]" = queue.Queue()
        self._ui_tid: Optional[int] = None
        self._job_id = None
        self._stopped = False

        # Must start from UI thread; AppGUI constructs this on UI thread.
        self.start()

    def start(self) -> None:
        if self._stopped:
            return
        try:
            self._ui_tid = threading.get_ident()
        except Exception:
            self._ui_tid = None
        try:
            self._schedule_next()
        except Exception as e:
            _log_exception_once("ui_dispatch_start", e)

    def stop(self) -> None:
        self._stopped = True
        try:
            if self._job_id is not None:
                try:
                    # UI thread only
                    self.root.after_cancel(self._job_id)
                except Exception:
                    pass
                self._job_id = None
        except Exception as e:
            _log_exception_once("ui_dispatch_stop", e)

    def is_ui_thread(self) -> bool:
        try:
            return (self._ui_tid is not None) and (threading.get_ident() == self._ui_tid)
        except Exception:
            return False

    def assert_ui_thread(self, where: str = "ui") -> bool:
        """Return True if on UI thread; otherwise log once (and optionally raise)."""
        ok = self.is_ui_thread()
        if ok:
            return True
        _violation(
            "ui_thread_violation",
            f"UI-thread-only code called from worker thread at {where}",
            context={"where": where, "thread": threading.get_ident(), "ui_thread": self._ui_tid},
        )
        return False

    def post(self, fn: Callable[[], Any]) -> None:
        """Thread-safe enqueue."""
        if self._stopped:
            return
        try:
            self._q.put(fn)
        except Exception:
            # best effort
            pass

    def _schedule_next(self) -> None:
        if self._stopped:
            return
        # root.after must be called on UI thread.
        self._job_id = self.root.after(self.interval_ms, self._pump)

    def _pump(self) -> None:
        if self._stopped:
            return
        try:
            max_tasks = 200
            n = 0
            while n < max_tasks:
                try:
                    fn = self._q.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception as e:
                    _log_exception_once("ui_dispatch_task", e)
                n += 1
        except Exception as e:
            _log_exception_once("ui_dispatch_pump", e)
        finally:
            try:
                self._schedule_next()
            except Exception as e:
                _log_exception_once("ui_dispatch_resched", e)


def safe_call(fn: Callable[[], Any], *, where: str = "safe_call") -> None:
    """Run callable on UI thread.

    - UI thread: runs immediately.
    - Worker thread: posts to the dispatcher.

    If the dispatcher is missing:
    - UI thread (main thread): runs immediately.
    - Worker thread: logs once and drops (strict).
    """
    d = get_default_dispatcher()

    if d is None:
        if is_ui_thread():
            try:
                fn()
            except Exception as e:
                _log_exception_once("safe_call", e, context={"where": where})
            return
        if _STRICT_MODE:
            _violation(
                "safe_call_no_dispatcher",
                "safe_call requested from worker thread but dispatcher is not set",
                context={"where": where, "thread": threading.get_ident(), "main_thread": _MAIN_TID},
            )
            return
        # non-strict legacy fallback (not recommended)
        try:
            fn()
        except Exception as e:
            _log_exception_once("safe_call_legacy", e, context={"where": where})
        return

    if d.is_ui_thread():
        try:
            fn()
        except Exception as e:
            _log_exception_once("safe_call", e, context={"where": where})
        return

    d.post(fn)


def safe_after(widget, ms: int, func: Callable[[], Any], track_list: Optional[list] = None, *, where: str = "safe_after") -> Optional[str]:
    """Thread-safe wrapper for widget.after.

    - UI thread: calls widget.after(ms, func) and returns job id.
    - Worker thread: posts a scheduling request; returns None.

    If the dispatcher is missing:
    - UI thread (main thread): calls widget.after.
    - Worker thread: logs once and drops (strict).
    """
    d = get_default_dispatcher()
    try:
        ms_i = int(ms)
    except Exception:
        ms_i = 0

    if d is None:
        if is_ui_thread():
            try:
                jid = widget.after(ms_i, func)
                if track_list is not None:
                    try:
                        track_list.append(jid)
                    except Exception:
                        pass
                return jid
            except Exception as e:
                _log_exception_once("safe_after", e, context={"where": where})
                return None

        if _STRICT_MODE:
            _violation(
                "safe_after_no_dispatcher",
                "safe_after requested from worker thread but dispatcher is not set",
                context={"where": where, "ms": ms_i, "thread": threading.get_ident(), "main_thread": _MAIN_TID},
            )
            return None

        # non-strict legacy fallback (unsafe; kept only for completeness)
        try:
            jid = widget.after(ms_i, func)
            if track_list is not None:
                try:
                    track_list.append(jid)
                except Exception:
                    pass
            return jid
        except Exception as e:
            _log_exception_once("safe_after_legacy", e, context={"where": where})
            return None

    if not d.is_ui_thread():
        d.post(lambda: safe_after(widget, ms_i, func, track_list=track_list, where=where))
        return None

    try:
        jid = widget.after(ms_i, func)
        if track_list is not None:
            try:
                track_list.append(jid)
            except Exception:
                pass
        return jid
    except Exception as e:
        _log_exception_once("safe_after", e, context={"where": where})
        return None


def safe_cancel(widget, job_id, *, where: str = "safe_cancel") -> None:
    """Thread-safe wrapper for widget.after_cancel(job_id)."""
    if job_id is None:
        return

    d = get_default_dispatcher()

    if d is None:
        if is_ui_thread():
            try:
                widget.after_cancel(job_id)
            except Exception:
                pass
            return

        if _STRICT_MODE:
            _violation(
                "safe_cancel_no_dispatcher",
                "safe_cancel requested from worker thread but dispatcher is not set",
                context={"where": where, "thread": threading.get_ident(), "main_thread": _MAIN_TID},
            )
            return

        try:
            widget.after_cancel(job_id)
        except Exception:
            pass
        return

    if not d.is_ui_thread():
        d.post(lambda: safe_cancel(widget, job_id, where=where))
        return

    try:
        widget.after_cancel(job_id)
    except Exception:
        pass
