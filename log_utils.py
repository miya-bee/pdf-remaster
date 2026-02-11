# -*- coding: utf-8 -*-
from __future__ import annotations

import traceback
from typing import Any, Dict, Optional, Set

import os
import sys
import time
import threading
import platform
import logging
from logging.handlers import RotatingFileHandler

# Exception logging (throttled)
_LOG_ONCE_KEYS: Set[str] = set()


# File logging (for public release / support)
_LOGGER_INITIALIZED = False
_LOG_FILE_PATH: Optional[str] = None
_TEE_FILE = None
_TEE_LOCK = threading.Lock()
_ORIG_STDOUT = None
_ORIG_STDERR = None


class _TeeWriter:
    def __init__(self, original, tee_file):
        self._orig = original
        self._tee = tee_file

    def write(self, s):
        if not s:
            return 0
        n = 0
        try:
            n = self._orig.write(s)
        except Exception:
            pass
        try:
            with _TEE_LOCK:
                self._tee.write(s)
                if "\n" in s or len(s) > 512:
                    self._tee.flush()
        except Exception:
            pass
        return n

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            with _TEE_LOCK:
                self._tee.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(getattr(self._orig, "isatty", lambda: False)())
        except Exception:
            return False


def get_log_path() -> Optional[str]:
    return _LOG_FILE_PATH


def _default_log_dir(*, portable_mode: bool, base_dir: Optional[str] = None) -> str:
    # Portable: alongside the app folder; otherwise: LOCALAPPDATA/<APP_FULLNAME>/logs
    try:
        if portable_mode and base_dir:
            d = os.path.join(base_dir, "logs")
        else:
            try:
                from settings_io import local_config_dir  # lightweight helper
                d = os.path.join(local_config_dir(), "logs")
            except Exception:
                base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
                d = os.path.join(base, "PDF_Remaster_v1_0_0", "logs")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return os.path.join(os.getcwd(), "logs")


def setup_app_logging(*, portable_mode: bool = False, base_dir: Optional[str] = None, tee_stdio: bool = True) -> Optional[str]:
    """Initialize rotating file logs + optional stdout/stderr tee (idempotent)."""
    global _LOGGER_INITIALIZED, _LOG_FILE_PATH, _TEE_FILE, _ORIG_STDOUT, _ORIG_STDERR
    if _LOGGER_INITIALIZED:
        return _LOG_FILE_PATH

    log_dir = _default_log_dir(portable_mode=portable_mode, base_dir=base_dir)
    log_path = os.path.join(log_dir, "app.log")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace"):
            pass
    except Exception:
        log_path = None

    # Tee stdout/stderr so print-based logs are captured too.
    if tee_stdio and log_path:
        try:
            if _ORIG_STDOUT is None:
                _ORIG_STDOUT = sys.stdout
            if _ORIG_STDERR is None:
                _ORIG_STDERR = sys.stderr
            _TEE_FILE = open(log_path, "a", encoding="utf-8", errors="replace")
            sys.stdout = _TeeWriter(_ORIG_STDOUT, _TEE_FILE)  # type: ignore
            sys.stderr = _TeeWriter(_ORIG_STDERR, _TEE_FILE)  # type: ignore
        except Exception:
            pass

    if log_path:
        try:
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            if not getattr(root, "_pdfremaster_has_file_handler", False):
                fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
                fh.setFormatter(fmt)
                root.addHandler(fh)

                # Console handler (write to original stderr to avoid tee double-write)
                ch = logging.StreamHandler(_ORIG_STDERR or sys.__stderr__)
                ch.setFormatter(fmt)
                root.addHandler(ch)

                setattr(root, "_pdfremaster_has_file_handler", True)
        except Exception:
            pass

    _LOG_FILE_PATH = log_path
    _LOGGER_INITIALIZED = True
    return _LOG_FILE_PATH


def log_environment() -> None:
    """Write environment info once (helps support)."""
    try:
        logger = logging.getLogger("env")
        logger.info("=== Environment ===")
        try:
            from constants import APP_FULLNAME
            logger.info(f"app={APP_FULLNAME}")
        except Exception:
            pass
        logger.info(f"python={sys.version.replace(os.linesep, ' ')}")
        logger.info(f"platform={platform.platform()}")
        logger.info(f"frozen={bool(getattr(sys, 'frozen', False))}")
        if _LOG_FILE_PATH:
            logger.info(f"log_file={_LOG_FILE_PATH}")

        # Optional deps (best effort)
        try:
            import numpy as np  # type: ignore
            nv = str(getattr(np, '__version__', '?'))
            logger.info(f"numpy={nv}")
            # Numpy 2.x は依存ライブラリによって互換性問題が出やすい（public release向けに注意喚起）
            try:
                import re
                m = re.match(r"^(\d+)", nv)
                if m and int(m.group(1)) >= 2:
                    logger.warning(
                        "numpy 2.x が検出されました。環境によっては依存ライブラリの互換性問題が起きる可能性があります。"
                        " 問題が出る場合は requirements.txt の環境（numpy 1.x）を推奨します。"
                    )
            except Exception:
                pass
        except Exception as e:
            logger.info(f"numpy=unavailable ({e})")
        try:
            import cv2  # type: ignore
            logger.info(f"opencv={getattr(cv2, '__version__', '?')}")
        except Exception as e:
            logger.info(f"opencv=unavailable ({e})")
        try:
            import torch  # type: ignore
            logger.info(f"torch={getattr(torch, '__version__', '?')}")
            try:
                cuda_ok = bool(torch.cuda.is_available())
                logger.info(f"cuda_available={cuda_ok}")
                if cuda_ok:
                    try:
                        n = int(torch.cuda.device_count())
                        logger.info(f"cuda_devices={n}")
                        if n > 0:
                            logger.info(f"cuda_device0={torch.cuda.get_device_name(0)}")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.info(f"torch=unavailable ({e})")
        logger.info("=== End Environment ===")
    except Exception:
        return


def log_exception_once(key: str, exc: BaseException, *, context: Optional[Dict[str, Any]] = None, prefix: str = "") -> None:
    """Log an exception only once per key to avoid spamming console.

    Designed to replace silent exception swallowing (e.g. `except Exception: pass`)
    while keeping the app resilient.

    Args:
        key: feature/function identifier (e.g. "ZoomImageViewer.render")
        exc: caught exception
        context: optional details (page number, strength, paths...) to aid debugging
        prefix: optional string prepended to the key (rarely needed)
    """
    if not key:
        key = "exception"

    # Use key + stable context keys as the throttle key (so a single bad page doesn't spam logs).
    throttle_key = key
    try:
        if context:
            # include only keys (not values) in throttle key to keep it stable
            throttle_key = key + "|" + ",".join(sorted(str(k) for k in context.keys()))
    except Exception:
        throttle_key = key

    if throttle_key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(throttle_key)

    # Build message
    ctx_s = ""
    if context:
        try:
            items = []
            for k, v in context.items():
                try:
                    items.append(f"{k}={v}")
                except Exception:
                    items.append(f"{k}=?")
            ctx_s = " | " + ", ".join(items)
        except Exception:
            ctx_s = ""

    try:
        msg = f"[WARN] {prefix}{key}: {type(exc).__name__}: {exc}{ctx_s}"
    except Exception:
        msg = f"[WARN] {prefix}{key}: (exception){ctx_s}"
    # Also write to file logger (best effort)
    try:
        logger = logging.getLogger("pdf_remaster.exception")
        logger.error(msg)
        try:
            tb2 = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if tb2:
                logger.error(tb2)
        except Exception:
            pass
    except Exception:
        pass

    print(msg)

    # Print traceback (best effort)
    try:
        tb = traceback.format_exc()
        if tb and ("NoneType: None" not in tb):
            print(tb)
    except Exception:
        return


# Backward-compatible alias used throughout the codebase
def _log_exception_once(key: str, exc: BaseException, *, prefix: str = "", context: Optional[Dict[str, Any]] = None) -> None:
    log_exception_once(key, exc, context=context, prefix=prefix)

def debug_exceptions_enabled() -> bool:
    """Return True if swallowed-exception debug logging is enabled.

    Enable by either:
      - constants.DEBUG_LOG_EXCEPTIONS = True
      - env var PDFREMASTER_DEBUG_EXCEPTIONS=1 / true / yes / on
    """
    try:
        from constants import DEBUG_LOG_EXCEPTIONS  # type: ignore
        if bool(DEBUG_LOG_EXCEPTIONS):
            return True
    except Exception:
        pass
    try:
        v = str(os.environ.get("PDFREMASTER_DEBUG_EXCEPTIONS", "")).strip().lower()
        return v in ("1", "true", "yes", "on")
    except Exception:
        return False


def debug_log_exception_once(
    key: str,
    exc: BaseException,
    *,
    context: Optional[Dict[str, Any]] = None,
    prefix: str = "[DEBUG][swallowed] ",
) -> None:
    """Log a swallowed exception (throttled) only when debug mode is enabled."""
    if not debug_exceptions_enabled():
        return
    try:
        log_exception_once(f"dbg_{key}", exc, context=context, prefix=prefix)
    except Exception:
        pass


# Backward-compatible internal helper name (matches existing style)
def _debug_log_exception_once(key: str, exc: BaseException, *, context: Optional[Dict[str, Any]] = None) -> None:
    debug_log_exception_once(key, exc, context=context)
