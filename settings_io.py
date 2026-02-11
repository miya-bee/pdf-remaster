# -*- coding: utf-8 -*-
from __future__ import annotations

"""Settings I/O and path helpers for PDF Remaster_v1_0_0 (split edition).

Design goals:
- Side-effect free (no heavy imports)
- Spawn-safe (Windows multiprocessing friendly)
- Reusable by GUI and non-GUI entrypoints (CLI/test)

This module centralizes:
- Where the config file lives (portable vs local)
- JSON read/write (UTF-8)
"""

import os
import sys
import json
import shutil
import time
import tempfile
from typing import Optional, Tuple, Dict, Any

from constants import APP_FULLNAME, CONFIG_FILENAME
from log_utils import _debug_log_exception_once


def get_app_base_dir() -> str:
    """Return the directory of the app bundle.

    - If frozen (PyInstaller etc.): directory of sys.executable
    - Else: directory of this file (same folder as run_app.py)
    """
    try:
        if getattr(sys, "frozen", False):
            return os.path.dirname(os.path.abspath(sys.executable))
    except Exception as e:
        # best-effort fallback
        _debug_log_exception_once("get_app_base_dir", e)
        pass
    return os.path.dirname(os.path.abspath(__file__))


def portable_config_path(app_base_dir: Optional[str] = None) -> str:
    base = app_base_dir or get_app_base_dir()
    return os.path.join(base, CONFIG_FILENAME)


def local_config_dir() -> str:
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, APP_FULLNAME)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        # If we fail to create dir, keep returning the candidate path.
        # The caller can handle write failure and fallback.
        _debug_log_exception_once("local_config_dir_makedirs", e)
        pass
    return d


def local_config_path() -> str:
    return os.path.join(local_config_dir(), CONFIG_FILENAME)


def config_path(portable_mode: bool, app_base_dir: Optional[str] = None) -> str:
    return portable_config_path(app_base_dir) if portable_mode else local_config_path()


def read_config(path: str) -> Tuple[Optional[dict], Optional[Exception]]:
    """Read config JSON. Returns (cfg_dict, error)."""
    if not path or not os.path.isfile(path):
        return None, None

    def _read_json(p: str) -> Tuple[Optional[dict], Optional[Exception]]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                return cfg, None
            return None, ValueError("config is not a dict")
        except Exception as e:
            return None, e

    cfg, err = _read_json(path)
    if cfg is not None:
        return cfg, None

    # Fallback: attempt to recover from a backup
    bak = path + ".bak"
    try:
        if os.path.isfile(bak):
            cfg2, err2 = _read_json(bak)
            if cfg2 is not None:
                # Mark for caller (GUI) to notify user and re-save.
                try:
                    cfg2["_recovered_from_backup"] = True
                    cfg2["_primary_read_error"] = str(err) if err else ""
                except Exception as e:
                    _debug_log_exception_once("read_config_mark_recovered", e)
                    pass
                return cfg2, None
            # if backup exists but also failed, return the original error
            if err2 is not None:
                return None, err
    except Exception as e:
        _debug_log_exception_once("read_config_backup", e)
        pass
    return None, err


def write_config(path: str, cfg: Dict[str, Any]) -> Tuple[bool, Optional[Exception]]:
    """Write config JSON. Returns (ok, error)."""
    if not path:
        return False, ValueError("empty path")
    try:
        # Ensure directory exists for local path
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        # ignore mkdir failure; the actual open() will report the real error
        _debug_log_exception_once("write_config_makedirs", e)
        pass


    ok, err = atomic_write_json(path, cfg, make_backup=True)
    return ok, err


def atomic_write_json(
    path: str,
    obj: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    make_backup: bool = False,
) -> Tuple[bool, Optional[Exception]]:
    """Atomically write JSON to `path`.

    - Writes to a temp file in the same directory and then os.replace() to prevent partial/corrupt files.
    - Optionally creates/updates a `.bak` backup (best-effort).
    """
    if not path:
        return False, ValueError("empty path")

    d = os.path.dirname(os.path.abspath(path))
    try:
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        # allow open() to report the real error
        _debug_log_exception_once("atomic_write_json_makedirs", e)
        pass

    # Backup existing (best effort)
    if make_backup:
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                shutil.copy2(path, path + ".bak")
        except Exception as e:
            _debug_log_exception_once("atomic_write_json_backup", e)
            pass

    # Temp file in the same directory (atomic replace requires same filesystem)
    base = os.path.basename(path)
    pid = 0
    try:
        pid = int(os.getpid())
    except Exception:
        pid = 0
    tmp = os.path.join(d or os.getcwd(), f"{base}.{pid}.{int(time.time()*1000)}.tmp")

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                _debug_log_exception_once("atomic_write_json_fsync", e)
                pass
        os.replace(tmp, path)
        return True, None
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception as e:
            _debug_log_exception_once("atomic_write_json_cleanup", e)
            pass
        return False, e


def is_dir_writable(path: str) -> bool:
    """Best-effort: check if a directory is writable.

        We intentionally avoid tempfile.NamedTemporaryFile on Windows because
        it can keep handles open in a way that confuses delete-on-close.
        """
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # if mkdir fails, it is effectively not writable
        return False

    fd = None
    test = None
    try:
        # Use a unique temp filename inside the target directory.
        # We avoid NamedTemporaryFile on Windows because it can keep handles open.
        fd, test = tempfile.mkstemp(prefix=".write_test_", suffix=".tmp", dir=path)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("ok")
        fd = None  # closed by fdopen
        try:
            os.remove(test)
        except Exception as e:
            # not fatal
            _debug_log_exception_once("is_dir_writable_cleanup", e)
            pass
        return True
    except Exception:
        return False
    finally:
        # Best-effort cleanup on failure paths
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if test and os.path.exists(test):
            try:
                os.remove(test)
            except Exception:
                pass



def detect_portable_mode(app_base_dir: Optional[str] = None) -> bool:
    """Decide portable mode for ZIP distribution.

    Priority:
      1) If portable config exists -> portable
      2) Else if local config exists -> local
      3) Else: choose portable if app folder is writable
    """
    base = app_base_dir or get_app_base_dir()
    try:
        if os.path.isfile(portable_config_path(base)):
            return True
    except Exception as e:
        _debug_log_exception_once("detect_portable_mode_check", e)
        pass
    try:
        if os.path.isfile(local_config_path()):
            return False
    except Exception as e:
        _debug_log_exception_once("detect_portable_mode_check", e)
        pass
    return bool(is_dir_writable(base))
