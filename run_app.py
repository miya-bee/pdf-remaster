# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import traceback
import py_compile
import tempfile

from constants import CONFIG_FILENAME
import settings_io
from log_utils import setup_app_logging, log_environment, get_log_path


def _show_fatal_dialog(title: str, message: str) -> None:
    """Best-effort: show a Windows message box for fatal startup errors."""
    try:
        if sys.platform.startswith("win"):
            import ctypes  # type: ignore
            MB_OK = 0x0
            MB_ICONERROR = 0x10
            ctypes.windll.user32.MessageBoxW(None, str(message), str(title), MB_OK | MB_ICONERROR)
            return
    except Exception:
        pass
    # Fallback: print to stderr
    try:
        sys.stderr.write(f"{title}\n{message}\n")
    except Exception:
        pass


def _startup_sanity_check() -> None:
    """Detect common packaging / merge errors early (before GUI starts)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize file logging early (for public release / support).
    # ZIP配布では「アプリフォルダに書けるならポータブル」を既定にする。
    # 既存の設定がある場合はそれを優先。
    portable_mode = bool(settings_io.detect_portable_mode(base_dir))
    setup_app_logging(portable_mode=portable_mode, base_dir=base_dir, tee_stdio=True)
    log_environment()

    frozen = bool(getattr(sys, "frozen", False))

    # 1) Syntax check only app modules shipped in this ZIP.
    #    (Users may keep other .py files in the same folder; do not fail on those.)
    #    In frozen/exe builds, source .py files may not exist; skip syntax compile in that case.
    if not frozen:
        modules = [
            "constants.py",
            "embed.py",
            "engine.py",
            "gui.py",
            "image_ops.py",
            "log_utils.py",
            "ocr_pipeline.py",
            "ocr_worker.py",
            "page_pipeline.py",
            "pdf_compose.py",
            "pdf_io.py",
            "process_flow.py",
            "process_runner.py",
            "run_app.py",
            "settings_io.py",
            "text_embed.py",
            "ui_dispatch.py",
            "zoom_viewer.py",
        ]
        for fn in modules:
            fp = os.path.join(base_dir, fn)
            if not os.path.isfile(fp):
                continue
            try:
                fd, cfile = tempfile.mkstemp(prefix=os.path.basename(fp) + ".", suffix=".pyc")
                try:
                    os.close(fd)
                except Exception:
                    pass
                try:
                    py_compile.compile(fp, cfile=cfile, doraise=True)
                finally:
                    try:
                        os.remove(cfile)
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(f"Syntax check failed: {os.path.basename(fp)}\n{e}") from e

    # 2) Must-have methods for zoom viewer (prevents black window / missing callbacks).
    from zoom_viewer import ZoomImageViewer  # noqa: F401
    required = ["_render", "_apply_bold_now", "_poll_bold_results", "_apply_bin_now", "_schedule_render"]
    missing = [name for name in required if not hasattr(ZoomImageViewer, name)]
    if missing:
        raise RuntimeError(f"ZoomImageViewer missing required methods: {missing}")

    # 3) Basic GUI entry points exist.
    import gui as _gui  # noqa: F401
    if not hasattr(_gui, "main") or not callable(getattr(_gui, "main")):
        raise RuntimeError("gui.main is missing or not callable")


def main():
    try:
        _startup_sanity_check()
    except Exception as e:
        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        lp = get_log_path()
        if lp:
            msg += f"\n\nLog file: {lp}\n"
        _show_fatal_dialog("PDF Remaster startup error", msg)
        raise

    from gui import main as _main
    _main()


if __name__ == "__main__":
    main()
