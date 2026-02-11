# -*- coding: utf-8 -*-
from __future__ import annotations

"""
pdf_io.py
PDF (PyMuPDF/fitz) I/O wrappers.

Purpose:
- Centralize PyMuPDF version-difference handling (save args / open / metadata) in one place.
- Keep engine.py focused on orchestration and image/OCR logic.
"""

from typing import Any, Dict, Optional
import fitz
from log_utils import _debug_log_exception_once


def open_document(path: str) -> fitz.Document:
    """Open a PDF document."""
    return fitz.open(path)


def new_document() -> fitz.Document:
    """Create a new empty PDF document."""
    return fitz.open()


def save_document(doc: fitz.Document, path: str, **kwargs: Any) -> None:
    """
    Save PDF with best-effort compatibility across PyMuPDF versions.

    - First saves to a temporary file in the same directory, then atomically replaces the target.
      This prevents leaving a broken/partial PDF when the process is cancelled or crashes mid-save.
    - Tries with provided kwargs first; if TypeError (unsupported args), progressively drops newer kwargs and retries.
    """
    import os
    import tempfile

    out_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    base_name = os.path.basename(path)
    tmp_path = None

    # Windows: must close file handle before fitz.save writes into it, so use mkstemp.
    fd, tmp_path = tempfile.mkstemp(prefix=base_name + ".", suffix=".tmp", dir=out_dir)
    try:
        try:
            os.close(fd)
        except Exception:
            pass

        def _try_save(p: str, _kw: Dict[str, Any]) -> None:
            # Fast path
            try:
                doc.save(p, **_kw)
                return
            except TypeError:
                pass

            # Drop args that are version-sensitive (older PyMuPDF)
            drop_order = [
                "deflate_images",
                "use_objstms",
                "clean",
                "garbage",
                "deflate",
                "incremental",
            ]

            kw2 = dict(_kw)
            for k in drop_order:
                if k in kw2:
                    kw2.pop(k, None)
                    try:
                        doc.save(p, **kw2)
                        return
                    except TypeError:
                        continue
                    except Exception:
                        raise

            # Last resort: no kwargs
            doc.save(p)

        _try_save(tmp_path, dict(kwargs))

        # Atomic replace (same filesystem)
        os.replace(tmp_path, path)
        tmp_path = None  # replaced
        return
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def apply_pdf_metadata(doc: fitz.Document, in_meta: Dict[str, Any], app_fullname: str) -> None:
    """
    Apply input PDF metadata to output PDF, excluding empty/None values.
    Also appends app_fullname to 'producer' if present.
    """
    allow = {"title","author","subject","keywords","creator","producer","creationDate","modDate","trapped"}
    meta: Dict[str, str] = {}
    for k, v in dict(in_meta or {}).items():
        if k in allow and v:
            meta[k] = str(v)

    if "producer" in meta:
        if app_fullname and (app_fullname not in meta["producer"]):
            meta["producer"] = (meta["producer"] + f" | {app_fullname}").strip()
    else:
        if app_fullname:
            meta["producer"] = app_fullname

    try:
        doc.set_metadata(meta)
    except Exception as e:
        # Metadata is non-critical; ignore if unsupported
        _debug_log_exception_once("pdf_set_metadata", e)
        pass
