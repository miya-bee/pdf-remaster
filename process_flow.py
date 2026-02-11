# -*- coding: utf-8 -*-
from __future__ import annotations

"""
process_flow.py

process_pdf() 内で使う「安全な小物」を切り出したモジュール。
- 既存ファイルを上書きしないための出力パス生成
- pending（メタ＋OCR）から順序どおりにページを書き出すフラッシュ処理

このモジュールは engine.py を import しない（循環依存を避ける）。
"""

import os
from typing import Any, Dict, Callable, Optional


def make_unique_path(path: str) -> str:
    """If path exists, append (1),(2)... before extension."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = f"{base}({n}){ext}"
        if not os.path.exists(cand):
            return cand
        n += 1


def flush_ready_pages(
    engine: Any,
    out_doc: Any,
    *,
    use_mp_ocr: bool,
    pending: Dict[int, Dict[str, Any]],
    start_index: int,
    log_exception_once: Optional[Callable[[str, BaseException], None]] = None,
) -> int:
    """
    pending に溜めたページ情報を、start_index から順番に「埋め込み可能なものだけ」出力する。

    pending[i] は少なくとも "meta" が必要。
    use_mp_ocr=True の場合は "words_data"（またはフォールバック用 "ocr_png"）が必要。
    """
    def _log_once(key: str, exc: BaseException, *, context: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort throttled exception log (compatible with 2-arg callbacks)."""
        if not log_exception_once:
            return
        try:
            # Newer signature: (key, exc, context=...)
            log_exception_once(key, exc, context=context)  # type: ignore[misc]
            return
        except TypeError:
            pass
        except Exception:
            return
        try:
            # Legacy signature: (key, exc)
            log_exception_once(key, exc)  # type: ignore[misc]
        except Exception:
            return

    next_to_write = start_index
    while True:
        item = pending.get(next_to_write)
        if not item:
            break
        if "meta" not in item:
            break
        if use_mp_ocr and ("words_data" not in item):
            # OCR結果待ち
            break

        meta = item["meta"]
        embed_scale = 1.0
        ocr_w_px = 0
        ocr_h_px = 0
        try:
            # Newer meta: (bg_jpeg, w_px, h_px, src_rect, embed_scale, ocr_w_px, ocr_h_px)
            bg_jpeg_bytes, w_px, h_px, src_rect, embed_scale, ocr_w_px, ocr_h_px = meta
        except Exception:
            try:
                # Legacy meta: (bg_jpeg, w_px, h_px, src_rect, embed_scale)
                bg_jpeg_bytes, w_px, h_px, src_rect, embed_scale = meta
            except Exception:
                # Oldest meta: (bg_jpeg, w_px, h_px, src_rect)
                bg_jpeg_bytes, w_px, h_px, src_rect = meta
                embed_scale = 1.0

        # tokens
        ocr_words_data = 0
        ocr_err = ""
        ocr_fallback_used = False
        ocr_ink_ratio = 0.0
        ocr_empty_reason = ""
        ocr_empty_suspected_nonblank = False

        if use_mp_ocr:
            words_data = item.get("words_data", [])
            err = item.get("ocr_err", "")
            ocr_err = str(err or "")
            try:
                ocr_words_data = int(len(words_data)) if isinstance(words_data, list) else 0
            except Exception:
                ocr_words_data = 0
            if ocr_err:
                try:
                    engine.log(f"[WARN] OCR worker error on page {next_to_write+1}:\n{ocr_err}")
                except Exception:
                    pass

            tokens = engine._tokens_from_wordsdata_or_fallback(
                words_data,
                fallback_png=item.get("ocr_png"),
                page_no_1based=next_to_write + 1,
            )
        else:
            tokens = item.get("tokens", [])
            # In single-process mode, we store a rough size as 'ocr_words_data' for diagnostics.
            try:
                ocr_words_data = int(item.get("ocr_words_data", 0))
            except Exception:
                ocr_words_data = 0

        # Per-page OCR diagnostics (ink ratio + fallback)
        try:
            ocr_ink_ratio = float(item.get("ocr_ink_ratio", 0.0) or 0.0)
        except Exception:
            ocr_ink_ratio = 0.0
        try:
            pages = getattr(engine, "_ocr_fallback_pages", None)
            ocr_fallback_used = bool(isinstance(pages, set) and (int(next_to_write + 1) in pages))
        except Exception:
            ocr_fallback_used = False

        # If OCR produced no tokens, decide a short reason tag (for later reporting)
        if not tokens:
            # Threshold in percent of 'dark-ish' pixels; tuned to avoid false positives.
            # (Blank pages are usually near 0.0, while even lightly-inked pages exceed 0.5.)
            try:
                ocr_empty_suspected_nonblank = bool(float(ocr_ink_ratio) >= 0.5)
            except Exception:
                ocr_empty_suspected_nonblank = False

            if ocr_err:
                ocr_empty_reason = "worker_error"
            elif use_mp_ocr:
                if ocr_words_data <= 0:
                    ocr_empty_reason = "worker_empty"
                else:
                    # Worker returned something, but token conversion dropped all.
                    valid_like = 0
                    try:
                        for w in (words_data if isinstance(words_data, list) else []):
                            if not isinstance(w, dict):
                                continue
                            txt = w.get("text", "")
                            pts = w.get("points", [])
                            if txt and isinstance(pts, list) and len(pts) == 4:
                                valid_like += 1
                                if valid_like >= 3:
                                    break
                    except Exception:
                        valid_like = 0
                    ocr_empty_reason = "wordsdata_invalid" if valid_like == 0 else "wordsdata_dropped"
            else:
                ocr_empty_reason = "singleproc_empty"

        # compose
        try:
            stats_page = engine._compose_page(
                out_doc,
                bg_jpeg_bytes,
                w_px,
                h_px,
                tokens,
                src_rect,
                page_index=next_to_write,
                ocr_w_px=int(ocr_w_px) if int(ocr_w_px or 0) > 0 else None,
                ocr_h_px=int(ocr_h_px) if int(ocr_h_px or 0) > 0 else None,
            )
            if stats_page is not None:
                try:
                    # Attach OCR diagnostics into EmbedPageStats (regression prevention)
                    try:
                        stats_page.ocr_words_data = int(ocr_words_data)
                        stats_page.ocr_err = str(ocr_err or "")
                        stats_page.ocr_fallback_used = bool(ocr_fallback_used)
                        stats_page.ocr_ink_ratio = float(ocr_ink_ratio)
                        stats_page.ocr_empty_suspected_nonblank = bool(ocr_empty_suspected_nonblank)
                        stats_page.ocr_empty_reason = str(ocr_empty_reason or "")
                    except Exception:
                        pass

                    engine._embed_stats.append(stats_page)
                except Exception as e:
                    if log_exception_once:
                        log_exception_once("flush_embed_stats", e)
        except Exception as e:
            # ここで例外を飲むとページ順が崩れるので、そのまま投げる
            raise

        # free
        try:
            del pending[next_to_write]
        except Exception:
            pass
        next_to_write += 1

        # memory hygiene
        try:
            engine._maybe_store_shrink()
        except Exception as e:
            _log_once('maybe_store_shrink', e, context={'page_no': int(next_to_write)+1})

    return next_to_write
