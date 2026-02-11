# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List, Tuple
import io

import fitz  # PyMuPDF

from log_utils import _log_exception_once

from embed import OcrToken, EmbedOptions, EmbedPageStats


def compose_page(
    self,
    out_doc: fitz.Document,
    bg_jpeg_bytes: bytes,
    w_px: int,
    h_px: int,
    tokens: List[OcrToken],
    src_page_rect: fitz.Rect,
    page_index: Optional[int] = None,
    *,
    ocr_w_px: Optional[int] = None,
    ocr_h_px: Optional[int] = None,
) -> Optional[EmbedPageStats]:
    """ページ合成（改修版）: トークン/文字単位で不可視テキストを埋め込み、品質統計とフォールバックを行う。"""

    # stats object (even if tokens empty, return minimal stats for visibility)
    stats = EmbedPageStats(page_index=int(page_index) if page_index is not None else -1)

    # 1) ページ作成と背景描画
    if self.output_page_mode == "original":
        page_w_pt = float(src_page_rect.width)
        page_h_pt = float(src_page_rect.height)
        page = out_doc.new_page(width=page_w_pt, height=page_h_pt)
        page_rect = fitz.Rect(0, 0, page_w_pt, page_h_pt)

        sx = page_w_pt / float(w_px) if w_px else 1.0
        sy = page_h_pt / float(h_px) if h_px else 1.0
    else:
        # pixel mode
        page = out_doc.new_page(width=float(w_px), height=float(h_px))
        page_rect = fitz.Rect(0, 0, float(w_px), float(h_px))
        sx, sy = 1.0, 1.0

    # 背景画像を挿入（PyMuPDFのバージョン差で戻り値が None / dict / int の可能性がある）
    xref_bg = None  # int or None
    _xref_bg_raw = None
    try:
        _xref_bg_raw = page.insert_image(page_rect, stream=bg_jpeg_bytes, overlay=False)
    except Exception:
        # 互換フォールバック（戻り値なし）
        page.insert_image(page_rect, stream=bg_jpeg_bytes, overlay=False)
        _xref_bg_raw = None

    # xref を安全に int 化（失敗したら None のまま）
    try:
        if isinstance(_xref_bg_raw, dict) and "xref" in _xref_bg_raw:
            xref_bg = int(_xref_bg_raw.get("xref"))
        elif _xref_bg_raw is None:
            xref_bg = None
        else:
            xref_bg = int(_xref_bg_raw)
    except Exception:
        xref_bg = None

    # 背景画像がPDF上で実際に配置されたbboxから、px→pt変換のスケールとオフセットを確定
    # insert_image で指定した page_rect と、実bboxが微小にズレる環境があるため、それを吸収する
    img_bbox = page_rect
    try:
        infos = page.get_image_info(xrefs=True)
        if infos:
            # まずxref一致を優先（取得できた場合）
            if xref_bg is not None:
                for it in infos:
                    try:
                        if int(it.get("xref", -1)) == xref_bg and it.get("bbox"):
                            r = fitz.Rect(it["bbox"])
                            if r.is_valid and r.get_area() > 0:
                                img_bbox = r
                                break
                    except Exception:
                        continue

            # xref一致で拾えない場合は最大面積画像を採用（背景が最大である前提）
            if img_bbox == page_rect:
                def _area(it):
                    bb = it.get("bbox", None)
                    if not bb:
                        return 0.0
                    r = fitz.Rect(bb)
                    return float(r.get_area()) if r.is_valid else 0.0
                best = max(infos, key=_area)
                bb = best.get("bbox", None)
                if bb:
                    r = fitz.Rect(bb)
                    if r.is_valid and r.get_area() > 0:
                        img_bbox = r
    except Exception as e:
        _log_exception_once('L2443', e)

    # 実bbox基準のオフセットとスケール（px→pt）
    ox = float(img_bbox.x0)
    oy = float(img_bbox.y0)
    sx_base = float(img_bbox.width) / float(w_px) if w_px else float(sx)
    sy_base = float(img_bbox.height) / float(h_px) if h_px else float(sy)

    if not tokens:
        # nothing to embed
        if self.embed_quality_debug:
            self.log(f"[Q] page {stats.page_index+1 if stats.page_index>=0 else '?'}: tokens=0 (skip)")
        return stats

    stats.total_tokens = len(tokens)

    # 2) 読み順の決定とソート
    dir_mode = str(getattr(self, "reading_direction_mode", "") or "").lower().strip()
    direction = "vertical" if dir_mode == "vertical" else "horizontal"
    stats.direction = direction
    sorted_base = self._sort_tokens_reading_order(tokens, direction)

    # 2.5) 本文領域推定（段組/小注・図表などの割り込みを減らす）
    body_rect = self._estimate_body_region(tokens, w_px, h_px, direction)
    stats.body_rect_used = bool(body_rect is not None)
    sorted_tokens = self._reorder_tokens_with_body_priority(sorted_base, body_rect, w_px, h_px)

    # 2.6) token座標系スケール決定（OCR入力画像サイズが分かる場合は“推定”しない）
    tok_sx = 1.0
    tok_sy = 1.0
    _used_ratio = False
    try:
        if ocr_w_px is not None and ocr_h_px is not None:
            ow = int(ocr_w_px)
            oh = int(ocr_h_px)
            if ow > 0 and oh > 0 and int(w_px) > 0 and int(h_px) > 0:
                tok_sx = float(w_px) / float(ow)
                tok_sy = float(h_px) / float(oh)
                # clamp (defensive)
                tok_sx = max(0.02, min(50.0, float(tok_sx)))
                tok_sy = max(0.02, min(50.0, float(tok_sy)))
                _used_ratio = True
    except Exception:
        _used_ratio = False

    if not _used_ratio:
        tok_sx, tok_sy = self._estimate_token_coord_scale(sorted_tokens, w_px, h_px)

    stats.token_scale_x = float(tok_sx)
    stats.token_scale_y = float(tok_sy)
    sx_text = float(sx_base) * float(tok_sx)
    sy_text = float(sy_base) * float(tok_sy)

    # 3) xscale（original モードで非等方の場合に効く）
    xscale = 1.0
    try:
        if float(sy_base) != 0.0:
            xscale = float(sx_base) / float(sy_base)
            xscale = max(0.85, min(1.25, float(xscale)))
    except Exception:
        xscale = 1.0

    # --- embedding pass helper ---
    def _embed_pass(pass_opt: EmbedOptions, stage: int) -> EmbedPageStats:
        # reset counters but keep page_index and direction/body flags
        st = EmbedPageStats(page_index=stats.page_index)
        st.direction = stats.direction
        st.total_tokens = stats.total_tokens
        st.body_rect_used = stats.body_rect_used
        st.fallback_stage = stage

        # pass option: force per-page direction (so placement matches reading order)
        try:
            pass_opt.page_direction = str(stats.direction).lower().strip()
        except Exception as e:
            _log_exception_once('L2499', e)

        # iterate in the decided order (Shapeで一括commitして高速化)
        sh: Optional[fitz.Shape] = None
        try:
            try:
                sh = page.new_shape()
            except Exception:
                sh = None

            # 追加: ページメトリクス推定（pt座標で）
            embed_metrics = self._compute_embed_page_metrics(sorted_tokens, sx_text, sy_text)

            # 追加: 縦書き列の代表X（列スナップ用）。run-mode時のみ使用。
            col_ctx = None
            try:
                if str(getattr(pass_opt, "page_direction", "")).lower().strip() == "vertical" and getattr(pass_opt, "vertical_run_mode", True):
                    tol_rd = self._compute_reading_tolerance(sorted_tokens)
                    tol_x = self._compute_vertical_column_tolerance(sorted_tokens, tol_rd)
                    col_ctx = self._compute_vertical_column_anchors(sorted_tokens, tol_x, body_rect=body_rect)
            except Exception:
                col_ctx = None

            for tok in sorted_tokens:
                if self.stop_flag.is_set():
                    break
                st.total_tokens = stats.total_tokens  # keep
                # token empty skip counted inside _insert_token_precise
                self._insert_token_precise(page, tok, sx_text, sy_text, offset_x=ox, offset_y=oy, shape=sh, xscale=xscale, stats=st, opt=pass_opt, metrics=embed_metrics, col_ctx=col_ctx)
        finally:
            if sh is not None:
                try:
                    sh.commit(overlay=True)
                except Exception:
                    # commit failure means nothing was written; count as one error sample for visibility
                    if st is not None:
                        st.insert_errors += 1

        return st

    # primary pass
    primary_opt = EmbedOptions(vertical_chunking=False, vertical_chunk_max=6, force_xscale_1=False, force_rotate0=False, tag="primary")
    pass1 = _embed_pass(primary_opt, stage=0)

            # decide fallback (staged)
    def _check_need_fallback(st: EmbedPageStats) -> Tuple[bool, str]:
        """Return (need_fallback, reason) based on current thresholds."""
        if not getattr(self, "embed_fallback_enabled", False):
            return False, ""
        try:
            min_attempts = int(getattr(self, "embed_fallback_min_attempts", 30))
            err_th = float(getattr(self, "embed_fallback_error_rate", 0.02))
            ok_th = float(getattr(self, "embed_fallback_ok_rate", 0.90))
        except Exception:
            min_attempts, err_th, ok_th = 30, 0.02, 0.90

        # primary trigger (enough samples)
        if st.attempted_inserts >= min_attempts:
            if st.error_rate() >= err_th:
                return True, f"error_rate={st.error_rate():.3f}"
            if st.ok_rate() <= ok_th:
                return True, f"ok_rate={st.ok_rate():.3f}"

        # corner case: attempts=0 but tokens exist (e.g., invalid insert path)
        if st.attempted_inserts == 0 and st.total_tokens > 0 and st.skipped_empty < st.total_tokens:
            return True, "no_attempts"

        return False, ""

    def _recreate_page():
        """Delete current page and recreate it to avoid double-text layering."""
        nonlocal page, page_rect
        try:
            out_doc.delete_page(page.number)
        except Exception as e:
            _log_exception_once('L2574', e)

        if self.output_page_mode == "original":
            page_w_pt = float(src_page_rect.width)
            page_h_pt = float(src_page_rect.height)
            page = out_doc.new_page(width=page_w_pt, height=page_h_pt)
            page_rect = fitz.Rect(0, 0, page_w_pt, page_h_pt)
        else:
            page = out_doc.new_page(width=float(w_px), height=float(h_px))
            page_rect = fitz.Rect(0, 0, float(w_px), float(h_px))

        page.insert_image(page_rect, stream=bg_jpeg_bytes, overlay=False)

    need_fb1, reason1 = _check_need_fallback(pass1)
    final_stats = pass1

    if need_fb1 and not self.stop_flag.is_set():
        # Stage 1: moderate fallback (smaller vertical chunks + optional xscale/rotate relaxation)
        _recreate_page()

        base_chunk = int(getattr(primary_opt, "vertical_chunk_max", 6))
        div = int(getattr(self, "embed_fallback_stage1_chunk_divisor", 2) or 2)
        min_chunk = int(getattr(self, "embed_fallback_stage1_min_chunk", 2) or 2)
        chunk1 = max(min_chunk, max(1, base_chunk // max(1, div)))

        stage1_opt = EmbedOptions(
            vertical_chunking=True,
            vertical_run_mode=False,
            vertical_chunk_max=int(chunk1),
            force_xscale_1=bool(getattr(self, "embed_fallback_stage1_force_xscale_1", True)),
            force_rotate0=bool(getattr(self, "embed_fallback_stage1_force_rotate0", False)),
            tag="fallback1",
        )
        pass2 = _embed_pass(stage1_opt, stage=1)
        pass2.fallback_reason = reason1
        final_stats = pass2

        # Stage 2: final safe mode (no chunking + disable xscale and rotation) if stage 1 is still bad
        if bool(getattr(self, "embed_fallback_staged", True)):
            need_fb2, reason2 = _check_need_fallback(pass2)
            if need_fb2 and not self.stop_flag.is_set():
                _recreate_page()

                stage2_opt = EmbedOptions(
                    vertical_chunking=False,
                    vertical_chunk_max=1,
                    force_xscale_1=bool(getattr(self, "embed_fallback_stage2_force_xscale_1", True)),
                    force_rotate0=bool(getattr(self, "embed_fallback_stage2_force_rotate0", True)),
                    tag="fallback2",
                )
                pass3 = _embed_pass(stage2_opt, stage=2)
                pass3.fallback_reason = f"{reason1} -> {reason2}"
                final_stats = pass3

    # log
    if self.embed_quality_debug:
        pno = final_stats.page_index + 1 if final_stats.page_index >= 0 else "?"
        fb = f" fb=Y(s{final_stats.fallback_stage}:{final_stats.fallback_reason})" if final_stats.fallback_stage > 0 else " fb=N"
        self.log(
            f"[Q] page {pno}: dir={final_stats.direction} tokens={final_stats.total_tokens} "
            f"attempts={final_stats.attempted_inserts} ok={final_stats.ok_inserts} err={final_stats.insert_errors} "
            f"skip_empty={final_stats.skipped_empty} err_rate={final_stats.error_rate():.3f}{fb}"
        )

    return final_stats


