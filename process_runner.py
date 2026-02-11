# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import gc
import json
import queue
import multiprocessing as mp
from typing import List, Optional, Dict, Any, Tuple

# Optional dependency: torch (used for CUDA cache management and GPU-related checks)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


import fitz  # PyMuPDF
import cv2

from constants import APP_FULLNAME, DEFAULT_BG_SCALE_PERCENT, DEFAULT_MAX_OUTPUT_DPI
import pdf_io
import process_flow
import settings_io as _settings_io
from log_utils import _log_exception_once

# Unified page pipeline (shared by final/preview)
from page_pipeline import run_page_pipeline

def process_pdf(engine, input_pdf: str, output_dir: str) -> str:
    """Run the full PDF enhancement pipeline.

    This is factored out of PdfOcrEnhanceEngine.process_pdf to keep engine.py smaller
    and reduce import-side effects. The function is intentionally written to be a near-
    verbatim move of the original method for safety.
    """
    self = engine
    if not os.path.isfile(input_pdf):
        raise FileNotFoundError(input_pdf)
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(output_dir)

    try:
        self._embed_stats = []
    except Exception as e:
        _log_exception_once('L2675', e)

    # Per-page OCR fallback tracking (for invisible-text regression prevention)
    try:
        self._ocr_fallback_pages = set()  # 1-based page numbers
    except Exception:
        pass

    def _estimate_ink_ratio_percent(bgr_img) -> float:
        """Rough 'ink' ratio (dark pixels %) for OCR image.

        Used only for diagnostics when OCR returns 0 tokens, to distinguish
        truly blank pages from OCR failures.
        """
        try:
            if bgr_img is None:
                return 0.0
            # Downscale for speed
            h, w = bgr_img.shape[:2]
            if h <= 0 or w <= 0:
                return 0.0
            scale = 256.0 / float(max(h, w))
            if scale < 1.0:
                nh = max(1, int(h * scale))
                nw = max(1, int(w * scale))
                small = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                small = bgr_img
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Count 'dark-ish' pixels (tolerant threshold)
            thr = 240
            dark = int((gray < thr).sum())
            tot = int(gray.size)
            if tot <= 0:
                return 0.0
            return max(0.0, min(100.0, 100.0 * float(dark) / float(tot)))
        except Exception:
            return 0.0

    self.log(f"[INFO] 入力PDF: {input_pdf}")
    self.log(f"[INFO] 出力先: {output_dir}")
    self.log(f"[INFO] 動作モード: {self.device_str}")
    self.log(f"[INFO] 設定: DPI={self.base_dpi}, JPEG品質={self.jpeg_quality}, 背景縮小={self.bg_scale_percent}%, 出力上限DPI={self.max_output_dpi}, GrayAuto={'ON' if self.auto_grayscale else 'OFF'}(ratio<{self.gray_color_ratio_percent:.2f}%), GrayQoff={self.gray_jpeg_quality_offset}, 二値化強度={self.binarize_strength}, 文字太さ（閲覧）={self.text_boldness}, ESRGAN tile={self.esrgan_tile}")
    self.log(f"[INFO] 出力ページモード: {self.output_page_mode}")
    self.log(f"[INFO] Deskew: {'ON' if self.enable_deskew else 'OFF'} (max={self.deskew_max_deg}deg)")
    self.log(f"[INFO] OCR並列: workers={self.ocr_workers} (0=無効)")
    self.log(f"[INFO] store_shrink: {self.store_shrink}")
    if self.font_path:
        self.log(f"[INFO] フォント: {self.font_path}")

    in_doc = pdf_io.open_document(input_pdf)
    total = in_doc.page_count

    # ===== PDF構造の保持（目次/メタデータ） =====
    try:
        in_toc = in_doc.get_toc()  # [[lvl, title, page], ...]
    except Exception:
        in_toc = []
    try:
        in_meta = in_doc.metadata or {}
    except Exception:
        in_meta = {}

    def _apply_pdf_metadata(doc: fitz.Document):
        """入力PDFのメタデータを出力PDFへ反映（Noneは除外）"""
        try:
            pdf_io.apply_pdf_metadata(doc, in_meta, APP_FULLNAME)
        except Exception as e:
            _log_exception_once('L2717', e)
    base = os.path.splitext(os.path.basename(input_pdf))[0]

    # Work output (to allow periodic close/reopen safely on Windows)
    # 既存ファイルがある場合は (1),(2)... を付けて上書き回避
    out_final = os.path.join(output_dir, f"{base}_{APP_FULLNAME}.pdf")
    out_final = process_flow.make_unique_path(out_final)

    # Cancel behavior: whether to write partial output when stopped.
    # GUI can set `engine.keep_partial_on_cancel`.
    try:
        keep_partial_on_cancel = bool(getattr(self, "keep_partial_on_cancel", True))
    except Exception:
        keep_partial_on_cancel = True

    # Return path (empty string means "no output written")
    out_path: str = out_final

    # work ファイルは最終出力名に追従（クラッシュ時の残骸衝突も避ける）
    work_base = os.path.splitext(out_final)[0]
    out_work_a = work_base + ".workA.pdf"
    out_work_b = work_base + ".workB.pdf"
    for _p in (out_work_a, out_work_b):
        try:
            if os.path.exists(_p):
                os.remove(_p)
        except Exception as e:
            _log_exception_once('L2734', e)

    # Determine flush interval (close/reopen) for stability
    if total >= 120:
        flush_every = 8
    elif total >= 100:
        flush_every = 10
    elif total >= 60:
        flush_every = 15
    elif total >= 40:
        flush_every = 20
    else:
        flush_every = 0  # small docs: no need

    self.log(f"[INFO] 安定化: flush_every={flush_every} ページ（0=無効）")

    out_doc = pdf_io.new_document()
    _apply_pdf_metadata(out_doc)
    next_flush_path = out_work_a
    alt_flush_path = out_work_b

    # ===== OCR multiprocessing setup (optional) =====
    use_mp_ocr = (self.ocr_workers >= 1)
    ctx = None
    task_q = None
    result_q = None
    workers: List[mp.Process] = []

    if use_mp_ocr:
        # GPU is busy with SR; keep OCR on CPU to avoid contention
        ocr_device = "cpu"
        use_mp_ocr, ctx, task_q, result_q, workers = self._start_mp_ocr_workers(self.ocr_workers, device_str=ocr_device)

    pending: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0

    try:
        for i in range(total):
            if self.stop_flag.is_set():
                self.log("[INFO] 中断しました。")
                break

            self.progress(i + 1, total)
            self.log(f"[INFO] ページ処理中: {i+1}/{total}")

            page = in_doc.load_page(i)
            src_rect = page.rect

            # 1) render
            rgb = self._render_page_to_rgb(page)

            # Unified page pipeline (final mode)

            # (Optional) one-time debug log when boldness is enabled
            if int(getattr(self, 'text_boldness', 0)) != 0:
                try:
                    if not getattr(self, '_did_log_text_boldness_apply', False):
                        self.log(f"[DEBUG] 背景の文字太さ調整を適用します: text_boldness={int(self.text_boldness)}")
                        self._did_log_text_boldness_apply = True
                except Exception:
                    pass

            pipe = run_page_pipeline(
                self,
                rgb=rgb,
                src_rect=src_rect,
                page_index=i,
                mode="final",
                final_apply_boldness=True,
                final_apply_scale=True,
            )

            ang = float(pipe.deskew_angle)
            if abs(ang) >= 0.3:
                self.log(f"[INFO] Deskew: page {i+1} angle={ang:+.2f} deg")

            ocr_bgr = pipe.ocr_bgr
            bg_jpeg = pipe.bg_jpeg_bytes
            w_px = int(pipe.bg_w_px or 0)
            h_px = int(pipe.bg_h_px or 0)
            embed_scale = float(pipe.embed_scale or 1.0)

            # Lightweight diagnostic to detect OCR-zero pages on non-blank content.
            # Stored in pending so flush_ready_pages can attach it to embed stats.
            try:
                pending.setdefault(i, {})["ocr_ink_ratio"] = float(_estimate_ink_ratio_percent(ocr_bgr))
            except Exception:
                pending.setdefault(i, {})["ocr_ink_ratio"] = 0.0

            if ocr_bgr is None or bg_jpeg is None or w_px <= 0 or h_px <= 0:
                raise RuntimeError("ページパイプラインの生成に失敗しました（ocr_bgr/bg_jpeg が空）")

            # IMPORTANT: free large arrays early
            del page, rgb
            gc.collect()
            # record OCR input size (px) for accurate token coord scaling
            try:
                _oh, _ow = ocr_bgr.shape[:2]
                ocr_w_px = int(_ow)
                ocr_h_px = int(_oh)
            except Exception:
                ocr_w_px = 0
                ocr_h_px = 0

            # store meta: (jpeg_bytes, w_px, h_px, src_rect, embed_scale, ocr_w_px, ocr_h_px)
            pending.setdefault(i, {})["meta"] = (bg_jpeg, w_px, h_px, src_rect, float(embed_scale), int(ocr_w_px), int(ocr_h_px))

            # 5) OCR either via mp workers or single-process
            if use_mp_ocr:
                # encode OCR image to PNG bytes and send to worker
                ok, buf = cv2.imencode(".png", ocr_bgr)
                if not ok:
                    raise RuntimeError("OCR画像のPNGエンコードに失敗しました。")
                png_bytes = buf.tobytes()

                # Keep OCR image bytes for single-process fallback if worker returns empty / error
                pending.setdefault(i, {})["ocr_png"] = png_bytes

                # enqueue (non-blocking; drain results if queue is full to avoid deadlock)
                if task_q is not None:
                    def _on_taskq_full():
                        nonlocal next_to_write
                        if use_mp_ocr and result_q is not None:

                            self._mp_drain_results(result_q, pending)
                        next_to_write = process_flow.flush_ready_pages(self, out_doc, use_mp_ocr=use_mp_ocr, pending=pending, start_index=next_to_write, log_exception_once=_log_exception_once)
                    self._mp_put_task(task_q, (i, png_bytes), workers, on_queue_full=_on_taskq_full)
                del buf, png_bytes
            else:
                tokens = self._run_ocr_singleproc(ocr_bgr)
                pending.setdefault(i, {})["tokens"] = tokens
                # Record a rough OCR output size for diagnostics
                try:
                    pending.setdefault(i, {})["ocr_words_data"] = int(len(tokens))
                except Exception:
                    pending.setdefault(i, {})["ocr_words_data"] = 0

            del ocr_bgr
            self._maybe_store_shrink()

            # Drain OCR results (non-blocking) and try to flush pages
            if use_mp_ocr and result_q is not None:

                self._mp_drain_results(result_q, pending)
            next_to_write = process_flow.flush_ready_pages(self, out_doc, use_mp_ocr=use_mp_ocr, pending=pending, start_index=next_to_write, log_exception_once=_log_exception_once)

            # Periodic out_doc close/reopen for large docs (stability)
            if flush_every and (i + 1) % flush_every == 0 and out_doc.page_count > 0:
                self.log(f"[INFO] 安定化: {flush_every}ページごとに出力PDFを保存→再オープンします。")
                pdf_io.save_document(out_doc,
                    next_flush_path,
                    garbage=4, clean=1,
                    deflate=1, deflate_images=1,
                    use_objstms=1
                )
                out_doc.close()

                # Re-open output doc (carry pages)
                out_doc = pdf_io.open_document(next_flush_path)
                _apply_pdf_metadata(out_doc)

                # swap paths
                next_flush_path, alt_flush_path = alt_flush_path, next_flush_path

                # Also refresh input doc (optional defensive)
                try:
                    in_doc.close()
                    in_doc = pdf_io.open_document(input_pdf)
                except Exception as e:
                    _log_exception_once('L2996', e)

                self._maybe_store_shrink()
                gc.collect()
                if torch is not None and self.device_str == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        _log_exception_once('L3004', e)

        # ===== finalize: wait remaining OCR results if mp enabled =====
        if use_mp_ocr and (not self.stop_flag.is_set()):
            # send sentinel to workers
            if task_q is not None:
                for _ in workers:
                    def _on_taskq_full_sentinel():
                        nonlocal next_to_write
                        if use_mp_ocr and result_q is not None:

                            self._mp_drain_results(result_q, pending)
                        next_to_write = process_flow.flush_ready_pages(self, out_doc, use_mp_ocr=use_mp_ocr, pending=pending, start_index=next_to_write, log_exception_once=_log_exception_once)
                    self._mp_put_task(task_q, None, workers, on_queue_full=_on_taskq_full_sentinel, timeout_sec=120.0)

            # collect results until all pages are written or stop
            self.log("[INFO] OCR結果の回収中...")
            # 進捗が一定時間止まったら、ワーカー死活を確認して中断（デッドロック防止）
            last_progress = time.time()
            max_no_progress_sec = 120.0  # ここを長めにすると待機を許容
            while next_to_write < total and not self.stop_flag.is_set():
                # Drain some results (blocking with timeout)
                try:
                    idx, words_data, err = result_q.get(timeout=1.0)  # type: ignore
                    last_progress = time.time()
                    if idx == -1:
                        raise RuntimeError(f"OCR worker initialization failed:\n{err}")
                    pending.setdefault(idx, {})["words_data"] = words_data
                    pending[idx]["ocr_err"] = err
                except queue.Empty:
                    # 結果待ち：一定時間進捗が無い場合はワーカー死活を確認して中断（無限待ち防止）
                    if (time.time() - last_progress) > max_no_progress_sec:
                        try:
                            alive = sum(1 for p in workers if p.is_alive())
                        except Exception:
                            alive = -1
                        if alive == 0:
                            raise RuntimeError("OCRワーカーが全て停止しました。環境依存の初期化失敗（CUDA/ONNX/torch）やメモリ不足の可能性があります。")
                        raise RuntimeError(f"OCR結果の回収が {max_no_progress_sec:.0f} 秒以上進まず停止しました（デッドロック防止で中断）。")
                    pass

                old_next = next_to_write
                next_to_write = process_flow.flush_ready_pages(self, out_doc, use_mp_ocr=use_mp_ocr, pending=pending, start_index=next_to_write, log_exception_once=_log_exception_once)
                if next_to_write != old_next:
                    last_progress = time.time()

            # join workers
            for p in workers:
                try:
                    p.join(timeout=2.0)
                except Exception as e:
                    _log_exception_once('L3049', e)
            # ensure workers are not left hanging (avoid shutdown hang)
            for p in workers:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=1.0)
                except Exception as e:
                    _log_exception_once('L3049_term', e)


        # flush remaining in single-process mode
        next_to_write = process_flow.flush_ready_pages(self, out_doc, use_mp_ocr=use_mp_ocr, pending=pending, start_index=next_to_write, log_exception_once=_log_exception_once)

        # save final
        cancelled = bool(getattr(self, 'stop_flag', None) is not None and self.stop_flag.is_set())

        if cancelled and (not keep_partial_on_cancel):
            # User requested to stop and does not want partial outputs.
            out_path = ""
            try:
                self.log("[INFO] 中断: 設定により途中結果PDFは出力しません。")
            except Exception:
                pass

        elif out_doc.page_count > 0:
            # 目次(Outline)とメタデータを反映してから保存
            try:
                _apply_pdf_metadata(out_doc)
            except Exception as e:
                _log_exception_once('L3060', e)
            try:
                if in_toc:
                    out_doc.set_toc(in_toc)
            except Exception as e:
                self.log(f"[WARN] 目次(TOC)のコピーに失敗: {e}")
            pdf_io.save_document(out_doc,out_final, garbage=4, clean=1, deflate=1, deflate_images=1, use_objstms=1)

            # --- Invisible-text regression prevention: surface OCR-zero pages ---
            try:
                st_list = list(getattr(self, "_embed_stats", []) or [])
                if st_list:
                    tokens0 = [st for st in st_list if int(getattr(st, "total_tokens", 0) or 0) <= 0]
                    tokens0_nb = [st for st in tokens0 if bool(getattr(st, "ocr_empty_suspected_nonblank", False))]
                    fb_pages = sum(1 for st in st_list if bool(getattr(st, "ocr_fallback_used", False)))

                    if tokens0_nb:
                        try:
                            pages = ",".join(str(int(getattr(st, "page_index", -1)) + 1) for st in tokens0_nb[:10])
                            if len(tokens0_nb) > 10:
                                pages += ",..."
                        except Exception:
                            pages = ""
                        self.log(
                            f"[WARN] 透明テキスト(検索/選択): OCR tokens=0 かつ内容あり疑いのページが {len(tokens0_nb)} ページあります。"
                            f"{(' pages=' + pages) if pages else ''}"
                        )
                        self.log(
                            "[HINT] 対策: 入力DPIを上げる / 二値化(前処理)を少し上げる / OCR workers=0で試す（並列OCRの空出力を回避）"
                        )
                    elif tokens0:
                        self.log(f"[INFO] 透明テキスト: OCR tokens=0 のページが {len(tokens0)} ページあります（白紙/図版などは正常の可能性）。")

                    if fb_pages:
                        self.log(f"[INFO] OCR fallback: worker空出力のため singleproc に切替えたページ={fb_pages}")
            except Exception:
                pass

            # 埋め込み品質レポート（JSON）
            try:
                if getattr(self, "embed_quality_write_json", False) and getattr(self, "_embed_stats", None):
                    qpath = process_flow.make_unique_path(os.path.splitext(out_final)[0] + ".embed_quality.json")
                    rows = []
                    for st in self._embed_stats:
                        try:
                            rows.append({
                                "page_index": int(getattr(st, "page_index", -1)),
                                "direction": getattr(st, "direction", ""),
                                "total_tokens": int(getattr(st, "total_tokens", 0)),
                                # OCR diagnostics (v81)
                                "ocr_words_data": int(getattr(st, "ocr_words_data", 0)),
                                "ocr_err": str(getattr(st, "ocr_err", "") or ""),
                                "ocr_fallback_used": bool(getattr(st, "ocr_fallback_used", False)),
                                "ocr_ink_ratio": float(getattr(st, "ocr_ink_ratio", 0.0)),
                                "ocr_empty_suspected_nonblank": bool(getattr(st, "ocr_empty_suspected_nonblank", False)),
                                "ocr_empty_reason": str(getattr(st, "ocr_empty_reason", "") or ""),
                                "skipped_empty": int(getattr(st, "skipped_empty", 0)),
                                "attempted_inserts": int(getattr(st, "attempted_inserts", 0)),
                                "ok_inserts": int(getattr(st, "ok_inserts", 0)),
                                "insert_errors": int(getattr(st, "insert_errors", 0)),
                                "error_rate": float(getattr(st, "error_rate", lambda: 0.0)()),
                                "ok_rate": float(getattr(st, "ok_rate", lambda: 0.0)()),
                                "vertical_tokens": int(getattr(st, "vertical_tokens", 0)),
                                "horizontal_tokens": int(getattr(st, "horizontal_tokens", 0)),
                                "vertical_chars": int(getattr(st, "vertical_chars", 0)),
                                "vertical_exception_chars": int(getattr(st, "vertical_exception_chars", 0)),
                                "vertical_chunk_inserts": int(getattr(st, "vertical_chunk_inserts", 0)),
                                "vrun_samples": int(getattr(st, "vrun_samples", 0)),
                                "vrun_target_ratio_mean": float(getattr(st, "vrun_target_ratio_mean", lambda: 0.0)()),
                                "vrun_target_ratio_min": float(getattr(st, "vrun_target_ratio_min", 0.0)),
                                "vrun_target_ratio_max": float(getattr(st, "vrun_target_ratio_max", 0.0)),
                                "vrun_xscale_mean": float(getattr(st, "vrun_xscale_mean", lambda: 0.0)()),
                                "body_rect_used": bool(getattr(st, "body_rect_used", False)),
                                "fallback_stage": int(getattr(st, "fallback_stage", 0)),
                                "fallback_reason": getattr(st, "fallback_reason", ""),
                            })
                        except Exception as e:
                            _log_exception_once('L3095', e)

                    # summary
                    total_pages = len(rows)
                    sum_attempts = sum(r.get("attempted_inserts", 0) for r in rows)
                    sum_ok = sum(r.get("ok_inserts", 0) for r in rows)
                    sum_err = sum(r.get("insert_errors", 0) for r in rows)
                    sum_fb = sum(1 for r in rows if int(r.get("fallback_stage", 0)) > 0)
                    sum_fb1 = sum(1 for r in rows if int(r.get("fallback_stage", 0)) == 1)
                    sum_fb2 = sum(1 for r in rows if int(r.get("fallback_stage", 0)) == 2)
                    sum_tokens0 = sum(1 for r in rows if int(r.get("total_tokens", 0)) <= 0)
                    sum_tokens0_nonblank = sum(1 for r in rows if int(r.get("total_tokens", 0)) <= 0 and bool(r.get("ocr_empty_suspected_nonblank", False)))
                    sum_ocr_fallback = sum(1 for r in rows if bool(r.get("ocr_fallback_used", False)))
                    summary = {
                        "total_pages": total_pages,
                        "attempted_inserts": sum_attempts,
                        "ok_inserts": sum_ok,
                        "insert_errors": sum_err,
                        "error_rate": (float(sum_err) / float(sum_attempts)) if sum_attempts else 0.0,
                        "fallback_pages": sum_fb,
                        "fallback_stage1_pages": sum_fb1,
                        "fallback_stage2_pages": sum_fb2,
                        "tokens0_pages": sum_tokens0,
                        "tokens0_suspected_nonblank_pages": sum_tokens0_nonblank,
                        "ocr_fallback_pages": sum_ocr_fallback,
                        "version": f"{APP_FULLNAME}|embed_quality_v81",
                    }
                    # --- Font usage summary (v80) ---
                    try:
                        fm_counts = dict(getattr(self, "_font_method_counts", {}) or {})
                    except Exception:
                        fm_counts = {}
                    fm_effective = ""
                    try:
                        if fm_counts:
                            fm_effective = max(fm_counts.items(), key=lambda kv: kv[1])[0]
                    except Exception:
                        fm_effective = ""
                    try:
                        self._font_method_effective = str(fm_effective)
                        self.log(
                            f"[INFO] font_usage: embedded={fm_counts.get('embedded',0)} "
                            f"fontfile={fm_counts.get('fontfile',0)} "
                            f"builtin={fm_counts.get('builtin',0)} "
                            f"helv={fm_counts.get('helv',0)} "
                            f"effective={fm_effective} "
                            f"fallback_used={bool(getattr(self, '_font_fallback_used', False))}"
                        )
                    except Exception:
                        pass

                    settings = {
                        "base_dpi": int(getattr(self, "base_dpi", 0)),
                        "jpeg_quality": int(getattr(self, "jpeg_quality", 0)),
                        "bg_scale_percent": int(getattr(self, "bg_scale_percent", 100)),
                        "max_output_dpi": int(getattr(self, "max_output_dpi", 0)),
                        "binarize_strength": int(getattr(self, "binarize_strength", 0)),
                        "text_boldness": int(getattr(self, "text_boldness", 0)),
                        "esrgan_tile": int(getattr(self, "esrgan_tile", 0)),
                        "output_page_mode": str(getattr(self, "output_page_mode", "")),
                        "model_path": str(getattr(self, "model_path", "")),
                        "enable_deskew": bool(getattr(self, "enable_deskew", False)),
                        "deskew_max_deg": float(getattr(self, "deskew_max_deg", 0.0)),
                        "auto_grayscale": bool(getattr(self, "auto_grayscale", False)),
                        "gray_color_ratio_percent": float(getattr(self, "gray_color_ratio_percent", 0.0)),
                        "gray_chroma_p99": float(getattr(self, "gray_chroma_p99", 0.0)),
                        "gray_jpeg_quality_offset": int(getattr(self, "gray_jpeg_quality_offset", 0)),
                        "font_spec": str(getattr(self, "font_spec", "")),
                        "font_path": str(getattr(self, "font_path", "")),
                        "font_method_preferred": str(getattr(self, "_font_method_preferred", "")),
                        "fontname_preferred": str(getattr(self, "_embed_fontname", "") or getattr(self, "_fontname_preferred", "")),
                        "font_method_counts": dict(getattr(self, "_font_method_counts", {}) or {}),
                        "font_fallback_used": bool(getattr(self, "_font_fallback_used", False)),
                        "font_method_effective": str(getattr(self, "_font_method_effective", "")),
                        "fontname_last_used": str(getattr(self, "_fontname_last_used", "")),
                    }
                    payload = {"summary": summary, "settings": settings, "pages": rows}

                    ok, err = _settings_io.atomic_write_json(qpath, payload, indent=2, ensure_ascii=False, make_backup=False)
                    if not ok:
                        raise err if err else RuntimeError("failed to write embed quality json")
                    self.log(f"[INFO] 埋め込み品質レポート: {qpath}")
            except Exception as e:
                self.log(f"[WARN] 埋め込み品質レポート保存に失敗: {e}")
            if cancelled:
                self.log(f"[DONE] 中断（途中結果）: {out_final}")
            else:
                self.log(f"[DONE] 出力PDF: {out_final}")
        else:
            # No pages were written.
            if cancelled:
                out_path = ""
                try:
                    self.log("[INFO] 中断: 出力ページが無いため、ファイルは作成しません。")
                except Exception:
                    pass
            else:
                raise RuntimeError("出力ページが0件です（中断またはエラー）。")

    finally:
        try:
            in_doc.close()
        except Exception as e:
            _log_exception_once('L3131', e)
        try:
            out_doc.close()
        except Exception as e:
            _log_exception_once('L3135', e)

        # Remove intermediate work files (best-effort).
        for _p in (out_work_a, out_work_b):
            try:
                if _p and os.path.exists(_p):
                    os.remove(_p)
            except Exception:
                pass

        # ensure workers end if error/stop (moved to ocr_pipeline)
        if use_mp_ocr:
            try:
                self._shutdown_mp_ocr_workers(task_q, result_q, workers)
            except Exception as e:
                _log_exception_once('shutdown_mp_ocr', e)
    return out_path
