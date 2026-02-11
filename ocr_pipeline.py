# -*- coding: utf-8 -*-
from __future__ import annotations

"""OCR pipeline helpers (split step).

- Single-process OCR execution
- Multiprocessing OCR worker management
- Conversion of worker output into OcrToken
- Robust fallback when multiprocessing output is empty

This module is intentionally free of GUI imports and does not initialize heavy
models at import time.
"""

import time
import queue
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

from embed import OcrToken
from ocr_worker import _mp_ocr_worker_loop, YOMITOKU_AVAILABLE as WORKER_YOMITOKU_AVAILABLE

from log_utils import _debug_log_exception_once


class OcrPipelineMixin:
    """Mixin: OCR-related helpers.

    Requirements on the host class:
      - self.log(str)
      - self.stop_flag (threading.Event)
      - self.device_str (str)
      - self.ocr (callable or None)
      - self._init_yomitoku() method
      - self._calc_angle_deg(pts) method
    """

    # -------------------------
    # Multiprocessing OCR helpers
    # -------------------------
    def _start_mp_ocr_workers(
        self,
        num_workers: int,
        *,
        device_str: str = "cpu",
    ) -> Tuple[bool, Optional[mp.context.BaseContext], Optional[Any], Optional[Any], List[mp.Process]]:
        """Start OCR worker processes.

        Returns:
            (use_mp, ctx, task_q, result_q, workers)
        """
        try:
            n = int(num_workers)
        except Exception:
            n = 0

        if n <= 0:
            return False, None, None, None, []

        if not WORKER_YOMITOKU_AVAILABLE:
            try:
                self.log("[WARN] YomiToku が見つからないため、OCR並列は無効化します。")
            except Exception:
                pass
            return False, None, None, None, []

        try:
            ctx = mp.get_context("spawn")
            task_q = ctx.Queue(maxsize=max(4, n * 2))
            result_q = ctx.Queue(maxsize=max(8, n * 4))

            workers: List[mp.Process] = []
            try:
                self.log(f"[INFO] OCRワーカープロセス起動: {n} 個（device={device_str}）")
            except Exception:
                pass

            for _ in range(n):
                p = ctx.Process(target=_mp_ocr_worker_loop, args=(task_q, result_q, device_str), daemon=True)
                p.start()
                workers.append(p)

            return True, ctx, task_q, result_q, workers

        except Exception as e:
            try:
                self.log(f"[WARN] OCRワーカー起動に失敗しました: {type(e).__name__}: {e}")
            except Exception:
                pass
            return False, None, None, None, []

    def _mp_drain_results(
        self,
        result_q: Any,
        pending: Dict[int, Dict[str, Any]],
    ) -> None:
        """Drain result_q into pending dict.

        Each result is (page_index, words_data, err_str).
        Worker init failure is signaled by page_index == -1.
        """
        while True:
            try:
                idx, words_data, err = result_q.get_nowait()
            except queue.Empty:
                break

            if idx == -1:
                raise RuntimeError(f"OCR worker initialization failed:\n{err}")

            pending.setdefault(int(idx), {})["words_data"] = words_data
            pending[int(idx)]["ocr_err"] = err

    def _mp_put_task(
        self,
        task_q: Any,
        payload: Any,
        workers: List[Any],
        *,
        on_queue_full: Optional[Callable[[], None]] = None,
        timeout_sec: float = 120.0,
    ) -> None:
        """Non-blocking put with backpressure handling to avoid deadlocks."""
        t0 = time.time()
        while True:
            try:
                if getattr(self, "stop_flag", None) is not None and self.stop_flag.is_set():
                    return
            except Exception:
                pass

            try:
                task_q.put(payload, block=False)
                return
            except queue.Full:
                if on_queue_full is not None:
                    try:
                        on_queue_full()
                    except Exception:
                        pass

                if (time.time() - t0) > float(timeout_sec):
                    alive = -1
                    try:
                        alive = sum(1 for p in workers if getattr(p, "is_alive", lambda: False)())
                    except Exception:
                        alive = -1
                    raise RuntimeError(
                        f"OCRタスク投入が {timeout_sec:.0f} 秒以上進まず停止しました（デッドロック防止で中断）。alive_workers={alive}"
                    )

                time.sleep(0.01)

    # -------------------------
    # Token conversion and fallback
    # -------------------------
    def _wordsdata_to_tokens(self, words_data: List[dict], *, page_no_1based: Optional[int] = None) -> List[OcrToken]:
        tokens: List[OcrToken] = []
        if fitz is None:
            return tokens
        for w in words_data:
            text = w.get("text", "")
            points = w.get("points", [])
            if not text or not points or len(points) != 4:
                continue
            try:
                pts = [(float(p[0]), float(p[1])) for p in points]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                angle = self._calc_angle_deg(pts)
                tokens.append(OcrToken(text=str(text), points=pts, rect=rect, angle=angle))
            except Exception as e:
                _debug_log_exception_once("ocr_wordsdata_to_tokens", e, context={"page_no": int(page_no_1based) if page_no_1based else None})
                continue
        return tokens

    def _tokens_from_wordsdata_or_fallback(
        self,
        words_data: List[dict],
        *,
        fallback_png: Optional[bytes],
        page_no_1based: int,
    ) -> List[OcrToken]:
        """Convert worker output to tokens, with a robust single-process fallback."""
        tokens = self._wordsdata_to_tokens(words_data, page_no_1based=page_no_1based)
        if tokens:
            return tokens

        # Worker produced output but token conversion resulted in 0 tokens (important to surface in debug).
        if (not tokens) and words_data:
            try:
                cnt = int(getattr(self, "_ocr_wordsdata_parse_empty_count", 0)) + 1
                setattr(self, "_ocr_wordsdata_parse_empty_count", cnt)
                if cnt <= 5 or (cnt % 50 == 0):
                    self.log(
                        f"[WARN] OCR worker returned words_data but token conversion produced 0 tokens on page {page_no_1based}. "
                        f"words_data={len(words_data)}. This may cause invisible text to be missing."
                    )
            except Exception as e:
                _debug_log_exception_once("ocr_wordsdata_parse_empty_warn", e)

        if not fallback_png:
            return tokens

        if np is None or cv2 is None:
            return tokens

        try:
            arr = np.frombuffer(fallback_png, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                try:
                    self.log(f"[WARN] OCR fallback decode failed on page {page_no_1based} (imdecode returned None)")
                except Exception:
                    pass
                return tokens

            tokens_fb = self._run_ocr_singleproc(img)
            if tokens_fb:
                # Record per-page fallback usage (for later reporting)
                try:
                    pages = getattr(self, "_ocr_fallback_pages", None)
                    if isinstance(pages, set):
                        pages.add(int(page_no_1based))
                except Exception:
                    pass
                _fc = getattr(self, "_ocr_fallback_count", 0) + 1
                setattr(self, "_ocr_fallback_count", _fc)
                if _fc == 1:
                    try:
                        self.log(
                            "[WARN] OCR worker produced empty output; falling back to single-process OCR on affected pages. "
                            "If this happens frequently, set OCR workers=0."
                        )
                    except Exception:
                        pass
                if _fc <= 5 or (_fc % 50 == 0):
                    try:
                        self.log(f"[INFO] OCR fallback (singleproc) succeeded on page {page_no_1based}: tokens={len(tokens_fb)}")
                    except Exception:
                        pass
                return tokens_fb

        except Exception as e:
            try:
                self.log(f"[WARN] OCR fallback failed on page {page_no_1based}: {type(e).__name__}: {e}")
            except Exception:
                pass

        return tokens

    # -------------------------
    # Single-process OCR execution
    # -------------------------
    def _run_ocr_singleproc(self, ocr_bgr: "np.ndarray") -> List[OcrToken]:
        """Run OCR in the main process and return tokens."""
        if getattr(self, "ocr", None) is None:
            self._init_yomitoku()

        try:
            results, _ = self.ocr(ocr_bgr)
        except Exception:
            # fall back to CPU re-init
            try:
                self.log("[WARN] YomiToku OCRが失敗。CPUで再初期化して再試行します。")
            except Exception:
                pass
            try:
                self.device_str = "cpu"
            except Exception:
                pass
            self.ocr = None
            self._init_yomitoku()
            results, _ = self.ocr(ocr_bgr)

        words = getattr(results, "words", None)
        if words is None and isinstance(results, dict):
            words = results.get("words")

        tokens: List[OcrToken] = []
        if not words:
            return tokens

        if fitz is None:
            return tokens

        for w in words:
            content = getattr(w, "content", None) if not isinstance(w, dict) else w.get("content")
            points = getattr(w, "points", None) if not isinstance(w, dict) else w.get("points")
            if not content or not points or len(points) != 4:
                continue

            try:
                pts = [(float(p[0]), float(p[1])) for p in points]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                angle = self._calc_angle_deg(pts)
                tokens.append(OcrToken(text=str(content), points=pts, rect=rect, angle=angle))
            except Exception:
                continue

        return tokens
    def _shutdown_mp_ocr_workers(self, task_q: Any, result_q: Any, workers: List[Any]) -> None:
        """Best-effort shutdown of multiprocessing OCR workers (safe on Windows).

        Goals:
        - Always exit (no hang) even if queues are full or workers are stuck.
        - Prefer graceful exit (sentinel) but fall back to terminate.
        - Close queues / cancel join threads to avoid shutdown deadlocks on Windows.

        Args:
            task_q: OCR task queue (payloads / sentinel None)
            result_q: OCR result queue
            workers: worker process objects
        """
        if not workers:
            return

        # 1) Ask workers to exit by sending sentinel None (one per worker).
        # Use non-blocking puts to avoid hanging if the queue is full.
        try:
            if task_q is not None:
                deadline = time.time() + 2.0
                sent = 0
                while sent < len(workers) and time.time() < deadline:
                    try:
                        task_q.put(None, block=False)
                        sent += 1
                    except queue.Full:
                        # Make a little room / time for workers to consume.
                        time.sleep(0.02)
                    except Exception:
                        break
        except Exception:
            pass

        # 2) Give workers a moment to exit gracefully.
        for p in workers:
            try:
                if hasattr(p, "join"):
                    p.join(timeout=0.8)
            except Exception:
                pass

        # 3) Terminate any remaining workers.
        for p in workers:
            try:
                if hasattr(p, "is_alive") and p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

        # 4) Final join / close processes.
        for p in workers:
            try:
                if hasattr(p, "join"):
                    p.join(timeout=1.5)
            except Exception:
                pass
            try:
                if hasattr(p, "close"):
                    p.close()
            except Exception:
                pass

        # 5) Close queues to avoid interpreter shutdown deadlocks (Windows).
        for q in (task_q, result_q):
            try:
                if q is None:
                    continue
                if hasattr(q, "close"):
                    q.close()
                if hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
            except Exception:
                pass

        # Ensure they are gone
        for p in workers:
            try:
                if hasattr(p, "is_alive") and p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    if hasattr(p, "close"):
                        p.close()
                except Exception:
                    pass
            except Exception:
                pass

