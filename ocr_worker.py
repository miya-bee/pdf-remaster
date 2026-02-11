# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import traceback

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore

YOMITOKU_AVAILABLE = False
try:
    from yomitoku import OCR as YomiTokuOCR
    YOMITOKU_AVAILABLE = True
except Exception:
    YOMITOKU_AVAILABLE = False
    YomiTokuOCR = None  # type: ignore

# Multiprocessing OCR worker (top-level for spawn)
# =========================
def _mp_ocr_worker_loop(
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    device_str: str = "cpu"
):
    """
    Worker process:
      - Initialize YomiToku once
      - Receive tasks: (page_index, png_bytes)
      - Return results: (page_index, words_data, err_str)
         words_data: List[{"text": str, "points": [[x,y],...]}]

    Notes:
      - result_q が満杯のとき、無限ブロックするとメイン側が task_q.put で詰まった瞬間に
        相互待ち（デッドロック）になり得るため、timeout + リトライで永久停止を避ける。
    """
    import queue as _q  # worker内ローカル（spawn環境で安全）

    def _safe_put_result(payload, max_wait_sec: float = 120.0) -> None:
        """result_q.put の永久ブロックを避ける（満杯でも一定間隔でリトライ）"""
        t0 = time.time()
        while True:
            try:
                result_q.put(payload, block=True, timeout=2.0)
                return
            except _q.Full:
                if (time.time() - t0) > max_wait_sec:
                    raise RuntimeError("result_q.put が長時間ブロックしました（デッドロック防止）")
                time.sleep(0.01)
                continue

    try:
        if not YOMITOKU_AVAILABLE:
            raise RuntimeError("YomiToku is not available in worker process.")

        ocr = YomiTokuOCR(visualize=False, device=device_str)

        while True:
            # get() は無限待ちだと停止検知が遅れるためタイムアウト付きで回す
            try:
                item = task_q.get(timeout=2.0)
            except _q.Empty:
                continue
            if item is None:
                break

            page_index, png_bytes = item
            try:
                arr = np.frombuffer(png_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                results, _ = ocr(img)

                words = getattr(results, "words", None)
                if words is None and isinstance(results, dict):
                    words = results.get("words")

                out = []
                if words:
                    for w in words:
                        content = getattr(w, "content", None) if not isinstance(w, dict) else w.get("content")
                        points = getattr(w, "points", None) if not isinstance(w, dict) else w.get("points")
                        if not content or not points or len(points) != 4:
                            continue
                        out.append({
                            "text": str(content),
                            "points": [[float(p[0]), float(p[1])] for p in points],
                        })

                _safe_put_result((page_index, out, ""))

            except Exception:
                _safe_put_result((page_index, [], traceback.format_exc()))

    except Exception:
        # Critical init failure: signal by putting special -1
        try:
            t0 = time.time()
            while True:
                try:
                    result_q.put((-1, [], traceback.format_exc()), block=True, timeout=2.0)
                    break
                except _q.Full:
                    if (time.time() - t0) > 120.0:
                        break
                    time.sleep(0.01)
                    continue
        except Exception:
            # If we cannot notify the parent process via result_q, at least emit to stderr.
            try:
                import sys as _sys
                _sys.stderr.write("[WARN] OCR worker failed to report init failure to parent via result_q.\n")
                _sys.stderr.write(traceback.format_exc())
            except Exception:
                pass


# =========================
