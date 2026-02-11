# -*- coding: utf-8 -*-
from __future__ import annotations

"""Image processing utilities and mixin for PDF_Remaster_v1_0_0.

This module intentionally holds OpenCV-heavy routines (SR tiling, deskew, OCR preproc, boldness)
to reduce risk of accidental GUI/PDF logic regressions and to keep responsibilities separated.
"""

import math
import traceback

# Optional dependency: torch (used for CUDA cache management and GPU-related checks)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


import numpy as np
import cv2

# -----------------------------
# Exception logging (throttled)
# -----------------------------
_LOG_ONCE_KEYS = set()

def _log_exception_once(key: str, exc: BaseException, *, prefix: str = "") -> None:
    """Log an exception only once per key to avoid spamming console."""
    if key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(key)
    try:
        msg = f"[WARN] {prefix}{key}: {type(exc).__name__}: {exc}"
    except Exception:
        msg = f"[WARN] {prefix}{key}: (exception)"
    print(msg)
    try:
        tb = traceback.format_exc()
        if tb and ("NoneType: None" not in tb):
            print(tb)
    except Exception:
        return


def _weighted_median(vals, wts):
    """Weighted median (robust). vals/wts are iterables."""
    pairs = []
    for v, w in zip(vals, wts):
        try:
            w = float(w)
            if w <= 0:
                continue
            pairs.append((float(v), w))
        except Exception:
            continue
    if not pairs:
        return 0.0
    pairs.sort(key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total * 0.5:
            return float(v)
    return float(pairs[-1][0])


def _weighted_mad(vals, wts, med):
    """Weighted median absolute deviation around med."""
    try:
        dev = [abs(float(v) - float(med)) for v in vals]
        return float(_weighted_median(dev, wts)) if dev else 0.0
    except Exception:
        return 0.0


def estimate_deskew_angle_by_hough(
    rgb_img,
    *,
    target_w: int = 1200,
    axis_tol: float = 12.0,
    min_len_ratio: float = 0.16,
    min_len_min: int = 60,
    threshold: int = 100,
    max_line_gap: int = 12,
    max_lines: int = 3000,
    min_count: int = 12,
    min_sumlen: float = 2800.0,
    mad_limit: float = 0.55,
    min_angle: float = 0.30,
):
    """Estimate deskew angle via HoughLinesP.

    Returns:
      (angle_apply_deg, ok, method_tag, mad, count, sumlen)

    angle_apply_deg is the angle to pass to cv2.getRotationMatrix2D (positive = CCW).
    This estimator is designed to be conservative and avoid tilting straight pages.
    """
    try:
        rgb = np.asarray(rgb_img)
        if rgb.ndim != 3 or rgb.shape[0] < 20 or rgb.shape[1] < 20:
            return 0.0, False, "hough-smallimg", 0.0, 0, 0.0

        h0, w0 = int(rgb.shape[0]), int(rgb.shape[1])
        if int(target_w) > 0 and w0 > int(target_w):
            scale = float(target_w) / float(max(1, w0))
            nh = max(24, int(h0 * scale))
            rgb_s = cv2.resize(rgb, (int(target_w), int(nh)), interpolation=cv2.INTER_AREA)
        else:
            rgb_s = rgb

        gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)
        # light binarization for clearer edges
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(bw, 50, 150, apertureSize=3)
        try:
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        except Exception:
            pass

        min_len = int(max(int(min_len_min), float(min_len_ratio) * float(min(edges.shape[0], edges.shape[1]))))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=int(threshold),
            minLineLength=int(min_len),
            maxLineGap=int(max_line_gap),
        )
        if lines is None or len(lines) < 10:
            return 0.0, False, "hough-none", 0.0, 0, 0.0

        cand_h, w_h, sum_h = [], [], 0.0
        cand_v, w_v, sum_v = [], [], 0.0

        tol = float(axis_tol)
        for l in lines[: int(max_lines)]:
            try:
                x1, y1, x2, y2 = l[0]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                ln = (dx * dx + dy * dy) ** 0.5
                if ln < float(min_len):
                    continue
                ang = float(math.degrees(math.atan2(dy, dx)))
                # normalize to [-90, 90]
                if ang < -90.0:
                    ang += 180.0
                if ang > 90.0:
                    ang -= 180.0

                # accept only near-axis
                if abs(ang) <= tol:
                    cand_h.append(ang)
                    w_h.append(ln)
                    sum_h += ln
                elif abs(abs(ang) - 90.0) <= tol:
                    dev = (ang - 90.0) if ang > 0 else (ang + 90.0)
                    cand_v.append(dev)
                    w_v.append(ln)
                    sum_v += ln
            except Exception:
                continue

        best = None  # (sumlen, med, mad, tag, cnt)
        def _try(vals, wts, sumlen, tag):
            nonlocal best
            if len(vals) < int(min_count) or float(sumlen) < float(min_sumlen):
                return
            med = _weighted_median(vals, wts)
            mad = _weighted_mad(vals, wts, med)
            if float(mad) > float(mad_limit):
                return
            if best is None or float(sumlen) > float(best[0]):
                best = (float(sumlen), float(med), float(mad), str(tag), int(len(vals)))

        _try(cand_h, w_h, sum_h, "hough-h")
        _try(cand_v, w_v, sum_v, "hough-v")

        if best is None:
            return 0.0, False, "hough-lowconf", 0.0, 0, 0.0

        sumlen, med, mad, tag, cnt = best
        angle_apply = -float(med)  # rotate by -med to deskew
        if abs(angle_apply) < float(min_angle):
            return 0.0, False, "hough-small", float(mad), int(cnt), float(sumlen)
        return float(angle_apply), True, str(tag), float(mad), int(cnt), float(sumlen)
    except Exception:
        return 0.0, False, "hough-exc", 0.0, 0, 0.0

def apply_text_boldness_to_rgb(rgb_img: np.ndarray, strength: int) -> np.ndarray:
    """Apply view-only text boldness to an RGB image.

    Single source of truth for:
      - final PDF output background (engine/process)
      - preview / zoom viewer

    strength: -100..+100
      0   : no-op
      +   : bolder
      -   : thinner

    Notes:
      - +side geometry is internally scaled (historically step44 was too strong for output).
        We keep the current +gain (0.90) and apply user-requested multipliers on top of it elsewhere.
      - +side appearance (stroke darkness) is intentionally kept close to the older v97 monolithic
        implementation: the *added* stroke region is darkened strongly for a crisp look.
    """
    try:
        s = int(strength)
    except Exception:
        return rgb_img

    s = int(max(-100, min(100, s)))
    if rgb_img is None or s == 0:
        return rgb_img

    strength_abs = abs(s)

    # Internal gain: +side geometry only (keeps output from over-bloating at strength=100)
    strength_geom_f = (float(strength_abs) * 0.90) if s > 0 else float(strength_abs)
    strength_geom_f = max(0.0, min(100.0, strength_geom_f))
    strength_geom = int(strength_geom_f)  # legacy integer for -side paths

    if strength_geom <= 0:
        return rgb_img

    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Otsuで背景/文字を分離（スキャンPDFでは背景が明るい前提）
        otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ------------------------------------------------------------
        # + side (bolder): match older v97 look (crisper & darker strokes)
        # ------------------------------------------------------------
        if s > 0:
            sgf = float(max(0.0, min(100.0, strength_geom_f)))
            sa = int(max(0, min(100, strength_abs)))  # darkness follows the user's slider, not the internal gain

            # Older implementation widened the text mask around 50 (neutral) and kept it piecewise.
            t = int(max(0, min(255, float(otsu_t) + (sgf - 50.0) * 0.4)))
            _, th_bin = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)  # background=255 / text=0
            text = (th_bin == 0).astype(np.uint8)
            # Stroke expansion (v97-like) with a soft transition around breakpoints.
            # We intentionally smooth *both sides* of each breakpoint so there is no jump
            # when crossing it (e.g., 83 -> 84 when +gain makes geom cross 75).
            def _dilate_mask(mask_u8: np.ndarray, rad: int, it: int) -> np.ndarray:
                if int(rad) <= 0:
                    return mask_u8.astype(np.float32)
                k = 1 + 2 * int(rad)  # 3,5,7...
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                d = cv2.dilate(mask_u8, kern, iterations=int(it))
                return d.astype(np.float32)

            # v97 parameter levels: (radius, iterations)
            levels = [(0, 1), (1, 1), (2, 1), (3, 2)]
            bps = [0.0, 15.0, 45.0, 75.0, 100.0]

            # Wider soft-step band for smoother 83..100 progression
            band = 7.0  # half width (in "geom" units)

            def _smootherstep01(x: float) -> float:
                x = float(max(0.0, min(1.0, x)))
                return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

            sgc = float(max(0.0, min(100.0, sgf)))

            def _blend(m_lo: np.ndarray, m_hi: np.ndarray, w01: float) -> np.ndarray:
                w01 = _smootherstep01(w01)
                return (1.0 - w01) * m_lo + w01 * m_hi

            # Piecewise with smooth transitions around each breakpoint
            if sgc <= (bps[1] - band):
                cover = _dilate_mask(text, *levels[0])
            elif sgc < (bps[1] + band):
                w = (sgc - (bps[1] - band)) / (2.0 * band)
                cover = _blend(_dilate_mask(text, *levels[0]), _dilate_mask(text, *levels[1]), w)
            elif sgc <= (bps[2] - band):
                cover = _dilate_mask(text, *levels[1])
            elif sgc < (bps[2] + band):
                w = (sgc - (bps[2] - band)) / (2.0 * band)
                cover = _blend(_dilate_mask(text, *levels[1]), _dilate_mask(text, *levels[2]), w)
            elif sgc <= (bps[3] - band):
                cover = _dilate_mask(text, *levels[2])
            elif sgc < (bps[3] + band):
                w = (sgc - (bps[3] - band)) / (2.0 * band)
                cover = _blend(_dilate_mask(text, *levels[2]), _dilate_mask(text, *levels[3]), w)
            else:
                cover = _dilate_mask(text, *levels[3])

            # Added region only; coverage is float [0..1] for smooth transitions.
            stroke = np.clip(cover - text.astype(np.float32), 0.0, 1.0)
            if float(np.max(stroke)) <= 0.0:
                return rgb_img

            out = rgb_img.copy()

            # Darken only the added stroke region (close to v97)
            alpha = 0.25 + 0.55 * (sa / 100.0)  # 0.25..0.80
            alpha = float(min(0.80, max(0.0, alpha)))

            idx = stroke.astype(bool)
            if int(np.any(idx)):
                scale = (1.0 - (alpha * stroke[idx])).astype(np.float32)
                out[idx] = np.clip(out[idx].astype(np.float32) * scale[:, None], 0, 255).astype(np.uint8)

            return out

        # ------------------------------------------------------------
        # - side (thinner): keep current behavior (erode + inpaint)
        # ------------------------------------------------------------

        # |strength| が高いほどマスク閾値を動かす（-側=狭める）
        bias_scale = 18.0
        bias = int(round((float(strength_geom) / 100.0) * bias_scale))
        thr = int(max(0, min(255, int(otsu_t) - bias)))

        # 文字マスク（黒文字を想定）
        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

        # 細く: 文字ストロークを収縮 → 削れた領域を背景で置換（inpaint優先）
        ks = 3 if strength_geom < 70 else 5
        it2 = 1 if strength_geom < 25 else (2 if strength_geom < 55 else (3 if strength_geom < 85 else 4))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        ero = cv2.erode(th, kernel2, iterations=it2)
        rem = cv2.subtract(th, ero)  # 元文字から削れた部分

        out = rgb_img.copy()
        idx = rem > 0
        if int(np.any(idx)):
            inpainted = None
            try:
                inpainted = cv2.inpaint(out, rem, 2, cv2.INPAINT_TELEA)
            except Exception:
                inpainted = None
            if inpainted is None:
                try:
                    inpainted = cv2.medianBlur(out, 5)
                except Exception:
                    inpainted = out

            alpha = 0.25 + (float(strength_geom) / 100.0) * 0.70  # 0.25..0.95
            out[idx] = np.clip(
                out[idx].astype(np.float32) * (1.0 - alpha) + inpainted[idx].astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

        return out
    except Exception:
        # 失敗しても致命的にしない（背景が調整されないだけ）
        return rgb_img





# -------------------------
# Preview helpers (UI: binarize + residual deskew estimate)
# -------------------------

def binarize_preview_rgb(rgb_img, strength: int):
    """Return an RGB uint8 image representing OCR pre-processing (binarized preview).

    This is used in preview/zoom windows to let users tune binarization strength.
    The goal is a *stable* preview; it does not need to match OCR perfectly.

    UX rule:
      - strength == 0 means "show original" (no binarization).

    Notes:
      - SR/deskew outputs can contain slight ringing halos around strokes.
        Local/adaptive thresholding can turn those halos into "outline" artifacts
        even with small nonzero strength. To suppress that, we apply a gentle
        Gaussian blur for any strength>0 and use a global threshold with a small
        bias controlled by strength (stable, gradual).
    """
    if rgb_img is None:
        return rgb_img
    try:
        strength = int(strength)
    except Exception:
        return rgb_img
    strength = int(max(0, min(100, strength)))
    if strength == 0:
        return rgb_img

    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # gentle blur to suppress halo/outlines from SR
        if strength > 0:
            try:
                k = 3 if strength < 60 else 5
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            except Exception as e:
                _log_exception_once('binarize_preview_rgb.blur', e)

        # strength 1..100 -> threshold 225..95 (small strength keeps more white)
        t = int(225 - 1.3 * strength)
        t = max(40, min(240, t))
        th = (gray >= t).astype(np.uint8) * 255
        rgb = np.stack([th, th, th], axis=-1)
        return rgb
    except Exception as e:
        _log_exception_once('binarize_preview_rgb', e)
        return rgb_img


def estimate_residual_deskew_angle_preview(rgb_img, *, max_abs_deg: float = 3.0):
    """Estimate *small* additional rotation (deskew) for preview windows.

    This helper is intentionally conservative: if confidence is low, it returns 0°
    (do not auto-correct).

    Returns:
      (delta_deg, info_dict)
        - delta_deg: angle to pass to cv2.getRotationMatrix2D (positive = CCW).
        - info_dict: {ok, method, mad, count, sumlen}
    """
    info = {"ok": False, "method": "init", "mad": 0.0, "count": 0, "sumlen": 0.0}

    def _clamp_deg(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        if x < -float(max_abs_deg):
            return -float(max_abs_deg)
        if x > float(max_abs_deg):
            return float(max_abs_deg)
        return float(x)

    # 1) Hough (preferred) - share the same estimator as the output path, but stricter.
    try:
        ang_apply, ok, tag, mad, cnt, sumlen = estimate_deskew_angle_by_hough(
            rgb_img,
            target_w=900,
            axis_tol=10.0,
            min_count=12,
            min_sumlen=0.0,
            mad_limit=0.45,
            min_angle=0.25,
            max_lines=2500,
        )

        # If Hough says it's essentially straight, do NOT fall back to minAreaRect.
        if (not ok) and str(tag) == "hough-small":
            info.update({"ok": False, "method": "hough-straight", "mad": float(mad), "count": int(cnt), "sumlen": float(sumlen)})
            return 0.0, info

        if ok:
            delta = float(ang_apply)
            # if almost straight, do nothing
            if abs(delta) < 0.25:
                info.update({"ok": False, "method": f"{tag}-straight", "mad": float(mad), "count": int(cnt), "sumlen": float(sumlen)})
                return 0.0, info
            info.update({"ok": True, "method": str(tag), "mad": float(mad), "count": int(cnt), "sumlen": float(sumlen)})
            return _clamp_deg(delta), info
    except Exception as e:
        _log_exception_once('estimate_residual_deskew_angle_preview.hough', e)
        info.update({"ok": False, "method": "hough-exc"})

    # 2) Fallback: minAreaRect (very conservative)
    try:
        gray = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if coords.shape[0] < 500:
            info.update({"ok": False, "method": "minarearect-few"})
            return 0.0, info

        rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
        center, size, angle = rect
        w_rect, h_rect = float(size[0]), float(size[1])

        # normalize similar to common OpenCV box logic
        if w_rect < h_rect:
            angle = float(angle) + 90.0
        else:
            angle = float(angle)

        # map to [-45, 45]
        if angle < -45.0:
            angle += 90.0
        elif angle > 45.0:
            angle -= 90.0

        delta = -float(angle)
        if abs(delta) < 0.35:
            info.update({"ok": False, "method": "minarearect-straight", "count": int(coords.shape[0])})
            return 0.0, info
        # Residual auto-correct is for micro-tilt only. If big, refuse.
        if abs(delta) > max_abs_deg:
            info.update({"ok": False, "method": "minarearect-too-big", "count": int(coords.shape[0])})
            return 0.0, info

        info.update({"ok": True, "method": "minarearect", "mad": 0.0, "count": int(coords.shape[0]), "sumlen": 0.0})
        return _clamp_deg(delta), info
    except Exception as e:
        _log_exception_once('estimate_residual_deskew_angle_preview.minarearect', e)
        info.update({"ok": False, "method": "minarearect-exc"})
        return 0.0, info


class ImageOpsMixin:
    """Mixin that provides image-processing methods used by PdfOcrEnhanceEngine.

    These methods are moved out of engine.py to keep the core orchestration code smaller and safer.
    """

    def _choose_effective_tile(self, in_w: int, in_h: int) -> int:
        """入力画像サイズとメモリ見積もりから、ESRGANのtileを安全側に自動調整する。"""
        # ユーザー指定（0はAUTO）
        try:
            user_tile = int(self.esrgan_tile)
        except Exception:
            user_tile = 0
        if user_tile < 0:
            user_tile = 0

        if not getattr(self, "auto_tile_safety", True):
            return user_tile

        # 出力サイズ（SR後）は outscale 倍
        out_w = int(in_w * max(1, int(self.outscale)))
        out_h = int(in_h * max(1, int(self.outscale)))

        # ざっくり見積もり（テンソル/中間特徴量を含めて多めに）
        # - CUDA + half=True なら概ね float16 相当（2byte）で動くので、過度に保守的にならないように補正
        # - CPUはfloat32前提で、係数も大きめに取る
        bytes_per = 2 if self.device_str == "cuda" else 4  # fp16 / fp32
        est_bytes = out_w * out_h * 3 * bytes_per
        est_bytes *= 4 if self.device_str == "cuda" else 8  # 中間バッファ等の係数（安全側）

        ram = self._get_avail_ram_bytes()
        vram = self._get_avail_vram_bytes() if self.device_str == "cuda" else 0
        budget = vram if vram > 0 else ram

        # budgetが取れない環境では、サイズベースで安全に倒す
        if budget <= 0:
            if max(out_w, out_h) >= 3200:
                safe = 256
            else:
                safe = 0
        else:
            ratio = float(est_bytes) / float(max(1, budget))
            # ratioが大きいほど、より小さいタイルを選ぶ
            if ratio < 0.18 and max(out_w, out_h) < 2800:
                safe = 0
            elif ratio < 0.28:
                safe = 512
            elif ratio < 0.40:
                safe = 384
            elif ratio < 0.55:
                safe = 256
            else:
                safe = 192

            # CPUは巨大ページで tile=0 が危険なので、AUTO時は最小限のタイルを入れる
            if self.device_str != "cuda" and safe == 0 and max(out_w, out_h) >= 3000:
                safe = 256

        # clamp（タイルは画像より大きくできない）
        if safe > 0:
            safe = int(max(128, min(safe, min(in_w, in_h))))
            safe = int((safe // 8) * 8)

        # ユーザー指定がある場合の扱い
        # - CUDA環境では、推定は保守的になりがちなので「まずユーザー指定で試し、OOMなら縮めてリトライ」方式にする
        # - CPU環境ではメモリ枯渇時にOSごと不安定になり得るため、安全側に縮める
        if user_tile > 0:
            if safe > 0 and safe < user_tile:
                if self.device_str == "cuda":
                    self.log(f"[WARN] 推定メモリ超過の可能性がありますが、CUDAではまず tile={user_tile} で試行し、OOM時に tile={safe} へフォールバックします。")
                    return user_tile
                else:
                    self.log(f"[SAFE] 推定メモリ超過の可能性があるため tile を {user_tile}→{safe} に自動調整します。")
                    return safe
            return user_tile

        return safe


    # -------------------------
    # YomiToku init (single-process mode only)
    # -------------------------

    def _sr_x4(self, rgb_img: np.ndarray) -> np.ndarray:
        # SRをOFFにできる（公開/配布時の依存削減・安定性向上）
        try:
            if not bool(getattr(self, "enable_sr", True)):
                return rgb_img
        except Exception:
            pass
        # 入力サイズに応じて、安全側に tile を自動調整（AUTOや危険値の場合）
        h_in, w_in = rgb_img.shape[:2]
        desired_tile = self._choose_effective_tile(w_in, h_in)

        # upsampler が未初期化、または tile が変わった場合は再初期化
        if self.upsampler is None or (self._esrgan_tile_effective is not None and int(self._esrgan_tile_effective) != int(desired_tile)):
            self._init_esrgan(tile_override=desired_tile)

        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # SRの分割タイル数（進捗表示の Tile 97/140 の正体）をログに出す
        if getattr(self, "log_sr_tile_stats", True):
            try:
                tile_eff = int(getattr(self, "_esrgan_tile_effective", desired_tile) or 0)
                if tile_eff <= 0:
                    nx = ny = 1
                else:
                    nx = int(math.ceil(float(w_in) / float(tile_eff)))
                    ny = int(math.ceil(float(h_in) / float(tile_eff)))
                total = nx * ny
                key = (w_in, h_in, tile_eff, float(getattr(self, "outscale", 4.0)))
                if getattr(self, "_last_sr_tile_stat", None) != key:
                    self._last_sr_tile_stat = key
                    self.log(f"[INFO] SR input={w_in}x{h_in}px tile={tile_eff} outscale={getattr(self,'outscale',4.0)} -> approx tiles={nx}x{ny}={total}")
                    if total >= 200 and self.device_str != "cuda":
                        self.log("[WARN] SRのタイル総数が多く、CPU環境では処理が大幅に遅くなる可能性があります。DPIを下げる、SRをOFFにする、背景縮小率を下げる等を検討してください。")
            except Exception as e:
                _log_exception_once('L881', e)

        def _is_oom(err: Exception) -> bool:
            msg = str(err).lower()
            return ("out of memory" in msg) or ("cuda" in msg and "memory" in msg) or isinstance(err, MemoryError)

        try:
            out_bgr, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
            return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            if _is_oom(e):
                # 1) まずは tile を小さくして再試行（CUDA/CPU共通）
                self.log("[WARN] Real-ESRGANでメモリ不足の可能性。tileを調整して再試行します。")
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        _log_exception_once('L899', e)

                # 候補を作成（重複は除外）
                cand = []
                if desired_tile <= 0:
                    cand = [256, 192, 128]
                else:
                    cand = [max(128, int(desired_tile // 2)), 256, 192, 128]
                seen = set()
                cand2 = []
                for t in cand:
                    if t and t not in seen:
                        cand2.append(t)
                        seen.add(t)

                for t in cand2:
                    try:
                        self.upsampler = None
                        self._init_esrgan(tile_override=t)
                        out_bgr, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
                        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e2:
                        if not _is_oom(e2):
                            raise
                        continue

                # 2) CUDAならCPUへフォールバックして最後に再試行
                if self.device_str == "cuda":
                    self.log("[WARN] GPUでのSRが難しいため、CPUにフォールバックして再試行します（遅くなります）。")
                    self.device_str = "cpu"
                    self.upsampler = None
                    desired_tile = self._choose_effective_tile(w_in, h_in)
                    self._init_esrgan(tile_override=desired_tile)
                    out_bgr, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
                    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

            raise

    # -------------------------
    # Deskew (apply to SR image for BOTH routes)
    # -------------------------

    def _deskew_rgb(self, rgb_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Automatic skew correction (deskew) wrapper.

        The deskew implementation is kept as a module-level helper to reduce the
        size of this mixin and make it easier to unit-test. The engine calls
        this method.
        """
        return _deskew_rgb_impl(self, rgb_img)



    # -------------------------
    # OCR preprocessing (strength 0-100)
    # -------------------------

    def _preprocess_for_ocr(self, rgb_img: np.ndarray) -> np.ndarray:
        """OCR用の前処理を行い、BGR画像（3ch, uint8）を返す。

        - 入力: RGB
        - 出力: BGR（Tesseract等のOpenCV→BGR系処理と揃えるため）
        - 強度: self.binarize_strength (0-100)
        """
        strength = int(max(0, min(100, int(getattr(self, 'binarize_strength', 0)))))
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # strength <= 0 は「二値化無効」。OCR入力を原画像（BGR）で返す。
        if strength <= 0:
            return bgr

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # blur strength
        if strength < 35:
            k = 1
        elif strength < 75:
            k = 3
        else:
            k = 5
        if k > 1:
            gray = cv2.medianBlur(gray, k)

        otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bias = int(round((strength - 50) / 50 * 30))  # -30 .. +30
        thr = int(max(0, min(255, int(otsu_t) + bias)))

        _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        it_open = 0 if strength < 35 else (1 if strength < 75 else 2)
        if it_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=it_open)

        th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        return th_bgr

    # -------------------------
    # Background boldness (view-only)
    # -------------------------

    def _apply_text_boldness_to_rgb(self, rgb_img: np.ndarray, strength_override=None) -> np.ndarray:
        """閲覧用背景の「文字太さ」を調整する（画像処理）。

        重要: プレビュー/拡大ビューア/実出力で“同一”の関数を使うことで
        「プレビューと実出力で太さがズレる」問題を抑制します。
        """
        try:
            if strength_override is None:
                s0 = int(float(getattr(self, "text_boldness", 0)))
            else:
                s0 = int(float(strength_override))
        except Exception:
            s0 = 0
        return apply_text_boldness_to_rgb(rgb_img, int(s0))

def _deskew_rgb_impl(self, rgb_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Automatic skew correction (deskew).

    目的:
      - 「まっすぐなページを誤って回す」事故を最小化しつつ、
        実際に傾いている場合は安定して補正する。

    方針:
      1) 低解像度でエッジ→HoughLinesPを取り、水平/垂直“に近い”長い線だけで角度推定
         （文字ストロークの斜め成分に引っ張られにくい）
      2) 信頼度が低い場合のみ、従来の minAreaRect をフォールバックとして使用
      3) cover等で符号が曖昧な場合は、+/-/0 をスコア比較して誤回転を避ける
    """
    if not self.enable_deskew:
        return rgb_img, 0.0

    def _clamp_deg(a: float) -> float:
        if getattr(self, "deskew_max_deg", 0) > 0:
            return float(max(-self.deskew_max_deg, min(self.deskew_max_deg, float(a))))
        return float(a)

    def _score_horiz_alignment(rgb: np.ndarray, ang_apply: float) -> float:
        """Score horizontal alignment after applying ang_apply (CCW+).

        We use the total variation of per-row edge counts. When text lines are
        horizontally aligned, edge density tends to cluster in certain rows,
        increasing this score. For cover pages with mostly vertical/graphic
        content, the score improvement is usually small, so we can safely skip
        deskew to avoid introducing tilt.

        Note: ang_apply is the angle passed to cv2.getRotationMatrix2D.
        """
        try:
            h0, w0 = rgb.shape[:2]
            # Keep this cheap: operate on a downscaled version.
            target_w = 900
            if w0 > target_w:
                scale = target_w / float(w0)
                nh = max(24, int(h0 * scale))
                rgb_s = cv2.resize(rgb, (target_w, nh), interpolation=cv2.INTER_AREA)
            else:
                rgb_s = rgb

            if abs(float(ang_apply)) > 1e-6:
                hs, ws = rgb_s.shape[:2]
                M = cv2.getRotationMatrix2D((ws / 2.0, hs / 2.0), float(ang_apply), 1.0)
                rgb_s = cv2.warpAffine(
                    rgb_s,
                    M,
                    (ws, hs),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

            gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)
            bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            edges = cv2.Canny(bw, 50, 150, apertureSize=3)
            row = edges.sum(axis=1).astype(np.float32)
            if row.size < 3:
                return 0.0
            tv = float(np.sum(np.abs(np.diff(row))))
            return tv
        except Exception:
            return 0.0

    # --------- try Hough (preferred) ---------
    angle = 0.0
    ok = False
    method = ""
    try:
        angle_h, ok, method, _mad, _cnt, _sumlen = estimate_deskew_angle_by_hough(
            rgb_img,
            target_w=1200,
            axis_tol=12.0,
            min_count=12,
            min_sumlen=2800.0,
            mad_limit=0.55,
            min_angle=0.30,
            max_lines=3000,
        )
        # If Hough says the page is effectively straight, do NOT fall back to minAreaRect.
        # Falling back tends to introduce a small but visible erroneous tilt on straight pages.
        if (not ok) and str(method) == "hough-small":
            return rgb_img, 0.0
        if ok:
            angle = float(angle_h)
    except Exception:
        ok = False

    # --------- fallback: minAreaRect ---------
    if not ok:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if coords.shape[0] < 500:
            return rgb_img, 0.0

        rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
        try:
            (cx, cy), (w_rect, h_rect), ang2 = rect
            w_rect = float(w_rect)
            h_rect = float(h_rect)
            ang2 = float(ang2)
            # OpenCV version differences: normalize based on box aspect
            if w_rect < h_rect:
                ang2 = ang2 + 90.0
        except Exception:
            ang2 = float(rect[-1])

        if ang2 < -45.0:
            ang2 = ang2 + 90.0
        elif ang2 > 45.0:
            ang2 = ang2 - 90.0

        ang = float(ang2)
        angle = -float(ang)  # rotate opposite (deskew)
        # fallback is more conservative
        if abs(angle) < 1.20:
            return rgb_img, 0.0

    angle = _clamp_deg(angle)
    if abs(angle) < 0.30:
        return rgb_img, 0.0

    # --------- resolve sign + confidence (avoid tilting cover pages) ---------
    try:
        s0 = _score_horiz_alignment(rgb_img, 0.0)
        s_pos = _score_horiz_alignment(rgb_img, float(angle))
        s_neg = _score_horiz_alignment(rgb_img, -float(angle))

        best_score = float(s0)
        best_ang = 0.0
        if float(s_pos) > best_score:
            best_score = float(s_pos)
            best_ang = float(angle)
        if float(s_neg) > best_score:
            best_score = float(s_neg)
            best_ang = -float(angle)

        # If doing nothing is best, skip deskew (prevents erroneous cover-page tilt)
        if abs(best_ang) < 1e-9:
            return rgb_img, 0.0

        # Require a minimum improvement to apply deskew
        if float(s0) > 1e-6:
            if (best_score - float(s0)) / float(s0) < 0.02:  # <2% improvement => skip
                return rgb_img, 0.0
        else:
            # When baseline is near zero, require an absolute signal
            if best_score < 5000.0:
                return rgb_img, 0.0

        angle = float(best_ang)
    except Exception:
        # keep the original estimate if scoring fails
        pass

    h, w = rgb_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(
        rgb_img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, float(angle)
