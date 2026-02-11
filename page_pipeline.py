# -*- coding: utf-8 -*-
from __future__ import annotations

"""page_pipeline.py

処理本体（process_pdf）とプレビュー（test_page）の「ページ単位パイプライン」を統一する。

推奨フロー（ページ単位）:
  1) レンダリング（RGB）※レンダリング自体は engine._render_page_to_rgb で実施
  2) SR（元画像の連続階調RGB）                            : ImageOpsMixin._sr_x4
  3) Deskew（角度推定は元画像を低解像度化して行い、推定角をSR画像へ適用）
                                                     : ImageOpsMixin._deskew_rgb（推定）+ 回転適用
     ※「元画像で角度を推定 → SR出力に1回だけ回転を適用」の順が安定しやすい。
  4) 分岐A：背景用（必要なら太字化・縮小）                 : ImageOpsMixin._apply_text_boldness_to_rgb + scale + _encode_jpeg_bytes
  5) 分岐B：OCR用前処理（二値化など）                        : ImageOpsMixin._preprocess_for_ocr

注:
  - 本モジュールは GUI や PDF 合成を扱わない（循環依存回避）。
  - heavy model 初期化は行わない（engine 側のメソッドを呼ぶだけ）。
"""

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

from constants import DEFAULT_BG_SCALE_PERCENT, DEFAULT_MAX_OUTPUT_DPI
from log_utils import _debug_log_exception_once


@dataclass
class PagePipelineOutput:
    """Page-level pipeline artifacts.

    In "final" mode, only a small subset is populated to reduce memory usage.
    """

    page_index: int
    src_rect: "fitz.Rect"  # type: ignore[name-defined]

    # Shared
    deskew_angle: float = 0.0

    # Final artifacts
    bg_jpeg_bytes: Optional[bytes] = None
    bg_w_px: Optional[int] = None
    bg_h_px: Optional[int] = None
    embed_scale: float = 1.0
    ocr_bgr: Optional[np.ndarray] = None
    ocr_w_px: Optional[int] = None
    ocr_h_px: Optional[int] = None

    # Preview artifacts
    sr_rgb: Optional[np.ndarray] = None          # SR(+deskew) preview RGB (jpeg round-trip best-effort)
    sr_rgb_for_ocr: Optional[np.ndarray] = None  # SR(+deskew) raw RGB used as OCR/preproc base (no jpeg)
    bg_preview_rgb: Optional[np.ndarray] = None  # JPEG round-trip preview (best-effort)
    ocr_rgb: Optional[np.ndarray] = None         # OCR preproc as RGB


def _compute_embed_scale(
    engine,
    bg_rgb: np.ndarray,
    src_rect: "fitz.Rect",  # type: ignore[name-defined]
) -> float:
    """Compute effective embed scale (background downscale) consistent with process_runner."""

    # User requested scale
    user_scale = float(getattr(engine, "bg_scale_percent", DEFAULT_BG_SCALE_PERCENT)) / 100.0
    user_scale = max(0.10, min(1.0, user_scale))
    embed_scale = float(user_scale)

    # OutDPI clamp (only for original mode)
    cap_dpi = int(getattr(engine, "max_output_dpi", DEFAULT_MAX_OUTPUT_DPI))
    if getattr(engine, "output_page_mode", "fit") == "original" and cap_dpi > 0:
        try:
            w_pt = float(src_rect.width)
            h_pt = float(src_rect.height)
            if w_pt > 1.0 and h_pt > 1.0:
                inch_w = w_pt / 72.0
                inch_h = h_pt / 72.0
                h0b, w0b = bg_rgb.shape[:2]
                eff_base = max(w0b / inch_w, h0b / inch_h)
                eff_after = eff_base * embed_scale
                if eff_after > float(cap_dpi) * 1.01:
                    dpi_scale = float(cap_dpi) / eff_after
                    dpi_scale = max(0.05, min(1.0, dpi_scale))
                    embed_scale = max(0.05, min(1.0, embed_scale * dpi_scale))
        except Exception as e:
            _debug_log_exception_once("embed_scale_compute", e)
            pass

    return float(embed_scale)


def _resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return rgb
    h0, w0 = rgb.shape[:2]
    nw = max(1, int(w0 * float(scale)))
    nh = max(1, int(h0 * float(scale)))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def run_page_pipeline(
    engine,
    *,
    rgb: np.ndarray,
    src_rect: "fitz.Rect",  # type: ignore[name-defined]
    page_index: int,
    mode: str,
    preview_jpeg_roundtrip: bool = True,
    preview_apply_boldness: bool = False,
    final_apply_boldness: bool = True,
    final_apply_scale: bool = True,
) -> PagePipelineOutput:
    """Run a unified page pipeline.

    Args:
        engine: PdfOcrEnhanceEngine instance.
        rgb: Rendered RGB image.
        src_rect: Source page rect.
        page_index: 0-based page index.
        mode: "preview" or "final".
        preview_jpeg_roundtrip: if True, simulate final JPEG encode/decode for preview.
        preview_apply_boldness: if True, apply text boldness in preview artifacts.
        final_apply_boldness: if True, apply text boldness for output JPEG.
        final_apply_scale: if True, apply bg_scale_percent/max_output_dpi scaling for output.

    Returns:
        PagePipelineOutput
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) が利用できません。")

    if mode not in ("preview", "final"):
        raise ValueError(f"mode must be 'preview' or 'final', got: {mode}")


    # SR (x4)
    sr_rgb = engine._sr_x4(rgb)

    # Deskew: estimate angle on low-res ORIGINAL (more stable than on SR output),
    # then apply that angle to the SR image (single rotation on the high-res output).
    ang = 0.0
    try:
        if bool(getattr(engine, "enable_deskew", True)):
            h0, w0 = rgb.shape[:2]
            max_side = 1200
            scale = min(1.0, float(max_side) / float(max(h0, w0)))
            if scale < 1.0:
                small = cv2.resize(
                    rgb,
                    (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = rgb
            _tmp, ang = engine._deskew_rgb(small)
            ang = float(ang)
    except Exception:
        ang = 0.0

    if abs(float(ang)) > 0.0001:
        try:
            h, w = sr_rgb.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, float(ang), 1.0)
            sr_rgb2 = cv2.warpAffine(
                sr_rgb,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        except Exception:
            sr_rgb2 = sr_rgb
            ang = 0.0
    else:
        sr_rgb2 = sr_rgb
        ang = 0.0

    # Branch B: OCR preproc from deskewed SR image
    ocr_bgr = engine._preprocess_for_ocr(sr_rgb2)

    out = PagePipelineOutput(page_index=int(page_index), src_rect=src_rect, deskew_angle=float(ang))

    if mode == "preview":
        bg_src = sr_rgb2
        if preview_apply_boldness and int(getattr(engine, "text_boldness", 0)) != 0:
            bg_src = engine._apply_text_boldness_to_rgb(bg_src)

        bg_preview = bg_src
        if preview_jpeg_roundtrip and Image is not None:
            try:
                bg_jpeg = engine._encode_jpeg_bytes(bg_src, page_index=page_index)
                bg_preview = np.array(Image.open(io.BytesIO(bg_jpeg)).convert("RGB"))
            except Exception:
                bg_preview = bg_src

        out.sr_rgb = bg_preview
        out.sr_rgb_for_ocr = sr_rgb2
        out.bg_preview_rgb = bg_preview
        out.ocr_bgr = ocr_bgr
        try:
            hh, ww = ocr_bgr.shape[:2]
            out.ocr_w_px = int(ww)
            out.ocr_h_px = int(hh)
        except Exception:
            out.ocr_w_px = None
            out.ocr_h_px = None
        out.ocr_rgb = cv2.cvtColor(ocr_bgr, cv2.COLOR_BGR2RGB)
        return out

    # mode == "final"
    bg_src = sr_rgb2
    if final_apply_boldness and int(getattr(engine, "text_boldness", 0)) != 0:
        bg_src = engine._apply_text_boldness_to_rgb(bg_src)

    embed_scale = 1.0
    if final_apply_scale:
        embed_scale = _compute_embed_scale(engine, bg_src, src_rect)
        if embed_scale < 0.999:
            try:
                bg_src = _resize_rgb(bg_src, embed_scale)
            except Exception:
                embed_scale = 1.0

    h_px, w_px = bg_src.shape[:2]
    bg_jpeg = engine._encode_jpeg_bytes(bg_src, page_index=page_index)

    out.bg_jpeg_bytes = bg_jpeg
    out.bg_w_px = int(w_px)
    out.bg_h_px = int(h_px)
    out.embed_scale = float(embed_scale)
    out.ocr_bgr = ocr_bgr
    try:
        hh, ww = ocr_bgr.shape[:2]
        out.ocr_w_px = int(ww)
        out.ocr_h_px = int(hh)
    except Exception:
        out.ocr_w_px = None
        out.ocr_h_px = None
    return out
