# -*- coding: utf-8 -*-
from __future__ import annotations

"""
PDF Remaster_v1_0_0
Professional-grade pipeline:
- Render PDF pages (PyMuPDF) -> Real-ESRGAN SR -> optional deskew -> background image
- OCR branch: strong binarization -> YomiToku OCR (words + quad points)
- Merge: background image + invisible text aligned by BBox (render_mode=3)

Enhancements included:
- CUDA auto-detect with CPU fallback
- Settings UI (DPI / JPEG quality / binarize strength / ESRGAN tile / output page mode / font / model weights)
- Settings persistence with unique filename + portable mode
- Reset defaults + detailed help
- Model weights guide dialog + weights folder auto-create + model auto-detect combobox
- Memory hardening: frequent store_shrink + periodic close/reopen output doc
- Optional pipeline parallelism: SR (main) + OCR (multiprocessing worker(s)) producer-consumer
- Optional deskew (auto skew correction) applied to BOTH background and OCR branch for coordinate consistency

Windows note:
- Multiprocessing on Windows requires "if __name__ == '__main__':" guard (already present) and freeze_support for PyInstaller.
"""

import os
import io
import sys
import gc
import json
import time
import math
import statistics
import queue
import glob
import threading
import traceback
import webbrowser
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import cv2
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import pdf_io  # PDF I/O wrappers (safer split)
import pdf_compose  # PDF composition (background + invisible text)
import process_flow  # process helpers (unique path / flushing)
from image_ops import ImageOpsMixin
from ocr_pipeline import OcrPipelineMixin
from text_embed import TextEmbedMixin

# Unified page pipeline (shared by final/preview)
from page_pipeline import run_page_pipeline

# -----------------------------
# Exception logging (throttled)
# -----------------------------
from log_utils import _log_exception_once


# Multiprocessing (optional OCR pipeline)
import multiprocessing as mp


# Split modules
from embed import OcrToken, OcrLine, EmbedOptions, EmbedPageStats, EmbedPageMetrics


# =========================================
# Shared constants (side-effect free)
# =========================================
from constants import (
    APP_FULLNAME,
    APP_NAME,
    APP_VERSION,
    CONFIG_FILENAME,
    DEFAULT_AUTO_GRAYSCALE,
    DEFAULT_BG_SCALE_PERCENT,
    DEFAULT_BINARIZE_STRENGTH,
    DEFAULT_DESKEW_MAX_DEG,
    DEFAULT_DPI,
    DEFAULT_ENABLE_DESKEW,
    DEFAULT_ESRGAN_TILE,
    DEFAULT_FONT_PATH,
    DEFAULT_GRAY_CHROMA_P99,
    DEFAULT_GRAY_CHROMA_THRESHOLD,
    DEFAULT_GRAY_COLOR_RATIO_PERCENT,
    DEFAULT_GRAY_JPEG_QUALITY_OFFSET,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_OUTPUT_DPI,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_OCR_WORKERS,
    DEFAULT_OUTPUT_PAGE_MODE,
    DEFAULT_READING_MODE,
    DEFAULT_STORE_SHRINK,
    DEFAULT_TEXT_BOLDNESS,
    REAL_ESRGAN_ANIME_DOC_URL,
    REAL_ESRGAN_RELEASES_URL,
)

# Torch / ESRGAN / YomiToku
try:
    import torch
except Exception:
    torch = None

REAL_ESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REAL_ESRGAN_AVAILABLE = True
except Exception:
    REAL_ESRGAN_AVAILABLE = False

YOMITOKU_AVAILABLE = False
try:
    from yomitoku import OCR as YomiTokuOCR
    YOMITOKU_AVAILABLE = True
except Exception:
    YOMITOKU_AVAILABLE = False


# =========================
# Backend Engine
# =========================
class PdfOcrEnhanceEngine(ImageOpsMixin, OcrPipelineMixin, TextEmbedMixin):
    """
    Pipeline:
      - Render page -> SR -> (optional deskew) -> background JPEG bytes
      - OCR preproc -> YomiToku
      - Merge into PDF with invisible text

    Optional performance mode:
      - Multiprocessing OCR worker(s) (producer-consumer):
          main thread does SR (GPU/CPU),
          worker processes do OCR concurrently (CPU recommended).
    """

    def __init__(
        self,
        log_cb: Callable[[str], None],
        progress_cb: Callable[[int, int], None],
        stop_flag: threading.Event,
        model_path: str,
        outscale: int = 4,
        base_dpi: int = DEFAULT_DPI,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        bg_scale_percent: int = DEFAULT_BG_SCALE_PERCENT,
        max_output_dpi: int = DEFAULT_MAX_OUTPUT_DPI,
        binarize_strength: int = DEFAULT_BINARIZE_STRENGTH,  # 0-100
        text_boldness: int = DEFAULT_TEXT_BOLDNESS,          # -100..+100: 閲覧用の文字太さ（-細く / +太く）
        esrgan_tile: int = DEFAULT_ESRGAN_TILE,              # 0 disables tiling
        output_page_mode: str = DEFAULT_OUTPUT_PAGE_MODE,    # "original" or "pixel"
        font_path: str = DEFAULT_FONT_PATH,                  # TTF/OTF for embedding
        ocr_workers: int = DEFAULT_OCR_WORKERS,              # 0 disables mp pipeline
        enable_deskew: bool = DEFAULT_ENABLE_DESKEW,
        deskew_max_deg: float = DEFAULT_DESKEW_MAX_DEG,
        store_shrink: int = DEFAULT_STORE_SHRINK,            # 0 disables store_shrink calls
        auto_grayscale: bool = DEFAULT_AUTO_GRAYSCALE,
        gray_color_ratio_percent: float = DEFAULT_GRAY_COLOR_RATIO_PERCENT,
        gray_chroma_p99: float = DEFAULT_GRAY_CHROMA_P99,
        gray_jpeg_quality_offset: int = DEFAULT_GRAY_JPEG_QUALITY_OFFSET,
        weights_dir: str = "",
        enable_sr: bool = True,
    ):
        self.log = log_cb
        self.progress = progress_cb
        self.stop_flag = stop_flag

        # ESRGAN重みパス：UI入力/設定ファイルの取り込み時に、余計な引用符/空白/改行を除去し正規化する
        mp = (model_path or "").strip().strip('"').strip("'")
        mp = os.path.expandvars(os.path.expanduser(mp))
        # Windowsでも / と \ が混在しがちなので正規化
        mp = os.path.normpath(mp) if mp else ""
        self.model_path = mp
        self.outscale = int(outscale)

        self.base_dpi = int(base_dpi)
        self.jpeg_quality = int(jpeg_quality)
        self.bg_scale_percent = int(bg_scale_percent)
        self.max_output_dpi = int(max_output_dpi)
        # JPEG軽量化: ほぼ白黒ページの自動グレースケール化
        self.auto_grayscale = bool(auto_grayscale)
        try:
            self.gray_color_ratio_percent = float(gray_color_ratio_percent)
        except Exception:
            self.gray_color_ratio_percent = float(DEFAULT_GRAY_COLOR_RATIO_PERCENT)
        self.gray_color_ratio_percent = max(0.0, min(5.0, self.gray_color_ratio_percent))
        try:
            self.gray_chroma_p99 = float(gray_chroma_p99)
        except Exception:
            self.gray_chroma_p99 = float(DEFAULT_GRAY_CHROMA_P99)
        self.gray_chroma_p99 = max(5.0, min(80.0, self.gray_chroma_p99))

        # GrayAuto時の追加圧縮（グレーページのみ）
        try:
            self.gray_jpeg_quality_offset = int(gray_jpeg_quality_offset)
        except Exception:
            self.gray_jpeg_quality_offset = int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET)
        self.gray_jpeg_quality_offset = int(max(-40, min(0, self.gray_jpeg_quality_offset)))

        self.binarize_strength = int(max(0, min(100, binarize_strength)))
        try:
            _tb = int(float(text_boldness))
        except Exception:
            _tb = int(DEFAULT_TEXT_BOLDNESS)
        self.text_boldness = int(max(-100, min(100, _tb)))
        self.esrgan_tile = int(max(0, esrgan_tile))
        # 安全装置: tile=0(AUTO) でも巨大画像では自動的にタイリングを強制する
        self.auto_tile_safety = True
        self._esrgan_tile_effective = None  # 実際に使っているtile（_sr_x4で決定）

        self.output_page_mode = (output_page_mode or "original").strip().lower()
        if self.output_page_mode not in ("original", "pixel"):
            self.output_page_mode = "original"

        self.weights_dir = (weights_dir or "").strip()

        # SR enable/disable (Safe mode may turn this OFF)
        try:
            self.enable_sr = bool(enable_sr)
        except Exception:
            self.enable_sr = True

        # フォント指定（AUTO or パス）
        self.font_spec = (font_path or "").strip()
        self.font_path = self._resolve_font_path(self.font_spec, self.weights_dir)
        # 透明テキスト用フォント（CJK固定）: 出力PDFに埋め込むフォント名キャッシュ
        self._embed_fontname = None  # type: Optional[str]
        self._embed_font_ready = False
        # --- Font method tracking (v80) ---
        self._font_method_preferred = ""    # 'embedded' | 'fontfile' | 'builtin' | 'helv'
        self._fontname_preferred = ""
        self._fontname_last_used = ""
        self._font_method_counts = {"embedded": 0, "fontfile": 0, "builtin": 0, "helv": 0}
        self._font_fallback_used = False
        self._font_method_logged = False
        self._font_method_effective = ""

        # 埋め込み品質の“見える化”と自動フォールバック
        self.embed_quality_debug = True
        self.embed_quality_write_json = True
        self.embed_fallback_enabled = True
        self.embed_fallback_min_attempts = 30
        self.embed_fallback_error_rate = 0.02  # errors/attempts
        self.embed_fallback_ok_rate = 0.90     # ok/attempts
        # Staged fallback: try moderate settings before final safe mode
        self.embed_fallback_staged = True
        self.embed_fallback_stage1_chunk_divisor = 2   # e.g. 6 -> 3
        self.embed_fallback_stage1_min_chunk = 2
        self.embed_fallback_stage1_force_xscale_1 = True
        self.embed_fallback_stage1_force_rotate0 = False
        self.embed_fallback_stage2_force_xscale_1 = True
        self.embed_fallback_stage2_force_rotate0 = True
        self._embed_stats: List[EmbedPageStats] = []


        self.ocr_workers = int(max(0, ocr_workers))
        self.enable_deskew = bool(enable_deskew)
        self.deskew_max_deg = float(max(0.0, deskew_max_deg))
        self.store_shrink = int(max(0, min(100, store_shrink)))

        # Reading direction: auto/vertical/horizontal (GUI override)
        self.reading_direction_mode = str(DEFAULT_READING_MODE).lower()

        self.device = self._select_device()
        self.device_str = "cuda" if (self.device is not None and getattr(self.device, "type", "") == "cuda") else "cpu"

        self.upsampler: Optional[Any] = None
        self.ocr: Optional[Any] = None

    # -------------------------
    # Device select
    # -------------------------
    
    
    def _select_device(self) -> Optional["torch.device"]:
        if torch is None:
            self.log("[WARN] torch が import できません。GPU/CPU判定不可。CPU相当で進行します。")
            return None
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                self.log(f"[INFO] {APP_FULLNAME} 起動: CUDA 利用可能: GPU={name}")
            except Exception:
                self.log(f"[INFO] {APP_FULLNAME} 起動: CUDA 利用可能")
        else:
            self.log(f"[WARN] {APP_FULLNAME} 起動: CUDA が利用できません。CPUモードで実行します（処理に時間がかかります）。")
        return dev

    # -------------------------
    # Real-ESRGAN init
    # -------------------------
    def _init_esrgan(self, tile_override: int | None = None):
        if not REAL_ESRGAN_AVAILABLE:
            raise RuntimeError("Real-ESRGAN 関連が見つかりません。realesrgan / basicsr を確認してください。")

        # モデル重み(.pth)の解決：入力パスに余計な空白/引用符が混ざることがあるため、候補を作って確実に拾う
        mp_raw = (self.model_path or "")
        candidates = []
        if mp_raw:
            candidates.append(mp_raw)
            candidates.append(os.path.normpath(mp_raw))
            candidates.append(os.path.normpath(mp_raw.strip().strip('"').strip("'")))
        # 相対指定の場合は weights_dir からも探索
        if self.weights_dir:
            base = os.path.abspath(self.weights_dir)
            if mp_raw and not os.path.isabs(mp_raw):
                candidates.append(os.path.normpath(os.path.join(base, mp_raw)))
            # 典型ファイル名での自動探索
            candidates.append(os.path.normpath(os.path.join(base, DEFAULT_MODEL_FILENAME)))

        # 実在する最初の候補を採用
        resolved = ""
        for c in candidates:
            try:
                if c and os.path.isfile(c):
                    resolved = c
                    break
            except Exception as e:
                _log_exception_once('L666', e)

        if not resolved:
            # どの候補も見つからない場合：診断情報（repr）を含めて案内
            diag = " / ".join([repr(c) for c in candidates[:6]])
            raise FileNotFoundError(
                f"Real-ESRGANモデル重みが見つかりません: {mp_raw}\n"
                f"候補: {diag}\n"
                f"例: weights/{DEFAULT_MODEL_FILENAME} を配置し、GUIで指定してください。"
            )

        self.model_path = resolved

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        use_half = (self.device_str == "cuda")

        # tile決定：
        #  - GUIのtileが 0/負値 → AUTO扱い
        #  - _sr_x4 から tile_override が渡された場合はそれを優先
        try:
            tile = int(tile_override) if tile_override is not None else int(self.esrgan_tile)
        except Exception:
            tile = 0
        if tile < 0:
            tile = 0

        # 安全装置：CPUで tile=0（全画像一括）は高DPI/大ページでRAMを一気に消費しやすいので、
        # AUTO時はデフォルトでタイル分割を有効化する（必要なら _sr_x4 が画像サイズに応じて上書きする）
        if getattr(self, "auto_tile_safety", True):
            if tile == 0 and self.device_str != "cuda":
                tile = 256

        self._esrgan_tile_effective = tile

        self.log(f"[INFO] Real-ESRGAN 初期化: device={self.device_str}, half={use_half}, tile={tile} (user={self.esrgan_tile}, override={tile_override})")
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half
        )

    
    # -------------------------
    # ESRGAN tile safety helpers
    # -------------------------
    def _get_avail_ram_bytes(self) -> int:
        """Windowsの空き物理メモリを取得（失敗時は0）"""
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception as e:
            _log_exception_once('L738', e)
        return 0

    def _get_avail_vram_bytes(self) -> int:
        """CUDAが使える場合に推定VRAM（空きに近い値は取れないため総量を返す）。"""
        try:
            if torch is not None and torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                return int(getattr(p, "total_memory", 0) or 0)
        except Exception as e:
            _log_exception_once('L748', e)
        return 0


    def _init_yomitoku(self):
        if not YOMITOKU_AVAILABLE:
            raise RuntimeError("YomiToku が見つかりません。yomitoku をインストールしてください。")
        self.log(f"[INFO] YomiToku OCR 初期化: device={self.device_str}")
        self.ocr = YomiTokuOCR(visualize=False, device=self.device_str)

    # -------------------------
    # PDF page -> RGB image (numpy)
    # -------------------------
    def _render_page_to_rgb(self, page: fitz.Page) -> np.ndarray:
        dpi = max(72, int(self.base_dpi))
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if getattr(self, "log_render_stats", True):
            try:
                # ページごとのレンダ結果（DPIが効いているか確認用）
                self.log(f"[INFO] Render: dpi={dpi} -> {pix.width}x{pix.height}px")
            except Exception as e:
                _log_exception_once('L846', e)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)

    # -------------------------
    # Super-resolution x4
    # -------------------------


    def _calc_angle_deg(self, pts: List[Tuple[float, float]]) -> float:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return math.degrees(math.atan2((y1 - y0), (x1 - x0) + 1e-6))

    def _snap_rotate(self, angle: float) -> int:
        cands = [0, 90, 180, 270]
        a = angle % 360
        best = min(cands, key=lambda v: abs(v - a))
        return best if abs(best - a) <= 10 else 0

    def _scale_tokens(self, tokens: List[OcrToken], sx: float, sy: float) -> List[OcrToken]:
        if not tokens:
            return []
        sx = float(sx)
        sy = float(sy)
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return tokens
        out: List[OcrToken] = []
        for t in tokens:
            try:
                pts = [(float(p[0]) * sx, float(p[1]) * sy) for p in (t.points or [])]
                r = t.rect
                rect = fitz.Rect(float(r.x0) * sx, float(r.y0) * sy, float(r.x1) * sx, float(r.y1) * sy)
                out.append(OcrToken(text=str(t.text), points=pts, rect=rect, angle=float(getattr(t, "angle", 0.0))))
            except Exception:
                # スケール失敗時はそのまま
                out.append(t)
        return out

    # -------------------------
    # Group tokens into lines (stability)
    # -------------------------
    def _tokens_to_lines(self, tokens: List[OcrToken]) -> List[OcrLine]:
        if not tokens:
            return []

        heights = [max(1.0, t.rect.height) for t in tokens]
        med_h = float(np.median(heights))
        y_tol = med_h * 0.55

        def rot_of(t: OcrToken) -> int:
            return self._snap_rotate(t.angle)

        tokens_sorted = sorted(tokens, key=lambda t: (rot_of(t), (t.rect.y0 + t.rect.y1) * 0.5, t.rect.x0))

        clusters: List[List[OcrToken]] = []
        cluster_y: List[float] = []
        cluster_rot: List[int] = []

        for t in tokens_sorted:
            yc = (t.rect.y0 + t.rect.y1) * 0.5
            r = rot_of(t)

            best_i = -1
            best_d = 1e18
            for i, (cy, cr) in enumerate(zip(cluster_y, cluster_rot)):
                if cr != r:
                    continue
                d = abs(yc - cy)
                if d <= y_tol and d < best_d:
                    best_d = d
                    best_i = i

            if best_i < 0:
                clusters.append([t])
                cluster_y.append(yc)
                cluster_rot.append(r)
            else:
                clusters[best_i].append(t)
                cluster_y[best_i] = (cluster_y[best_i] * 0.8) + (yc * 0.2)

        lines: List[OcrLine] = []
        for toks in clusters:
            r = rot_of(toks[0])
            toks = sorted(toks, key=lambda t: t.rect.x0) if r in (0, 180) else sorted(toks, key=lambda t: t.rect.y0)

            parts = [toks[0].text]
            for a, b in zip(toks, toks[1:]):
                gap = (b.rect.x0 - a.rect.x1) if r in (0, 180) else (b.rect.y0 - a.rect.y1)
                if gap > med_h * 0.45:
                    parts.append(" ")
                parts.append(b.text)
            text = "".join(parts)

            x0 = min(t.rect.x0 for t in toks)
            y0 = min(t.rect.y0 for t in toks)
            x1 = max(t.rect.x1 for t in toks)
            y1 = max(t.rect.y1 for t in toks)
            rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
            fs = max(4.0, float(np.median([t.rect.height for t in toks])) * 0.90)

            bx = rect.x0
            by = rect.y1 - rect.height * 0.15
            baseline = fitz.Point(float(bx), float(by))

            lines.append(OcrLine(text=text, rect=rect, rotate=r, fontsize=fs, baseline=baseline, tokens=toks))

        lines.sort(key=lambda ln: (ln.rotate, ln.rect.y0, ln.rect.x0))
        return lines

    # -------------------------
    # Background encoding (JPEG)
    # -------------------------
    def _encode_jpeg_bytes(self, rgb_img: np.ndarray, page_index: Optional[int] = None) -> bytes:
        q = int(max(10, min(100, self.jpeg_quality)))
        pil = Image.fromarray(rgb_img)
        decision_gray = False  # GrayAuto判定

        # できるだけ容量を減らすための軽い最適化:
        # - ほぼ白黒ページは自動で L(グレースケール) に落とす（1chで大幅に小さくなる）
        #   判定は Lab 色度(chroma) から「色のある画素比率」と「p99」を見る（紙の黄ばみ等は許容）
        # - subsampling=2(4:2:0) + progressive + optimize
        try:
            if self.auto_grayscale and pil.mode == "RGB" and isinstance(rgb_img, np.ndarray) and rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                h, w = int(rgb_img.shape[0]), int(rgb_img.shape[1])
                max_side = max(h, w) if (h > 0 and w > 0) else 0
                if max_side > 0:
                    # 判定用に縮小（最大 256px）して高速化
                    if max_side > 256:
                        s = 256.0 / float(max_side)
                        sw = max(16, int(round(w * s)))
                        sh = max(16, int(round(h * s)))
                        samp = cv2.resize(rgb_img[:, :, :3], (sw, sh), interpolation=cv2.INTER_AREA)
                    else:
                        step = max(1, int(round(max_side / 128)))
                        samp = rgb_img[::step, ::step, :3]

                    lab = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                    a = lab[:, :, 1].astype(np.float32) - 128.0
                    b = lab[:, :, 2].astype(np.float32) - 128.0
                    # ページ全体の色かぶり（紙の黄ばみ等）を中央値で補正してから色度を計算
                    a0 = float(np.median(a))
                    b0 = float(np.median(b))
                    da = a - a0
                    db = b - b0
                    chroma = np.sqrt(da * da + db * db)

                    if chroma.size > 0:
                        thr = float(DEFAULT_GRAY_CHROMA_THRESHOLD)
                        colored_ratio = float(np.mean(chroma > thr))
                        p99 = float(np.percentile(chroma, 99))

                        # colored_ratio が小さく、かつ極端な色が無い場合はグレー化
                        decision_gray = (colored_ratio < (float(self.gray_color_ratio_percent) / 100.0)) and (p99 < float(self.gray_chroma_p99))
                        try:
                            if page_index is None:
                                ptag = "?"
                            else:
                                ptag = str(int(page_index) + 1)
                            self.log(f"[INFO] GrayAuto: {'Y' if decision_gray else 'N'} page={ptag} ratio={colored_ratio*100.0:.3f}% p99={p99:.2f} thr={float(DEFAULT_GRAY_CHROMA_THRESHOLD):.1f} p99_thr={float(self.gray_chroma_p99):.1f}")
                        except Exception as e:
                            _log_exception_once('L1285', e)

                        if decision_gray:
                            gray = cv2.cvtColor(rgb_img[:, :, :3], cv2.COLOR_RGB2GRAY)
                            pil = Image.fromarray(gray)
        except Exception as e:
            _log_exception_once('L1291', e)


        # GrayAutoでグレー化したページは追加圧縮を適用（品質オフセット: 例 -10）
        q_eff = int(q)
        try:
            if decision_gray:
                q_eff = int(max(25, min(100, int(q) + int(getattr(self, 'gray_jpeg_quality_offset', 0)))))
        except Exception:
            q_eff = int(q)

        buf = io.BytesIO()
        try:
            pil.save(buf, format="JPEG", quality=int(q_eff), optimize=True, progressive=True, subsampling=2)
        except Exception:
            # Pillowバージョン差異等で失敗した場合はフォールバック
            pil.save(buf, format="JPEG", quality=int(q_eff), optimize=True)
        out_bytes = buf.getvalue()
        try:
            if decision_gray and page_index is not None:
                self.log(f"[INFO] BGJPEG: page={int(page_index)+1} mode={pil.mode} q={int(q_eff)} bytes={len(out_bytes)}")
        except Exception as e:
            _log_exception_once('L1313', e)
        return out_bytes

    # -------------------------
    # Insert invisible text with font fallback
    # -------------------------

    def _get_page_dominant_direction(self, tokens: List[OcrToken]) -> str:
        """ページ内の角度傾向から、ページが「縦書き主体(vertical)」か「横書き主体(horizontal)」かを推定する。"""
        # GUI override
        mode = str(getattr(self, 'reading_direction_mode', 'auto')).lower().strip()
        if mode in ('vertical', 'horizontal'):
            return mode

        if not tokens:
            return "horizontal"

        vert_count = 0
        for t in tokens:
            a = float(getattr(t, "angle", 0.0)) % 360.0
            # 角度が ±45〜±135（≒90） または 225〜315（≒270）を縦書き相当とみなす
            if (45.0 <= a <= 135.0) or (225.0 <= a <= 315.0):
                vert_count += 1

        ratio = vert_count / max(1, len(tokens))
        return "vertical" if ratio > 0.4 else "horizontal"

    
    def _compute_reading_tolerance(self, tokens: List[OcrToken]) -> float:
        """
        読み順ソート用の許容誤差 tol を動的に決める。

        固定値だと、段組・小さい注釈・ルビ等で「同じ行/列」に属するはずの
        トークンが別クラスタになりやすい。
        そこで、ページ内トークンの「高さ（bboxのy1-y0）」中央値から tol を導出する。
        """
        hs: List[float] = []
        for t in tokens or []:
            try:
                h = float(t.rect.y1) - float(t.rect.y0)
                if h > 0:
                    hs.append(h)
            except Exception:
                continue

        if not hs:
            return 10.0

        hs.sort()
        med_h = hs[len(hs) // 2]

        # 中央値高さに連動（小さな注釈は小さく、本文は大きく）
        tol = med_h * 0.80

        # 極端に小さすぎ/大きすぎるのを防ぐ
        if tol < 3.0:
            tol = 3.0
        elif tol > 20.0:
            tol = 20.0
        return float(tol)

    def _compute_vertical_column_tolerance(self, tokens: List[OcrToken], base_tol: float) -> float:
            """
            縦書きページ用: 列クラスタリングの許容誤差（X方向）を推定する。

            base_tol（bbox高さ中央値由来）だけだと、段組や細かい注釈で
            「列の揺らぎ」と「列間距離」を区別しにくく、順序が崩れやすい。
            そこで、トークンのX中心座標分布から「列ピッチ（代表的な列間隔）」を推定し、
            base_tol と併用する。
            """
            try:
                bt = float(base_tol)
            except Exception:
                bt = 10.0

            xs: List[float] = []
            for t in tokens or []:
                try:
                    x0 = float(t.rect.x0)
                    x1 = float(t.rect.x1)
                    xs.append((x0 + x1) * 0.5)
                except Exception:
                    continue

            # トークンが少ないと推定が不安定なのでフォールバック
            if len(xs) < 6:
                return bt

            xs.sort(reverse=True)  # 右→左

            diffs: List[float] = []
            for i in range(len(xs) - 1):
                d = xs[i] - xs[i + 1]
                if d > 0:
                    diffs.append(d)

            if not diffs:
                return bt

            # 同一列内の微小なX揺らぎを除外し、列間距離っぽい差分だけを拾う
            thresh = max(bt * 1.2, 2.0)
            big = [d for d in diffs if d >= thresh]

            # 列間差分が十分に取れない場合はフォールバック
            if len(big) < 3:
                return bt

            big.sort()
            pitch = big[len(big) // 2]  # 列ピッチ（中央値）

            # ピッチの一部（だいたい半分強）を許容幅にすると列クラスタが安定しやすい
            tol_x = max(bt, pitch * 0.55)

            # 極端値の抑制（スキャン解像度やページサイズで暴れないように）
            if tol_x < 3.0:
                tol_x = 3.0
            elif tol_x > 80.0:
                tol_x = 80.0

            return float(tol_x)


    def _compute_vertical_column_anchors(self, tokens: List[OcrToken], tol_x: float, body_rect: Optional[fitz.Rect] = None) -> Dict[str, Any]:
        """縦書きの列揺らぎを減らすため、列ごとの代表X（左端/中心）を推定する（token座標系=px）。

        目的:
          - OCRのbboxが行内で左右に揺れると、縦書きrun-mode（回転一括挿入）の選択範囲がジグザグになりやすい。
          - 列ごとの代表Xに“軽くスナップ”させて、同一列の選択範囲を揃える。

        注意:
          - 本文領域推定（body_rect）がある場合は、その内側のみで列代表値を推定する。
            （傍注・図表キャプション等が混ざると列推定が崩れるため）
        """
        if not tokens:
            return {"tol_x": float(tol_x) if tol_x else 0.0, "col_x0": {}, "col_cx": {}, "body_rect": body_rect}

        try:
            tol_x_f = float(tol_x)
        except Exception:
            tol_x_f = 0.0
        if tol_x_f <= 1e-6:
            return {"tol_x": tol_x_f, "col_x0": {}, "col_cx": {}, "body_rect": body_rect}

        groups_x0: Dict[int, List[float]] = {}
        groups_cx: Dict[int, List[float]] = {}

        for t in tokens:
            try:
                r = getattr(t, "rect", None)
                if r is None:
                    continue
                x0 = float(getattr(r, "x0", 0.0))
                x1 = float(getattr(r, "x1", 0.0))
                y0 = float(getattr(r, "y0", 0.0))
                y1 = float(getattr(r, "y1", 0.0))
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
            except Exception:
                continue

            if body_rect is not None:
                try:
                    if not body_rect.contains(fitz.Point(cx, cy)):
                        continue
                except Exception:
                    # contains が失敗した場合は除外しない（保守）
                    pass

            col_id = int(cx / tol_x_f)
            groups_x0.setdefault(col_id, []).append(x0)
            groups_cx.setdefault(col_id, []).append(cx)

        col_x0: Dict[int, float] = {}
        col_cx: Dict[int, float] = {}

        for cid, arr in groups_x0.items():
            if not arr:
                continue
            try:
                col_x0[cid] = float(statistics.median(arr))
            except Exception:
                col_x0[cid] = float(sum(arr) / len(arr))

        for cid, arr in groups_cx.items():
            if not arr:
                continue
            try:
                col_cx[cid] = float(statistics.median(arr))
            except Exception:
                col_cx[cid] = float(sum(arr) / len(arr))

        return {"tol_x": tol_x_f, "col_x0": col_x0, "col_cx": col_cx, "body_rect": body_rect}

    def _estimate_body_region(self, tokens: List[OcrToken], w_px: int, h_px: int, direction: str) -> Optional[fitz.Rect]:
        """本文領域（メインテキストブロック）を簡易推定する。

        目的:
          - 段組/小さい注（ルビ・注釈・図表キャプション等）が本文順序に割り込むのを減らす
          - 推定した本文領域内トークンを優先して並べるための判定に使う

        方針（軽量ヒューリスティック）:
          1) トークン高さの中央値を基準に「本文っぽいサイズ」を選別
          2) 余白（ページ上下左右の一定割合）を除外して候補を作る
          3) 候補のBBox（分位点）から本文の外接矩形を作る
          4) 候補が少なすぎる場合は推定を諦める（None）
        """
        if not tokens or w_px <= 0 or h_px <= 0:
            return None

        # トークン高さ中央値
        heights = []
        for t in tokens:
            try:
                h = float(t.rect.y1) - float(t.rect.y0)
                if h > 0.5:
                    heights.append(h)
            except Exception as e:
                _log_exception_once('L1700', e)
        if not heights:
            return None

        med_h = float(statistics.median(heights))

        # 余白（ヘッダー/フッター/傍注を本文推定から外す）
        margin_x = max(8.0, float(w_px) * 0.05)
        margin_y = max(8.0, float(h_px) * 0.05)

        # 本文候補：中央値近辺のサイズ & 余白外
        candidates = []
        for t in tokens:
            try:
                x0, y0, x1, y1 = float(t.rect.x0), float(t.rect.y0), float(t.rect.x1), float(t.rect.y1)
                h = y1 - y0
                if h <= 0.5:
                    continue
                if x0 <= margin_x or x1 >= (float(w_px) - margin_x):
                    continue
                if y0 <= margin_y or y1 >= (float(h_px) - margin_y):
                    continue

                # 本文は「極端に小さい注」より大きいことが多い
                if h < med_h * 0.75:
                    continue
                # 極端に大きい見出し（1語だけ大きい）を避けるため上限も軽く設ける
                if h > med_h * 2.2:
                    continue

                candidates.append((x0, y0, x1, y1))
            except Exception as e:
                _log_exception_once('L1732', e)

        # 候補が少なければ緩和（余白だけ守ってサイズ条件を緩める）
        if len(candidates) < max(12, int(len(tokens) * 0.06)):
            candidates = []
            for t in tokens:
                try:
                    x0, y0, x1, y1 = float(t.rect.x0), float(t.rect.y0), float(t.rect.x1), float(t.rect.y1)
                    h = y1 - y0
                    if h <= 0.5:
                        continue
                    if x0 <= margin_x or x1 >= (float(w_px) - margin_x):
                        continue
                    if y0 <= margin_y or y1 >= (float(h_px) - margin_y):
                        continue
                    if h < med_h * 0.65:
                        continue
                    candidates.append((x0, y0, x1, y1))
                except Exception as e:
                    _log_exception_once('L1751', e)

        if len(candidates) < max(10, int(len(tokens) * 0.04)):
            return None

        xs0 = sorted([c[0] for c in candidates])
        ys0 = sorted([c[1] for c in candidates])
        xs1 = sorted([c[2] for c in candidates])
        ys1 = sorted([c[3] for c in candidates])

        def _pct(arr, p):
            if not arr:
                return 0.0
            k = int(round((len(arr) - 1) * float(p)))
            k = max(0, min(len(arr) - 1, k))
            return float(arr[k])

        # 5〜95% で外れ値を除去した本文外接矩形を作る
        x0 = _pct(xs0, 0.05)
        y0 = _pct(ys0, 0.05)
        x1 = _pct(xs1, 0.95)
        y1 = _pct(ys1, 0.95)

        # パディング（本文推定を少し広げて誤除外を減らす）
        pad = max(6.0, med_h * 0.40)
        x0 = max(0.0, x0 - pad)
        y0 = max(0.0, y0 - pad)
        x1 = min(float(w_px), x1 + pad)
        y1 = min(float(h_px), y1 + pad)

        # 変な矩形（極端に小さい/広がりがない）なら無効
        if (x1 - x0) < float(w_px) * 0.20 or (y1 - y0) < float(h_px) * 0.20:
            return None

        return fitz.Rect(x0, y0, x1, y1)

    def _reorder_tokens_with_body_priority(
        self,
        sorted_tokens: List[OcrToken],
        body_rect: Optional[fitz.Rect],
        w_px: int,
        h_px: int
    ) -> List[OcrToken]:
        """本文領域を優先して並べ替える（各グループ内の順序は保持）。

        グルーピング:
          - header: 上部余白（ページ上端近く）
          - footer: 下部余白（ページ下端近く）
          - body: 本文推定矩形内
          - other: それ以外（傍注・図表・キャプション等）

        目的:
          - 本文中に注釈が割り込む頻度を低減
        """
        if not sorted_tokens:
            return []

        margin_y = max(8.0, float(h_px) * 0.05)

        header: List[OcrToken] = []
        body: List[OcrToken] = []
        other: List[OcrToken] = []
        footer: List[OcrToken] = []

        for t in sorted_tokens:
            try:
                x0, y0, x1, y1 = float(t.rect.x0), float(t.rect.y0), float(t.rect.x1), float(t.rect.y1)
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
            except Exception:
                other.append(t)
                continue

            if cy <= margin_y * 1.2:
                header.append(t)
                continue
            if cy >= float(h_px) - (margin_y * 1.2):
                footer.append(t)
                continue

            if body_rect is not None:
                try:
                    if body_rect.contains(fitz.Point(cx, cy)):
                        body.append(t)
                    else:
                        other.append(t)
                except Exception:
                    other.append(t)
            else:
                body.append(t)

        return header + body + other + footer


    def _sort_tokens_reading_order(self, tokens: List[OcrToken], direction: str) -> List[OcrToken]:
        """読み順（コピー＆ペースト順序）を考慮してトークンをソートする。"""
        if not tokens:
            return []

        # 許容誤差（行/列の揺らぎ）
        tol = self._compute_reading_tolerance(tokens)

        if str(direction).lower() == "vertical":
            # 縦書き: 右→左 (列: X降順)、同じ列なら上→下 (Y昇順)
            # base tol（高さ中央値由来）に加え、X分布から列ピッチを推定して列クラスタを安定化
            tol_x = self._compute_vertical_column_tolerance(tokens, tol)

            def _x_center(tok: OcrToken) -> float:
                try:
                    return (float(tok.rect.x0) + float(tok.rect.x1)) * 0.5
                except Exception:
                    try:
                        return float(getattr(tok.rect, "x0", 0.0))
                    except Exception:
                        return 0.0

            def _y0(tok: OcrToken) -> float:
                try:
                    return float(tok.rect.y0)
                except Exception:
                    return 0.0

            def _x0(tok: OcrToken) -> float:
                try:
                    return float(tok.rect.x0)
                except Exception:
                    return 0.0

            return sorted(tokens, key=lambda t: (-int(_x_center(t) / tol_x), _y0(t), _x0(t)))
        else:
            # 横書き: 上→下 (Y昇順)、同じ行なら左→右 (X昇順)
            return sorted(tokens, key=lambda t: (int(float(t.rect.y0) / tol), float(t.rect.x0)))

    def _tategaki_exception_style(self, ch: str) -> str:
        """縦書き時に「縦中横/回転」に近い扱いをする例外文字を判定する。

        return: '' | 'punct' | 'rotate'
          - punct : 句読点（、。など）→セル右上寄せ（回転なし）
          - rotate: 括弧・長音など → 90/270度回転して配置
        """
        if not ch:
            return ""

        # 句読点（縦組みでは右上寄せされることが多い）
        punct = {"、", "。", "，", "．", ",", "."}
        if ch in punct:
            return "punct"

        # 括弧類（縦組みでは回転されることが多い）
        brackets = set("()（）[]［］{}｛｝〈〉《》「」『』〔〕【】")
        if ch in brackets:
            return "rotate"

        # 長音・ダッシュ類（縦組みでは縦方向に回転されることが多い）
        dashes = {"ー", "―", "—", "–", "－", "〜", "～"}
        if ch in dashes:
            return "rotate"

        return ""


    def _estimate_token_coord_scale(self, tokens: List[OcrToken], w_px: int, h_px: int) -> Tuple[float, float]:
        """OCRトークン座標系（px）が背景画像のpxサイズと一致しているか推定し、補正スケールを返す。

        OCR前処理（拡大/縮小・クロップ等）で token.rect の座標系が背景画像とズレると、
        透明文字の位置が一括でズレます。tokenの右端/下端の分布（p95）を w_px/h_px と比較し、
        必要なら補正係数 (tx, ty) を返します。通常は (1.0, 1.0)。
        """
        try:
            W = float(w_px) if w_px else 0.0
            H = float(h_px) if h_px else 0.0
            if W <= 0.0 or H <= 0.0 or not tokens:
                return 1.0, 1.0

            xs: List[float] = []
            ys: List[float] = []
            for t in tokens:
                try:
                    r = getattr(t, "rect", None)
                    x1 = float(getattr(r, "x1", 0.0))
                    y1 = float(getattr(r, "y1", 0.0))
                    if x1 > 0:
                        xs.append(x1)
                    if y1 > 0:
                        ys.append(y1)
                except Exception:
                    continue
            if not xs or not ys:
                return 1.0, 1.0

            def _percentile(vals: List[float], q: float) -> float:
                s = sorted(vals)
                q = max(0.0, min(1.0, float(q)))
                k = (len(s) - 1) * q
                f = int(math.floor(k))
                c = int(math.ceil(k))
                if f == c:
                    return float(s[f])
                return float(s[f]) * (c - k) + float(s[c]) * (k - f)

            p95x = _percentile(xs, 0.95)
            p95y = _percentile(ys, 0.95)
            if p95x <= 1e-6 or p95y <= 1e-6:
                return 1.0, 1.0

            tx = 1.0 if (0.85 * W) <= p95x <= (1.15 * W) else (W / p95x)
            ty = 1.0 if (0.85 * H) <= p95y <= (1.15 * H) else (H / p95y)

            def _sanitize(v: float) -> float:
                if not (0.20 <= float(v) <= 5.00):
                    return 1.0
                if abs(float(v) - 1.0) < 0.02:
                    return 1.0
                return float(v)

            tx = _sanitize(tx)
            ty = _sanitize(ty)
            return tx, ty
        except Exception:
            return 1.0, 1.0


    def _get_invisible_fitz_font(self, doc: fitz.Document) -> Optional[fitz.Font]:
        """透明テキスト挿入に使うフォントのメトリクス取得用（キャッシュあり）。"""
        try:
            cache_key = (getattr(self, "font_path", "") or "", getattr(self, "_embed_fontname", "") or "")
            if not hasattr(self, "_fitz_font_cache"):
                self._fitz_font_cache = {}
            if cache_key in self._fitz_font_cache:
                return self._fitz_font_cache[cache_key]

            fp = getattr(self, "font_path", "") or ""
            if fp and os.path.isfile(fp):
                try:
                    f = fitz.Font(fontfile=fp)
                    self._fitz_font_cache[cache_key] = f
                    return f
                except Exception as e:
                    _log_exception_once('L1990', e)

            # try embedded / builtin fontname
            try:
                fname = self._ensure_embed_fontname(doc)
                try:
                    f = fitz.Font(fontname=fname)
                    self._fitz_font_cache[cache_key] = f
                    return f
                except Exception as e:
                    _log_exception_once('L2000', e)
            except Exception as e:
                _log_exception_once('L2002', e)

            try:
                f = fitz.Font(fontname="japan")
                self._fitz_font_cache[cache_key] = f
                return f
            except Exception:
                return None
        except Exception:
            return None

    def _compute_embed_page_metrics(self, tokens: List[OcrToken], scale_x: float, scale_y: float) -> EmbedPageMetrics:
        """ページ内トークンの代表値（中央値）から、縦書きのフォントサイズ暴走などを抑えるメトリクスを作る。"""
        if not tokens:
            return EmbedPageMetrics()

        v_pitches: List[float] = []
        v_colws: List[float] = []
        h_heights: List[float] = []

        def _median(vals: List[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            n = len(s)
            mid = n // 2
            if n % 2 == 1:
                return float(s[mid])
            return (float(s[mid - 1]) + float(s[mid])) * 0.5

        for t in tokens:
            try:
                txt = (t.text or "").strip()
                if not txt:
                    continue
                x0 = float(t.rect.x0) * float(scale_x)
                y0 = float(t.rect.y0) * float(scale_y)
                x1 = float(t.rect.x1) * float(scale_x)
                y1 = float(t.rect.y1) * float(scale_y)
                rw = max(0.0, x1 - x0)
                rh = max(0.0, y1 - y0)

                ang = float(getattr(t, "angle", 0.0)) % 360.0
                is_v = (45.0 <= ang <= 135.0) or (225.0 <= ang <= 315.0)

                if is_v:
                    nchar = len(list(txt))
                    if nchar > 0 and rh > 0.0:
                        v_pitches.append(float(rh) / float(nchar))
                    if rw > 0.0:
                        v_colws.append(float(rw))
                else:
                    if rh > 0.0:
                        h_heights.append(float(rh))
            except Exception:
                continue

        v_pitch_med = _median(v_pitches)
        v_colw_med = _median(v_colws)

        v_fs_med = 0.0
        if v_pitch_med > 0.0 and v_colw_med > 0.0:
            v_fs_med = min(v_pitch_med, v_colw_med) * 0.90
        elif v_pitch_med > 0.0:
            v_fs_med = v_pitch_med * 0.90
        elif v_colw_med > 0.0:
            v_fs_med = v_colw_med * 0.90

        return EmbedPageMetrics(
            v_pitch_med=float(v_pitch_med),
            v_colw_med=float(v_colw_med),
            v_fs_med=float(v_fs_med),
            h_h_med=float(_median(h_heights)),
            v_samples=len(v_pitches),
            h_samples=len(h_heights),
        )


    def _insert_token_precise(
        self,
        page: fitz.Page,
        token: OcrToken,
        scale_x: float,
        scale_y: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        shape: Optional[fitz.Shape] = None,
        xscale: float = 1.0,
        stats: Optional[EmbedPageStats] = None,
        opt: Optional[EmbedOptions] = None,
        metrics: Optional[EmbedPageMetrics] = None,
        col_ctx: Optional[Dict[str, Any]] = None,
    ):
        """トークンを不可視テキストとして埋め込む（位置ズレ修正版）。"""
        if opt is None:
            opt = EmbedOptions()

        def _clamp(v: float, lo: float, hi: float) -> float:
            try:
                return max(float(lo), min(float(hi), float(v)))
            except Exception:
                return float(v)

        # font metrics (for baseline shift / width). Best effort.
        inv_font: Optional[fitz.Font] = None
        asc_desc_half: Optional[float] = None
        try:
            inv_font = self._get_invisible_fitz_font(page.parent)
            if inv_font is not None:
                asc_desc_half = (float(getattr(inv_font, "ascender", 1.0)) + float(getattr(inv_font, "descender", -0.3))) * 0.5
        except Exception:
            inv_font = None
            asc_desc_half = None


        # スケール適用後のRect（px→pt）
        x0 = float(offset_x) + float(token.rect.x0) * float(scale_x)
        y0 = float(offset_y) + float(token.rect.y0) * float(scale_y)
        x1 = float(offset_x) + float(token.rect.x1) * float(scale_x)
        y1 = float(offset_y) + float(token.rect.y1) * float(scale_y)
        rect_w = float(x1 - x0)
        rect_h = float(y1 - y0)

        txt = (token.text or "").strip()
        if not txt:
            if stats is not None:
                stats.skipped_empty += 1
            return

        angle = float(getattr(token, "angle", 0.0)) % 360.0

        # 方向指定（auto廃止）:
        # - opt.page_direction があればそれを最優先（passごとに変える場合に備える）
        # - なければ GUI で選ばれた self.reading_direction_mode を使う
        forced_dir = ""
        try:
            forced_dir = str(getattr(opt, "page_direction", "") or "").lower().strip() if opt is not None else ""
        except Exception:
            forced_dir = ""

        base_mode = str(getattr(self, "reading_direction_mode", "") or "").lower().strip()
        if base_mode not in ("vertical", "horizontal"):
            # 防御: 何らかの理由で未設定なら縦をデフォルト（縦書きPDFでの誤判定被害が大きいため）
            base_mode = "vertical"

        mode = forced_dir if forced_dir in ("vertical", "horizontal") else base_mode

        # 45-135(下向き), 225-315(上向き) を縦書き候補とみなす
        angle_vertical = (45.0 <= angle <= 135.0) or (225.0 <= angle <= 315.0)

        # 基本は指定モードに従う（ただし混在レイアウトの最低限の例外補正だけ行う）
        if mode == "vertical":
            # 縦書き指定: 角度が取れないOCRでも縦扱いに寄せる
            is_vertical = True
            # ただし「明らかに横長」なトークンは横書きとして扱う（見出し/英数字行など）
            if rect_w > rect_h * 1.4 and (angle < 45.0 or angle > 315.0):
                is_vertical = False
        else:
            # 横書き指定
            is_vertical = False
            # 例外的に、角度がしっかり縦を示す場合は縦扱い（欄外注など）
            if angle_vertical and rect_h > rect_w * 0.9:
                is_vertical = True


        # helper: counted insert (Shape経由でcommitをまとめると高速)
        def _try_insert(point: fitz.Point, text: str, fontsize: float, rotate: int, xscale_: float):
            if stats is not None:
                stats.attempted_inserts += 1
            try:
                if shape is not None:
                    self._insert_invisible_text_shape(shape, page, point, text, fontsize, rotate=int(rotate), xscale=float(xscale_))
                else:
                    self._insert_invisible_text(page, point, text, fontsize, rotate=int(rotate), xscale=float(xscale_))
                if stats is not None:
                    stats.ok_inserts += 1
                return True
            except Exception:
                # Shape経由が失敗した場合のみ、page.insert_text で再試行（成功すればエラー扱いしない）
                if shape is not None:
                    try:
                        self._insert_invisible_text(page, point, text, fontsize, rotate=int(rotate), xscale=float(xscale_))
                        if stats is not None:
                            stats.ok_inserts += 1
                        return True
                    except Exception as e:
                        _log_exception_once('L2188', e)
                if stats is not None:
                    stats.insert_errors += 1
                return False

        if is_vertical:
            if stats is not None:
                stats.vertical_tokens += 1

            # --- 縦書き：改行を抑える「トークン単位」埋め込み（コピー/検索の実用性を優先） ---
            # 1文字ずつY座標を変えて埋め込むと、多くのPDFビューアで「1文字=1行」と解釈され、
            # コピー時に各文字の後ろへ改行が入ってしまうことがあります。
            # そこで、縦書きトークンは「回転した1行テキスト」として1回で埋め込み、
            # xscale（ベースライン方向スケール）で高さ（rect_h）にフィットさせます。
            #
            # 注意: 句読点/括弧などの縦中横風の見た目再現は「不可視テキスト」には本質的に不要ですが、
            # 選択範囲の精密一致を重視する場合は vertical_run_mode を False にして文字単位に戻せます。
            if getattr(opt, "vertical_run_mode", True):
                fontname = self._ensure_embed_fontname(page.parent)
                # フォントサイズは「幅に合わせる」が基本。CJKは概ね正方形なので 1.2 係数でbbox幅を合わせる。
                fs_by_w = max(1.0, rect_w / 1.20)
                # 高さ方向の平均セルを下限にする（過大になりすぎないよう抑制）
                fs_by_h = max(1.0, rect_h / float(max(1, len(txt))))
                fs = min(fs_by_w, fs_by_h) * 0.98

                # 横書きのテキスト長（pt）を測り、縦方向（rect_h）に合わせてベースライン方向をスケールする
                try:
                    base_len = float(fitz.get_text_length(txt, fontname=fontname, fontsize=float(fs)))
                except Exception:
                    base_len = float(len(txt)) * float(fs)

                # run長さ補正（bboxがタイトで選択範囲が短く見える対策）
                try:
                    len_boost = float(getattr(opt, 'vertical_run_len_boost', 1.0) or 1.0)
                except Exception:
                    len_boost = 1.0
                len_boost = max(0.90, min(1.25, float(len_boost)))
                rect_h_base = float(rect_h)

                # bboxがタイトな場合、ページ全体の縦ピッチ中央値から「期待高さ」を推定して補正
                rect_h_target_base = float(rect_h_base)
                try:
                    if metrics is not None:
                        v_pitch = float(getattr(metrics, "v_pitch_med", 0.0) or 0.0)
                        if v_pitch > 0.0:
                            nchar = max(1, len(txt))
                            est = float(nchar) * float(v_pitch)
                            # 2% 以上短いときだけ補正し、過大補正は抑制
                            if est > float(rect_h_base) * 1.02:
                                rect_h_target_base = min(float(est), float(rect_h_base) * 1.22)
                except Exception:
                    rect_h_target_base = float(rect_h_base)

                rect_h_target = float(rect_h_target_base) * float(len_boost)
                # v76: 伸長しても「開始位置（上端）」は旧挙動に合わせて固定する（下方向に伸ばす）
                py_run = float(y0)

                if base_len > 1e-3:
                    xscale_run = float(rect_h_target) / float(base_len)
                else:
                    xscale_run = 1.0

                # 暴れ抑制（過度な歪みを避ける）
                xscale_run = max(0.35, min(2.85, float(xscale_run)))

                # fit diagnostics
                if stats is not None:
                    try:
                        stats.vrun_samples += 1
                        stats.vrun_rect_h_sum += float(rect_h_base)
                        stats.vrun_target_h_sum += float(rect_h_target)
                        ratio = (float(rect_h_target) / float(rect_h_base)) if float(rect_h_base) > 1e-6 else 1.0
                        stats.vrun_target_ratio_sum += float(ratio)
                        stats.vrun_xscale_sum += float(xscale_run)
                        stats.vrun_target_ratio_min = min(float(stats.vrun_target_ratio_min), float(ratio))
                        stats.vrun_target_ratio_max = max(float(stats.vrun_target_ratio_max), float(ratio))
                    except Exception as e:
                        _log_exception_once('VRUN_STATS', e, prefix='embed ')

                # 270度回転: 文字列の進行が「上→下」になり、コピー時も自然に連続します
                rot_run = 270

                # rotate=270 のbboxは概ね [x = px-0.2*fs .. px+1.0*fs], [y = py .. py+len*fs]
                # なので px を左端に合わせるには 0.2*fs だけ右へ寄せる
                px = x0
                # 列スナップ（同一列内の左右揺らぎを軽減）
                try:
                    if col_ctx is not None and isinstance(col_ctx, dict):
                        tol_x_px = float(col_ctx.get("tol_x", 0.0) or 0.0)
                        col_x0_map = col_ctx.get("col_x0", {}) or {}
                        br = col_ctx.get("body_rect", None)

                        rr = getattr(token, "rect", None)
                        if rr is not None and tol_x_px > 1e-6:
                            tx0 = float(getattr(rr, "x0", 0.0))
                            tx1 = float(getattr(rr, "x1", 0.0))
                            ty0 = float(getattr(rr, "y0", 0.0))
                            ty1 = float(getattr(rr, "y1", 0.0))
                            tcx = (tx0 + tx1) * 0.5
                            tcy = (ty0 + ty1) * 0.5

                            in_body = True
                            if br is not None:
                                try:
                                    in_body = bool(br.contains(fitz.Point(tcx, tcy)))
                                except Exception:
                                    in_body = True

                            if in_body:
                                cid = int(tcx / tol_x_px)
                                col_left = col_x0_map.get(cid, None)
                                if col_left is not None:
                                    # 過度なスナップ（傍注等の飛び）を避ける
                                    if abs(float(col_left) - float(tx0)) <= (tol_x_px * 0.75):
                                        x0 = float(offset_x) + float(col_left) * float(scale_x)
                                        x1 = x0 + rect_w  # rect_w は元token幅を維持
                except Exception as e:
                    _log_exception_once('L2265', e)

                px = x0 + (fs * 0.20)
                py = py_run

                _try_insert(fitz.Point(px, py), txt, fs, rotate=int(rot_run), xscale_=float(xscale_run))
                return

            chars = list(txt)
            if not chars:
                return

            char_count = len(chars)
            # 1文字あたりのセル高さ（位置は必ず元BBox内に収める）
            cell_h = rect_h / float(max(1, char_count))

            # 中心X座標
            center_x = (x0 + x1) * 0.5

            # 縦書き修正: まとめて埋め込まず、1文字ずつ座標計算して配置する
            for i, ch in enumerate(chars):
                style = self._tategaki_exception_style(ch)

                # セルのY中心
                cell_cy = y0 + (cell_h * i) + (cell_h * 0.5)

                # 基本フォントサイズ: 枠からはみ出ないよう安全率をかける
                # 縦書きは文字が詰まって見えることが多いため、少し小さめ(0.85-0.9)が安全
                base_fs = min(rect_w, cell_h) * 0.90
                # ページ代表値でクランプ（BBoxが粗いトークンで fontsize が暴走するのを防ぐ）
                if metrics is not None and getattr(metrics, "v_fs_med", 0.0) > 0.0 and getattr(metrics, "v_samples", 0) >= 6:
                    base_fs = _clamp(base_fs, metrics.v_fs_med * 0.70, metrics.v_fs_med * 1.50)
                base_fs = max(1.0, base_fs)

                # --- 文字ごとの配置ロジック ---
                if style == "rotate":
                    # 括弧・長音など（90度回転して配置）
                    # 回転時は基準点がずれるため補正が必要
                    rot = 270 if (225.0 <= angle <= 315.0) else 90
                    # 回転文字は少し小さくするとバランスが良い
                    fs_rot = base_fs * 0.95

                    # 回転の中心合わせ（経験則補正）
                    px = center_x - (fs_rot * 0.35)
                    py = cell_cy + (fs_rot * 0.15)  # わずかに下へ

                    if stats is not None:
                        stats.vertical_exception_chars += 1
                        stats.vertical_chars += 1
                    _try_insert(fitz.Point(px, py), ch, fs_rot, rotate=int(rot), xscale_=1.0)

                elif style == "punct":
                    # 句読点（、。）: 右上寄せ
                    fs_punct = base_fs * 0.6

                    # 句読点の配置基準（セル右上寄せ）
                    cell_top = y0 + (cell_h * i)
                    try:
                        w = float(inv_font.text_length(ch, fontsize=fs_punct)) if inv_font is not None else float(fs_punct)
                    except Exception:
                        w = float(fs_punct)

                    px = x1 - w - (fs_punct * 0.15)
                    bs = (asc_desc_half * fs_punct) if asc_desc_half is not None else (fs_punct * 0.38)
                    py = cell_top + (fs_punct * 0.85) + bs

                    # 「、」はやや左下へ
                    if ch == "、":
                        px -= fs_punct * 0.20
                        py += fs_punct * 0.20

                    if stats is not None:
                        stats.vertical_exception_chars += 1
                        stats.vertical_chars += 1
                    _try_insert(fitz.Point(px, py), ch, fs_punct, rotate=0, xscale_=1.0)

                else:
                    # 通常文字（正立）
                    # 文字幅とフォントメトリクスで中心→ベースラインを逆算（固定係数よりズレにくい）
                    try:
                        w = float(inv_font.text_length(ch, fontsize=base_fs)) if inv_font is not None else float(base_fs)
                    except Exception:
                        w = float(base_fs)
                    px = center_x - (w * 0.5)
                    bs = (asc_desc_half * base_fs) if asc_desc_half is not None else (base_fs * 0.38)
                    py = cell_cy + bs

                    if stats is not None:
                        stats.vertical_chars += 1
                    _try_insert(fitz.Point(px, py), ch, base_fs, rotate=0, xscale_=1.0)

        else:
            # 横書き処理（既存のロジックを使用）
            if stats is not None:
                stats.horizontal_tokens += 1

            fs = max(2.5, rect_h * 0.85)
            bx = x0
            by = y1 - (rect_h * 0.15)

            if opt.force_rotate0:
                rot = 0
            else:
                rot = self._snap_rotate(angle)

            xs = 1.0 if opt.force_xscale_1 else float(xscale)
            _try_insert(fitz.Point(bx, by), txt, fs, rotate=int(rot), xscale_=xs)
    def _compose_page(
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
        """ページ合成（分割版）: 実装は pdf_compose.compose_page に委譲。"""
        return pdf_compose.compose_page(
            self,
            out_doc,
            bg_jpeg_bytes,
            w_px,
            h_px,
            tokens,
            src_page_rect,
            page_index=page_index,
            ocr_w_px=ocr_w_px,
            ocr_h_px=ocr_h_px,
        )

    def _maybe_store_shrink(self):
        if self.store_shrink <= 0:
            return
        try:
            fitz.TOOLS.store_shrink(self.store_shrink)
        except Exception as e:
            _log_exception_once('L2646', e)

    # -------------------------
    # Main
    # -------------------------
    def process_pdf(self, input_pdf: str, output_dir: str) -> str:
        """Enhance a PDF and write the output PDF into output_dir.

        The implementation is delegated to process_runner.process_pdf to reduce the size of this module.
        """
        import process_runner
        return process_runner.process_pdf(self, input_pdf, output_dir)

    def test_page(self, input_pdf: str, page_no_1based: int, run_ocr: bool = True) -> Dict[str, Any]:
        """
        任意ページを「設定調整用」に単独処理します（PDFは出力しません）。
        - SR →（任意）Deskew → 背景用画像
        - OCR前処理（二値化等）
        - （任意）OCR実行してBBoxを抽出
        戻り値にはプレビュー用の画像（RGB）と統計情報を返します。

        ※Deskewは分岐前に適用することで、背景とOCR座標がズレない構造を維持します。
        """
        if not os.path.isfile(input_pdf):
            raise FileNotFoundError(input_pdf)

        # 1-based -> 0-based
        with pdf_io.open_document(input_pdf) as doc:
            total = doc.page_count
            if total <= 0:
                raise RuntimeError("PDFのページ数が0です。")
            p = int(page_no_1based) - 1
            p = max(0, min(total - 1, p))
            page = doc.load_page(p)
            src_rect = page.rect
            rgb = self._render_page_to_rgb(page)

        # Unified page pipeline (preview mode)

        pipe = run_page_pipeline(
            self,
            rgb=rgb,
            src_rect=src_rect,
            page_index=p,
            mode="preview",
            preview_jpeg_roundtrip=True,
            preview_apply_boldness=False,
        )

        ang = float(pipe.deskew_angle)
        # Preview background image: JPEG round-trip (best-effort) to match final appearance.
        bg_preview = pipe.bg_preview_rgb if pipe.bg_preview_rgb is not None else pipe.sr_rgb
        if bg_preview is None:
            bg_preview = rgb

        # OCR base (deskewed SR image) and OCR preproc
        sr_rgb2 = pipe.sr_rgb_for_ocr if pipe.sr_rgb_for_ocr is not None else bg_preview
        ocr_bgr = pipe.ocr_bgr if pipe.ocr_bgr is not None else self._preprocess_for_ocr(sr_rgb2)

        tokens: List[OcrToken] = []
        if run_ocr:
            tokens = self._run_ocr_singleproc(ocr_bgr)

        # Overlay preview (BBox)
        overlay = bg_preview.copy()
        if tokens:
            # BBoxを描画（視覚確認用）
            # 点群の四角形はYomiToku側のpointsだが、ここでは簡易に外接矩形で表示
            for t in tokens:
                x0, y0, x1, y1 = int(t.rect.x0), int(t.rect.y0), int(t.rect.x1), int(t.rect.y1)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

        ocr_rgb = cv2.cvtColor(ocr_bgr, cv2.COLOR_BGR2RGB)

        # hygiene
        del rgb, ocr_bgr
        self._maybe_store_shrink()
        gc.collect()
        if torch is not None and self.device_str == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                _log_exception_once('L3222', e)

        return {
            "page_index": p,
            "page_total": total,
            "src_rect": src_rect,
            "sr_rgb": bg_preview,
            "sr_rgb_for_ocr": sr_rgb2,
            "ocr_rgb": ocr_rgb,
            "overlay_rgb": overlay,
            "deskew_angle": float(ang),
            "token_count": int(len(tokens)),
            "device": self.device_str,
        }


    # -------------------------
    # Resource cleanup (GPU/CPU)
    # -------------------------
    def shutdown(self, aggressive: bool = True) -> None:
        """Best-effort cleanup for heavy resources.

        Notes:
        - On CUDA, most VRAM is only fully returned when the *process exits*.
          Still, deleting models + empty_cache can reduce VRAM significantly.
        - This method is safe to call multiple times.
        """
        # 1) Drop references to heavy models/objects
        try:
            if hasattr(self, "upsampler"):
                try:
                    self.upsampler = None  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(self, "ocr"):
                try:
                    self.ocr = None  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Run GC and free GPU caches
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

        try:
            if torch is not None and torch.cuda.is_available():
                # Free cached blocks
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # Collect interprocess cached memory (if available)
                try:
                    if aggressive and hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()  # type: ignore
                except Exception:
                    pass
                # Ensure all queued CUDA work is completed
                try:
                    if aggressive and hasattr(torch.cuda, "synchronize"):
                        torch.cuda.synchronize()  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass


# =========================
# GUI
# =========================
