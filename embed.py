# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict

import fitz

# OCR data structures
# =========================
@dataclass
class OcrToken:
    text: str
    points: List[Tuple[float, float]]  # 4 points (x,y)
    rect: fitz.Rect                    # axis-aligned bounding box
    angle: float                       # top-edge angle in degrees


@dataclass
class OcrLine:
    text: str
    rect: fitz.Rect
    rotate: int
    fontsize: float
    baseline: fitz.Point
    tokens: List[OcrToken]



# =========================
# Embedding quality stats / fallback
# =========================
@dataclass
class EmbedOptions:
    # Page reading direction override: ''(auto) | 'vertical' | 'horizontal'
    page_direction: str = ""
    # Vertical (tategaki) chunking
    vertical_chunking: bool = False
    vertical_chunk_max: int = 6  # max chars per chunk (non-exception run)
    # Horizontal adjustments
    force_xscale_1: bool = False
    # 縦書きで1文字ごと改行になりやすい問題を避けるため、1トークンを1回で埋め込む（回転＋xscaleで高さ合わせ）
    vertical_run_mode: bool = True
    # 縦書きrun-modeの「長さ（選択範囲）」がやや短く見える場合の補正（1.00=補正なし）
    vertical_run_len_boost: float = 1.10
    force_rotate0: bool = False  # if True, always rotate=0 for horizontal
    # Debug
    tag: str = "primary"


@dataclass
class EmbedPageStats:
    page_index: int
    direction: str = "horizontal"
    total_tokens: int = 0
    skipped_empty: int = 0
    token_scale_x: float = 1.0
    token_scale_y: float = 1.0

    # --- OCR diagnostics (regression prevention for invisible text) ---
    # NOTE: These fields are optional and are filled by process_flow when available.
    # They help detect pages where OCR unexpectedly produced no tokens, causing
    # invisible text (selection/search) to be missing.
    ocr_words_data: int = 0            # number of raw OCR word entries (mp worker output)
    ocr_err: str = ""                 # worker error string (if any)
    ocr_fallback_used: bool = False    # True if we fell back to single-process OCR
    ocr_ink_ratio: float = 0.0         # rough percent of dark pixels (0..100), to distinguish blank pages
    ocr_empty_suspected_nonblank: bool = False  # tokens=0 but ink_ratio suggests content exists
    ocr_empty_reason: str = ""        # short reason tag for tokens=0 cases


    attempted_inserts: int = 0
    ok_inserts: int = 0
    insert_errors: int = 0

    vertical_tokens: int = 0
    horizontal_tokens: int = 0

    vertical_chars: int = 0
    vertical_exception_chars: int = 0
    vertical_chunk_inserts: int = 0

    # vertical run-mode fit diagnostics (for tategaki selection-length tuning)
    vrun_samples: int = 0
    vrun_rect_h_sum: float = 0.0
    vrun_target_h_sum: float = 0.0
    vrun_target_ratio_sum: float = 0.0
    vrun_xscale_sum: float = 0.0
    vrun_target_ratio_min: float = 9999.0
    vrun_target_ratio_max: float = 0.0

    body_rect_used: bool = False
    fallback_stage: int = 0
    fallback_reason: str = ""

    def error_rate(self) -> float:
        if self.attempted_inserts <= 0:
            return 0.0
        return float(self.insert_errors) / float(self.attempted_inserts)

    def ok_rate(self) -> float:
        if self.attempted_inserts <= 0:
            return 0.0
        return float(self.ok_inserts) / float(self.attempted_inserts)

    def vrun_target_ratio_mean(self) -> float:
        if self.vrun_samples <= 0:
            return 0.0
        return float(self.vrun_target_ratio_sum) / float(self.vrun_samples)

    def vrun_xscale_mean(self) -> float:
        if self.vrun_samples <= 0:
            return 0.0
        return float(self.vrun_xscale_sum) / float(self.vrun_samples)


@dataclass
class EmbedPageMetrics:
    """ページ内の埋め込み用メトリクス（暴走防止・安定化用）。"""
    v_pitch_med: float = 0.0  # vertical 1-char cell height median (pt)
    v_colw_med: float = 0.0   # vertical column width median (pt)
    v_fs_med: float = 0.0     # suggested vertical fontsize median (pt)
    h_h_med: float = 0.0      # horizontal token height median (pt)
    v_samples: int = 0
    h_samples: int = 0



# =========================
