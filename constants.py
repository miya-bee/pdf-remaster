# -*- coding: utf-8 -*-
from __future__ import annotations

"""Shared constants for PDF Remaster_v1_0_0.

This module is intentionally side-effect free (no heavy imports, no runtime logic).
It is used by both GUI and Engine to avoid circular imports and fragile namespace hacks.
"""

# =========================================
# App identity
# =========================================
APP_NAME = "PDF Remaster"
APP_VERSION = "v1_0_0"
APP_FULLNAME = f"{APP_NAME}_{APP_VERSION}"

CONFIG_FILENAME = "pdf_remaster_v1_0_0__settings__config.json"

# Debug / dev flags
# When enabled, some swallowed exceptions (except Exception: pass) will be logged once.
DEBUG_LOG_EXCEPTIONS = False

# Defaults
DEFAULT_DPI = 100
DEFAULT_JPEG_QUALITY = 85
DEFAULT_AUTO_GRAYSCALE = True
DEFAULT_GRAY_COLOR_RATIO_PERCENT = 0.3  # %: "色のある画素"がこれ未満ならグレー化（小さいほどカラー保持）
DEFAULT_GRAY_CHROMA_THRESHOLD = 10.0    # Lab(a,b) の色度しきい値（大きいほどグレー判定が増える）
DEFAULT_GRAY_CHROMA_P99 = 18.0          # Lab色度のp99がこれ以上ならカラー保持
DEFAULT_GRAY_JPEG_QUALITY_OFFSET = -10   # GrayAuto時にJPEG品質を下げるオフセット（例:-10で容量↓ / 0で画質優先）

DEFAULT_BG_SCALE_PERCENT = 50  # 出力背景の縮小率(%) 50で容量大幅減
DEFAULT_MAX_OUTPUT_DPI = 400  # 出力実効DPI上限（originalモード時、0=無効）
DEFAULT_BINARIZE_STRENGTH = 0  # default: show original (no binarize)
DEFAULT_TEXT_BOLDNESS = 0  # -100..+100: 閲覧用の文字太さ（-細く / +太く）
DEFAULT_ESRGAN_TILE = 256
DEFAULT_OUTPUT_PAGE_MODE = "original"  # "original" or "pixel"
DEFAULT_FONT_PATH = "AUTO"
DEFAULT_READING_MODE = "vertical"  # "vertical" | "horizontal"

DEFAULT_MODEL_FILENAME = "RealESRGAN_x4plus_anime_6B.pth"

# New professional options
DEFAULT_OCR_WORKERS = 1          # 0 disables multiprocessing pipeline
DEFAULT_ENABLE_DESKEW = False
DEFAULT_DESKEW_MAX_DEG = 5.0     # cap correction range
DEFAULT_STORE_SHRINK = 100       # aggressive to avoid leaks

# Guide URLs
REAL_ESRGAN_RELEASES_URL = "https://github.com/xinntao/Real-ESRGAN/releases"
REAL_ESRGAN_ANIME_DOC_URL = "https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_model.md"

