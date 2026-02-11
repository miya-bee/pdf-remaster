# -*- coding: utf-8 -*-
from __future__ import annotations

"""GUI for PDF Remaster_v1_0_0 (split edition).

This module intentionally avoids importing the entire engine namespace into globals.
Instead we import only what we need (constants + Engine public API), which improves:
- Safety (fewer unexpected name collisions)
- Maintainability (explicit dependencies)
- Multiprocessing stability on Windows (spawn)
"""

import os
import sys
import json
import re
import time
import threading
import queue
import traceback
import glob
import math
import platform
import multiprocessing as _mp
mp = _mp  # alias for freeze_support() etc.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Optional deps (GUI can start even if these are missing; processing may require them)
try:
    from PIL import Image, ImageTk  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore


# -------------------------
# Preview binarization helper
# (moved to zoom_viewer.py; imported below as _binarize_preview_rgb)
# -------------------------

try:
    import fitz  # type: ignore
    import pdf_io  # type: ignore  # PDF I/O wrappers
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False
    fitz = None  # type: ignore

# Optional drag & drop (tkinterdnd2)
DND_AVAILABLE = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False
    DND_FILES = None  # type: ignore
    TkinterDnD = None  # type: ignore

from typing import Optional, Tuple, List, Dict

from constants import (
    APP_FULLNAME,
    CONFIG_FILENAME,
    DEFAULT_DPI,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_AUTO_GRAYSCALE,
    DEFAULT_GRAY_COLOR_RATIO_PERCENT,
    DEFAULT_GRAY_JPEG_QUALITY_OFFSET,
    DEFAULT_BG_SCALE_PERCENT,
    DEFAULT_MAX_OUTPUT_DPI,
    DEFAULT_BINARIZE_STRENGTH,
    DEFAULT_ESRGAN_TILE,
    DEFAULT_OUTPUT_PAGE_MODE,
    DEFAULT_READING_MODE,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_FONT_PATH,
    DEFAULT_TEXT_BOLDNESS,
    DEFAULT_STORE_SHRINK,
    DEFAULT_ENABLE_DESKEW,
    DEFAULT_DESKEW_MAX_DEG,
    DEFAULT_OCR_WORKERS,
    REAL_ESRGAN_RELEASES_URL,
    REAL_ESRGAN_ANIME_DOC_URL,
)

import settings_io as _settings_io

from engine import PdfOcrEnhanceEngine
from log_utils import _log_exception_once, get_log_path, setup_app_logging

from zoom_viewer import ZoomImageViewer, binarize_preview_rgb as _binarize_preview_rgb
from ui_dispatch import UiDispatcher, set_default_dispatcher, safe_after, safe_cancel, is_ui_thread, configure_ui_dispatch
from preview_window import show_preview_window as _show_preview_window_impl
# ZoomImageViewer has been moved to zoom_viewer.py for maintainability.

class AppGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
        self.root.title(f"{APP_FULLNAME}  (Real-ESRGAN + YomiToku)")
        self.root.geometry("900x800")

        # --- UI Dispatch (single source of truth) ---
        # Tkinter is not thread-safe. Any UI updates from worker threads must be routed.
        try:
            self.ui = UiDispatcher(self.root, interval_ms=30)
            set_default_dispatcher(self.ui)
            # Enforce strict UI dispatch (do not allow unsafe Tk calls from worker threads)
            configure_ui_dispatch(strict=True, raise_on_violation=False)
        except Exception as e:
            self.ui = None
            _log_exception_once('ui_dispatch_init', e)

        self.msg_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self._closing = False
        self._poll_job = None  # root.after job id for _poll_queue


        # --- Preview test engine cache (reuse heavy models for re-OCR in preview) ---
        self._test_engine = None
        self._test_engine_lock = threading.Lock()

        self._build_ui()

        # ensure weights dir exists (and show in log)
        self._ensure_weights_dir()

        # load settings (portable preferred if exists)
        self._detect_and_load_config()

        # --- SAFETY: 起動時は「文字太さ（閲覧）」を必ず 0 に戻す ---
        # 以前の実行で太字化=100などにしていた設定が残っていると、
        # ユーザーが 0 を指定したつもりでも内部に残値が適用されるケースがあったため、
        # 起動直後に明示的に 0 を書き込みます（透明文字やOCRには影響しません：背景画像のみ）。
        try:
            if hasattr(self, "var_bold"):
                self.var_bold.set(0)
            if hasattr(self, "sc_bold"):
                self.sc_bold.set(0)
            self._log("[INFO] 起動時に文字太さ（閲覧）を0へリセットしました。")
        except Exception as e:
            _log_exception_once('L3760', e)

        # refresh model candidates (after config load) (after config load)
        self._refresh_model_choices(keep_current=True)

        # 入力PDFパスが手入力で変わった場合もページ数を更新（デバウンス）
        self._in_trace_job = None
        try:
            self.var_in.trace_add("write", self._on_input_path_changed)
        except Exception as e:
            _log_exception_once('L3735', e)

        # 初期状態のページ数更新
        self._update_pdf_page_range()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queue()

    # -------------------------
    # Paths
    # -------------------------
    def _app_base_dir(self) -> str:
        try:
            return _settings_io.get_app_base_dir()
        except Exception as e:
            _log_exception_once('L3751', e)
            return os.path.dirname(os.path.abspath(__file__))

    def _portable_config_path(self) -> str:
        return _settings_io.portable_config_path(app_base_dir=self._app_base_dir())

    def _local_config_dir(self) -> str:
        return _settings_io.local_config_dir()

    def _local_config_path(self) -> str:
        return _settings_io.local_config_path()

    def _config_path(self) -> str:
        portable = bool(getattr(self, "var_portable", tk.BooleanVar(value=False)).get())
        return _settings_io.config_path(portable_mode=portable, app_base_dir=self._app_base_dir())

    def _update_config_path_label(self):
        self.var_cfgpath.set(self._config_path())

    # -------------------------
    # weights folder / model defaults
    # -------------------------
    def _recommended_weights_dir(self) -> str:
        base = os.path.join(self._app_base_dir(), "weights")
        try:
            os.makedirs(base, exist_ok=True)
            test = os.path.join(base, ".write_test")
            with open(test, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test)
            return base
        except Exception as e:
            _log_exception_once('L3787', e)

        local = os.path.join(self._local_config_dir(), "weights")
        try:
            os.makedirs(local, exist_ok=True)
        except Exception as e:
            _log_exception_once('L3793', e)
        return local

    def _ensure_weights_dir(self) -> str:
        d = self._recommended_weights_dir()
        try:
            os.makedirs(d, exist_ok=True)
            # fonts サブフォルダ（AUTO検出用）
            fdir = os.path.join(d, "fonts")
            os.makedirs(fdir, exist_ok=True)
            self._log(f"[INFO] weightsフォルダ: {d}")
            self._log(f"[INFO] fontsフォルダ: {fdir}")
        except Exception as e:
            self._log(f"[WARN] weights/fonts フォルダ作成に失敗: {d} / {e}")
        return d

    def _default_model_path(self) -> str:
        wdir = self._ensure_weights_dir()
        return os.path.join(wdir, DEFAULT_MODEL_FILENAME)

    def _open_folder(self, path: str):
        try:
            if os.path.isdir(path):
                os.startfile(path)
            else:
                os.startfile(os.path.dirname(path))
        except Exception as e:
            self._log(f"[WARN] フォルダを開けません: {path} / {e}")

    def _show_model_download_guide(self, current_path: str):
        """
        Real-ESRGAN重み(.pth)が見つからない時の案内（パス正規化・候補表示を強化）
        - 「推奨の既定パス」が固定で誤解を招きやすかったため、
          アプリ既定weights と ユーザー指定パス側の両方を表示する。
        """
        fname = "RealESRGAN_x4plus_anime_6B.pth"

        raw = current_path if current_path is not None else ""
        # 可能ならGUI側の正規化関数を利用（v6以降）
        try:
            norm = self._normalize_path_str(raw)
        except Exception:
            norm = str(raw).strip().strip('"').strip("'").strip()
            try:
                norm = os.path.normpath(norm)
            except Exception as e:
                _log_exception_once('L3839', e)

        # アプリ側の既定 weights フォルダ
        try:
            wdir = self._ensure_weights_dir()
        except Exception:
            wdir = ""
        rec_app = os.path.join(wdir, fname) if wdir else ""

        # ユーザー指定パスのディレクトリ側にも置けるように案内
        rec_user = ""
        try:
            if norm:
                rec_user = os.path.join(os.path.dirname(norm), fname)
        except Exception:
            rec_user = ""

        def _isfile(p):
            try:
                return bool(p) and os.path.isfile(p)
            except Exception:
                return False


        exists = _isfile(raw) or (norm and _isfile(norm))

        msg = []
        if exists:
            msg.append("【正常】モデル重みファイルは正しく認識されています。")
        else:
            msg.append("【エラー】Real-ESRGAN のモデル重みファイル（.pth）が見つかりません。")

        msg.append("")
        msg.append("現在の指定パス:")
        msg.append(f"  {raw}")
        if norm and norm != raw:
            msg.append("")
            msg.append("正規化後パス（参考）:")
            msg.append(f"  {norm}")
        msg.append("")
        msg.append("存在チェック:")
        msg.append(f"  isfile(指定) = {_isfile(raw)}")
        if norm:
            msg.append(f"  isfile(正規化) = {_isfile(norm)}")

        # ファイルが無い場合のみ、入手手順を表示
        if not exists:
            msg.append("")
            msg.append("手順:")
            msg.append("  0) アプリの「自動DL」ボタンを押す（推奨）")
            msg.append(f"  1) 公式ページ（Releases）から '{fname}' をダウンロード")
            msg.append("  2) どちらかのフォルダに保存（どちらでもOK）:")
            if rec_user:
                try:
                    msg.append(f"     - あなたの指定フォルダ: {os.path.dirname(norm)}")
                except Exception as e:
                    _log_exception_once('L3894', e)
            if wdir:
                msg.append(f"     - アプリ既定weights: {wdir}")
            msg.append("  3) 画面の「ESRGAN重み(.pth)」でそのファイルを指定して実行")
            msg.append("")
            msg.append("推奨の既定パス（候補）:")
            if rec_user:
                msg.append(f"  - {rec_user}")
            if rec_app and (rec_app != rec_user):
                msg.append(f"  - {rec_app}")


        # messagebox
        try:
            import tkinter.messagebox as messagebox
            if exists:
                messagebox.showinfo("確認", "\n".join(msg))
            else:
                messagebox.showerror("エラー", "\n".join(msg))
        except Exception:
            # 最終フォールバック
            try:
                print("\n".join(msg))
            except Exception as e:
                _log_exception_once('L3918', e)

    def _validate_model_path_or_guide(self) -> bool:
        """ESRGAN重みパスの存在確認（不可視文字混入やスラッシュ混在でも解決できるよう強化）"""
        # SR OFF のときはモデル不要
        try:
            if hasattr(self, "var_safe_mode") and bool(self.var_safe_mode.get()):
                return True
        except Exception:
            pass
        try:
            if hasattr(self, "var_enable_sr") and (not bool(self.var_enable_sr.get())):
                return True
        except Exception:
            pass
        raw = self.var_model.get()
        path = self._normalize_path_str(raw)
        if not path:
            self._show_model_download_guide("(未指定)")
            return False

        wdir = self._ensure_weights_dir()
        resolved = self._resolve_existing_file(path, fallback_dir=wdir, fallback_name="RealESRGAN_x4plus_anime_6B.pth")

        if not resolved:
            # 不可視文字混入の切り分け用
            try:
                self._log(f"[DEBUG] model_path raw={raw!r}")
                self._log(f"[DEBUG] model_path norm={path!r}")
                self._log(f"[DEBUG] isfile(raw)={os.path.isfile(raw)} / isfile(norm)={os.path.isfile(path)}")
            except Exception as e:
                _log_exception_once('L3938', e)
            self._show_model_download_guide(path)
            return False

        if resolved != raw:
            try:
                self.var_model.set(resolved)
                self._save_config()
                self._log(f"[INFO] ESRGAN重みパスを解決: {resolved}")
            except Exception as e:
                _log_exception_once('L3948', e)
        return True

    

    def _update_model_status_label(self) -> None:
        """Update the SR model status label (best-effort)."""
        try:
            if not hasattr(self, "lbl_model_status"):
                return
            try:
                sr_on = bool(self.var_enable_sr.get()) if hasattr(self, "var_enable_sr") else True
            except Exception:
                sr_on = True
            try:
                if hasattr(self, "var_safe_mode") and bool(self.var_safe_mode.get()):
                    sr_on = False
            except Exception:
                pass
            if not sr_on:
                self.lbl_model_status.configure(text="SR: OFF（モデル不要）")
                return
            mp = (self.var_model.get().strip() if hasattr(self, "var_model") else "").strip()
            if not mp:
                mp = self._default_model_path()
            exists = os.path.isfile(mp)
            if exists:
                msg = f"SRモデル: OK  ({os.path.basename(mp)})"
            else:
                msg = "SRモデル: 未検出  （自動DL / 案内 から取得してください）"
            self.lbl_model_status.configure(text=msg)
        except Exception:
            pass

    def _download_default_esrgan_weights(self) -> None:
        """Download the default ESRGAN model into weights/ and select it."""
        self._assert_ui("_download_default_esrgan_weights")

        wdir = self._ensure_weights_dir()
        dst = os.path.join(wdir, DEFAULT_MODEL_FILENAME)

        # If already present, just select.
        if os.path.isfile(dst):
            self.var_model.set(dst)
            self._refresh_model_choices(keep_current=True)
            self._update_model_status_label()
            self._save_config()
            messagebox.showinfo("モデル重み", "既に重みファイルが存在します。選択を更新しました。")
            return

        win = tk.Toplevel(self.root)
        win.title("SRモデル重みのダウンロード")
        win.geometry("520x150")
        try:
            win.transient(self.root)
            win.grab_set()
        except Exception:
            pass

        ttk.Label(win, text=f"ダウンロード先: {dst}").pack(anchor="w", padx=12, pady=(12, 4))

        pbar = ttk.Progressbar(win, mode="determinate", maximum=100)
        pbar.pack(fill="x", padx=12, pady=6)

        lbl2 = ttk.Label(win, text="準備中...")
        lbl2.pack(anchor="w", padx=12, pady=(0, 6))

        cancel = {"flag": False}

        def on_cancel():
            cancel["flag"] = True
            try:
                lbl2.configure(text="キャンセル中...")
            except Exception:
                pass

        ttk.Button(win, text="キャンセル", command=on_cancel).pack(anchor="e", padx=12, pady=(0, 10))

        def worker():
            def cancelled() -> bool:
                return bool(cancel["flag"])

            def prog(downloaded: int, total: int, status: str) -> None:
                def _u():
                    if not win.winfo_exists():
                        return
                    if total > 0 and downloaded >= 0:
                        pct = int(min(100, max(0, downloaded * 100 / total)))
                        pbar.configure(value=pct)
                        lbl2.configure(
                            text=f"{status}  {pct}%  ({downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB)"
                        )
                    else:
                        lbl2.configure(text=status)

                safe_after(win, 0, _u, where="esrgan_dl_progress")

            try:
                from weights_manager import ensure_default_esrgan
                ensure_default_esrgan(wdir, filename=DEFAULT_MODEL_FILENAME, progress_cb=prog, cancel_flag=cancelled)
            except Exception as e:
                def _err():
                    try:
                        if win.winfo_exists():
                            win.grab_release()
                            win.destroy()
                    except Exception:
                        pass
                    messagebox.showerror("ダウンロード失敗", f"SRモデルのダウンロードに失敗しました。\n\n{e}")

                safe_after(self.root, 0, _err, where="esrgan_dl_error")
                return

            def _done():
                try:
                    if win.winfo_exists():
                        win.grab_release()
                        win.destroy()
                except Exception:
                    pass
                self.var_model.set(dst)
                self._refresh_model_choices(keep_current=True)
                self._update_model_status_label()
                self._save_config()
                messagebox.showinfo("ダウンロード完了", "SRモデルをダウンロードしました。")

            safe_after(self.root, 0, _done, where="esrgan_dl_done")

        threading.Thread(target=worker, daemon=True).start()

    def _scan_weight_models(self) -> List[str]:
        wdir = self._ensure_weights_dir()
        patterns = [os.path.join(wdir, "*.pth"), os.path.join(wdir, "*.pt")]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        files = list(dict.fromkeys([os.path.abspath(p) for p in files]))
        files.sort(key=lambda p: os.path.basename(p).lower())
        return files

    def _make_unique_labels(self, paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
        label_to_path: Dict[str, str] = {}
        used = set()
        labels: List[str] = []

        for p in paths:
            base = os.path.basename(p)
            label = base
            if label in used:
                parent = os.path.basename(os.path.dirname(p))
                label = f"{base} — {parent}"
            n = 2
            while label in used:
                label = f"{base} — {n}"
                n += 1
            used.add(label)
            labels.append(label)
            label_to_path[label] = p

        return labels, label_to_path

    def _refresh_model_choices(self, keep_current: bool = True):
        if not hasattr(self, "cmb_model"):
            return

        current = (self.var_model.get().strip() if keep_current else "").strip()
        found = self._scan_weight_models()

        extra: List[str] = []
        if current and os.path.isfile(current) and (os.path.abspath(current) not in [os.path.abspath(x) for x in found]):
            extra = [os.path.abspath(current)]

        paths = extra + found
        labels, label_to_path = self._make_unique_labels(paths)
        self._model_label_to_path = label_to_path

        self.cmb_model["values"] = labels

        if current and os.path.isfile(current):
            chosen_label = None
            for lb, pp in label_to_path.items():
                if os.path.abspath(pp) == os.path.abspath(current):
                    chosen_label = lb
                    break
            if chosen_label:
                self.var_model_choice.set(chosen_label)
                self._update_model_status_label()
                return

        pref = None
        for lb, pp in label_to_path.items():
            if "anime" in os.path.basename(pp).lower():
                pref = lb
                break
        if pref:
            self.var_model_choice.set(pref)
            self.var_model.set(label_to_path[pref])
            self._update_model_status_label()
            return

        if labels:
            self.var_model_choice.set(labels[0])
            self.var_model.set(label_to_path[labels[0]])
        else:
            self.var_model_choice.set("")
            self.var_model.set(self._default_model_path())
        self._update_model_status_label()
    def _on_model_choice_selected(self, event=None):
        lb = self.var_model_choice.get().strip()
        if not lb:
            return
        mpth = getattr(self, "_model_label_to_path", {}).get(lb)
        if mpth:
            self.var_model.set(mpth)
            self._log(f"[INFO] モデル選択: {mpth}")
            self._save_config()

    # -------------------------
    # Build UI
    # -------------------------
    def _assert_ui(self, where: str) -> None:
        """Best-effort UI-thread assertion (logs once on violation)."""
        try:
            ui = getattr(self, 'ui', None)
            if ui is not None:
                ui.assert_ui_thread(where)
        except Exception:
            pass

    def _build_ui(self):
        pad = 8

        # --- Scrollable main container (keeps window size reasonable) ---
        # We render all existing widgets inside an inner frame placed on a Canvas.
        # This minimizes refactor: below code continues to use `frm` as before.
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=pad, pady=pad)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        frm = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=frm, anchor="nw")

        # Keep scrollregion updated
        def _on_frame_configure(_evt=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        # Make inner frame match the canvas width (so layout behaves like before)
        def _on_canvas_configure(evt):
            try:
                canvas.itemconfigure(win_id, width=evt.width)
            except Exception:
                pass

        frm.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (Windows)
        def _on_mousewheel(evt):
            try:
                # evt.delta is typically 120/-120 per notch on Windows
                canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
            except Exception:
                pass

        def _bind_mousewheel(_evt=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_evt=None):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # expose for potential future use/debug
        self._main_canvas = canvas
        self._main_vscroll = vbar

        # Input PDF
        in_row = ttk.Frame(frm)
        in_row.pack(fill="x", pady=(0, 6))
        ttk.Label(in_row, text="入力PDF").pack(side="left")
        self.var_in = tk.StringVar()
        self.ent_in = ttk.Entry(in_row, textvariable=self.var_in)
        self.ent_in.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(in_row, text="参照...", command=self._pick_input).pack(side="left")
        ttk.Button(in_row, text="フォルダ", command=self._open_input_folder).pack(side="left", padx=(6, 0))

        if DND_AVAILABLE:
            self.ent_in.drop_target_register(DND_FILES)
            self.ent_in.dnd_bind("<<Drop>>", self._on_drop)

        # Output destination
        out_row = ttk.Frame(frm)
        out_row.pack(fill="x", pady=(0, 8))
        ttk.Label(out_row, text="出力先").pack(side="left")

        # Output mode (A: input neighbor default)
        self._out_mode_label_to_code = {
            "入力PDFと同じフォルダ（推奨）": "neighbor",
            "アプリ内（outputフォルダ）": "app_output",
            "指定フォルダ": "custom",
        }
        self._out_mode_code_to_label = {v: k for k, v in self._out_mode_label_to_code.items()}
        self.var_out_mode = tk.StringVar(value=self._out_mode_code_to_label.get("neighbor", "入力PDFと同じフォルダ（推奨）"))
        self.cmb_out_mode = ttk.Combobox(
            out_row,
            textvariable=self.var_out_mode,
            values=list(self._out_mode_label_to_code.keys()),
            state="readonly",
            width=18,
        )
        self.cmb_out_mode.pack(side="left", padx=(6, 6))
        self.cmb_out_mode.bind("<<ComboboxSelected>>", lambda e: self._on_output_mode_changed())

        # var_out: display/effective output dir (readonly in non-custom modes)
        # var_out_custom: user-selected custom output dir (persisted)
        self.var_out = tk.StringVar()
        self.var_out_custom = tk.StringVar()
        self.ent_out = ttk.Entry(out_row, textvariable=self.var_out, state="readonly")
        self.ent_out.pack(side="left", fill="x", expand=True, padx=6)
        self.btn_pick_out = ttk.Button(out_row, text="参照...", command=self._pick_output)
        self.btn_pick_out.pack(side="left")
        self.btn_open_out = ttk.Button(out_row, text="開く", command=self._open_output_folder)
        self.btn_open_out.pack(side="left", padx=6)

        # reflect initial output mode state
        try:
            self._update_output_dir_display()
        except Exception:
            pass

        # Settings
        cfg = ttk.LabelFrame(frm, text="処理設定")
        cfg.pack(fill="x", pady=(0, 10))

        # Model weights row
        self.var_model = tk.StringVar(value=self._default_model_path())
        self.var_model_choice = tk.StringVar(value="")
        self._model_label_to_path: Dict[str, str] = {}

        # SR model weights (two-row layout)
        row0 = ttk.Frame(cfg)
        row0.pack(fill="x", padx=8, pady=6)
        ttk.Label(row0, text="SRモデル", width=18).pack(side="left")

        self.cmb_model = ttk.Combobox(row0, textvariable=self.var_model_choice, values=[], state="readonly", width=38)
        self.cmb_model.pack(side="left", padx=6)
        self.cmb_model.bind("<<ComboboxSelected>>", self._on_model_choice_selected)

        def pick_model():
            p = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pth *.pt"), ("All", "*.*")])
            if p:
                self.var_model.set(p)
                self._refresh_model_choices(keep_current=True)
                if not os.path.isfile(p):
                    self._log(f"[WARN] 選択した重みが見つかりません: {p}")
                else:
                    self._save_config()

        self.btn_model_autodl = ttk.Button(row0, text="自動DL（推奨）", command=self._download_default_esrgan_weights)
        self.btn_model_autodl.pack(side="left", padx=6)
        self.btn_model_guide = ttk.Button(row0, text="案内", command=lambda: self._show_model_download_guide(self.var_model.get().strip() or "(未指定)"))
        self.btn_model_guide.pack(side="left", padx=6)

        row0b = ttk.Frame(cfg)
        row0b.pack(fill="x", padx=28, pady=(0, 6))
        self.ent_model = ttk.Entry(row0b, textvariable=self.var_model)
        self.ent_model.pack(side="left", fill="x", expand=True, padx=6)

        self.btn_pick_model = ttk.Button(row0b, text="参照...", command=pick_model)
        self.btn_pick_model.pack(side="left")
        self.btn_open_weights = ttk.Button(row0b, text="weightsフォルダ", command=lambda: self._open_folder(self._ensure_weights_dir()))
        self.btn_open_weights.pack(side="left", padx=6)
        self.btn_reload_model = ttk.Button(row0b, text="再読み込み", command=lambda: self._refresh_model_choices(keep_current=True))
        self.btn_reload_model.pack(side="left", padx=6)


        # Model status
        self.lbl_model_status = ttk.Label(cfg, text="")
        self.lbl_model_status.pack(anchor="w", padx=28, pady=(0, 6))
        self._update_model_status_label()

        # SR enable / Safe mode
        self.var_enable_sr = tk.BooleanVar(value=True)
        self.var_safe_mode = tk.BooleanVar(value=False)
        row0x = ttk.Frame(cfg)
        row0x.pack(fill="x", padx=28, pady=(0, 6))
        self.chk_enable_sr = ttk.Checkbutton(row0x, text="SRを使用（高精細化）", variable=self.var_enable_sr, command=self._on_sr_mode_changed)
        self.chk_enable_sr.pack(side="left")
        self.chk_safe_mode = ttk.Checkbutton(row0x, text="安全モード（安定優先）", variable=self.var_safe_mode, command=self._on_sr_mode_changed)
        self.chk_safe_mode.pack(side="left", padx=14)
        ttk.Label(row0x, text="  ※安全モード: SR OFF + OCR並列=0").pack(side="left", padx=8)

        # Cancel behavior
        self.var_keep_partial_on_cancel = tk.BooleanVar(value=True)
        row0y = ttk.Frame(cfg)
        row0y.pack(fill="x", padx=28, pady=(0, 6))
        self.chk_keep_partial_on_cancel = ttk.Checkbutton(
            row0y,
            text="中断時に途中結果PDFを出力",
            variable=self.var_keep_partial_on_cancel,
        )
        self.chk_keep_partial_on_cancel.pack(side="left")
        ttk.Label(row0y, text="  ※OFF: 中断時はファイルを作成しません").pack(side="left", padx=8)

        # DPI
        self.var_dpi = tk.IntVar(value=DEFAULT_DPI)
        row1 = ttk.Frame(cfg)
        row1.pack(fill="x", padx=8, pady=6)
        ttk.Label(row1, text="DPI（PDF→画像化）", width=18).pack(side="left")
        ttk.Spinbox(row1, from_=72, to=600, textvariable=self.var_dpi, width=8).pack(side="left")
        ttk.Label(row1, text="  高いほど精細だが重い").pack(side="left", padx=8)

        # JPEG quality
        self.var_jpeg = tk.IntVar(value=DEFAULT_JPEG_QUALITY)
        row2 = ttk.Frame(cfg)
        row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="JPEG品質（背景）", width=18).pack(side="left")
        self.sc_jpeg = ttk.Scale(row2, from_=40, to=100, orient="horizontal",
                                 command=lambda v: self.var_jpeg.set(int(float(v))))
        self.sc_jpeg.set(self.var_jpeg.get())
        self.sc_jpeg.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(row2, textvariable=self.var_jpeg, width=4).pack(side="left")

        # Auto grayscale (ほぼ白黒ページを自動でグレー化して容量削減)
        self.var_gray_auto = tk.IntVar(value=1 if DEFAULT_AUTO_GRAYSCALE else 0)
        self.var_gray_ratio = tk.DoubleVar(value=float(DEFAULT_GRAY_COLOR_RATIO_PERCENT))
        self.var_gray_q_offset = tk.IntVar(value=int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET))
        row2a = ttk.Frame(cfg)
        row2a.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(
            row2a,
            text="グレースケール自動（ほぼ白黒ページを軽量化）",
            variable=self.var_gray_auto,
            command=lambda: self._save_config(),
        ).pack(side="left")
        ttk.Label(row2a, text="  色率閾値(%)").pack(side="left", padx=(12, 0))
        sp_gray = ttk.Spinbox(row2a, from_=0.0, to=5.0, increment=0.1, textvariable=self.var_gray_ratio, width=6)
        sp_gray.pack(side="left", padx=6)
        sp_gray.bind("<FocusOut>", lambda e: self._save_config())
        sp_gray.bind("<Return>", lambda e: self._save_config())
        ttk.Label(row2a, text="  グレー品質オフセット").pack(side="left", padx=(10, 0))
        sp_qoff = ttk.Spinbox(row2a, from_=-40, to=0, increment=1, textvariable=self.var_gray_q_offset, width=5)
        sp_qoff.pack(side="left", padx=6)
        sp_qoff.bind("<FocusOut>", lambda e: self._save_config())
        sp_qoff.bind("<Return>", lambda e: self._save_config())
        ttk.Label(row2a, text="  推奨:-10（容量↓） / 0で画質優先  |  色率:0.3推奨 / カラー保持なら0.0%推奨").pack(side="left", padx=8)

        # Background scale (output size control)
        self.var_bg_scale = tk.IntVar(value=DEFAULT_BG_SCALE_PERCENT)
        row2b = ttk.Frame(cfg)
        row2b.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2b, text="背景縮小率（出力）", width=18).pack(side="left")
        ttk.Spinbox(row2b, from_=25, to=100, increment=5, textvariable=self.var_bg_scale, width=6).pack(side="left")
        ttk.Label(row2b, text="  %  (50で容量大幅減 / 100=高解像度のまま)").pack(side="left", padx=8)
        # Output DPI cap (original mode)
        self.var_max_out_dpi = tk.IntVar(value=DEFAULT_MAX_OUTPUT_DPI)
        row2c = ttk.Frame(cfg)
        row2c.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2c, text="出力DPI上限（original）", width=18).pack(side="left")
        ttk.Spinbox(row2c, from_=0, to=2400, increment=50, textvariable=self.var_max_out_dpi, width=6).pack(side="left")
        ttk.Label(row2c, text="  (0=無効 / 例: 400でファイルサイズを抑制)").pack(side="left", padx=8)

        # Binarize strength
        self.var_bin = tk.IntVar(value=DEFAULT_BINARIZE_STRENGTH)
        row3 = ttk.Frame(cfg)
        row3.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3, text="二値化強度（OCR）", width=18).pack(side="left")
        self.sc_bin = ttk.Scale(row3, from_=0, to=100, orient="horizontal",
                                command=lambda v: self.var_bin.set(int(float(v))))
        self.sc_bin.set(self.var_bin.get())
        self.sc_bin.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(row3, textvariable=self.var_bin, width=4).pack(side="left")
        ttk.Label(row3, text="  高いほど背景除去強め（文字欠け注意）").pack(side="left", padx=8)


        # Text boldness (view) - 閲覧用の文字太さ（背景画像のみ）
        # 0=無効。右(+)で太く / 左(-)で細く。極端値は薄化・欠け・黒つぶれ注意。
        self.var_bold = tk.IntVar(value=DEFAULT_TEXT_BOLDNESS)
        row3b = ttk.Frame(cfg)
        row3b.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3b, text="文字太さ（閲覧）", width=18).pack(side="left")
        self.sc_bold = ttk.Scale(row3b, from_=-100, to=100, orient="horizontal",
                         command=lambda v: self.var_bold.set(int(float(v))))
        self.sc_bold.set(self.var_bold.get())
        self.sc_bold.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(row3b, textvariable=self.var_bold, width=5).pack(side="left")
        ttk.Label(row3b, text="  0=無効 / 右で太く / 左で細く（極端は薄化・欠け注意）").pack(side="left", padx=8)


        # ESRGAN tile
        self.var_tile = tk.IntVar(value=DEFAULT_ESRGAN_TILE)
        row4 = ttk.Frame(cfg)
        row4.pack(fill="x", padx=8, pady=6)
        ttk.Label(row4, text="ESRGAN tile（GPU）", width=18).pack(side="left")
        self.sp_tile = ttk.Spinbox(row4, from_=0, to=2048, increment=64, textvariable=self.var_tile, width=8)
        self.sp_tile.pack(side="left")
        ttk.Label(row4, text="  0=無効 / VRAM不足なら 128〜512推奨").pack(side="left", padx=8)

        # Output page mode
        self.var_page_mode = tk.StringVar(value=DEFAULT_OUTPUT_PAGE_MODE)
        row5 = ttk.Frame(cfg)
        row5.pack(fill="x", padx=8, pady=6)
        ttk.Label(row5, text="出力ページサイズ", width=18).pack(side="left")
        ttk.Combobox(row5, textvariable=self.var_page_mode, values=["original", "pixel"], state="readonly", width=12).pack(side="left")
        ttk.Label(row5, text="  original=元PDF物理サイズ / pixel=1px=1pt").pack(side="left", padx=8)

        # Reading direction mode (auto removed: 必ず指定)
        self.var_reading_mode = tk.StringVar(value="")
        row5b = ttk.Frame(cfg)
        row5b.pack(fill="x", padx=8, pady=6)
        ttk.Label(row5b, text="読み方向", width=18).pack(side="left")
        self.cmb_reading = ttk.Combobox(
            row5b,
            textvariable=self.var_reading_mode,
            values=["vertical", "horizontal"],
            state="readonly",
            width=12,
        )
        self.cmb_reading.pack(side="left")
        ttk.Label(row5b, text="  vertical=縦書き(右→左) / horizontal=横書き(左→右) ※必ず選択").pack(side="left", padx=8)

        # Font file (AUTO / custom)
        self.var_font = tk.StringVar(value=DEFAULT_FONT_PATH)
        row6 = ttk.Frame(cfg)
        row6.pack(fill="x", padx=8, pady=6)
        ttk.Label(row6, text="日本語フォント", width=18).pack(side="left")
        ttk.Entry(row6, textvariable=self.var_font).pack(side="left", fill="x", expand=True, padx=6)

        def pick_font():
            p = filedialog.askopenfilename(filetypes=[("Font", "*.ttf *.otf *.ttc"), ("All", "*.*")])
            if p:
                self.var_font.set(p)
                self._save_config()

        def set_auto_font():
            self.var_font.set("AUTO")
            self._save_config()

        ttk.Button(row6, text="参照...", command=pick_font).pack(side="left")
        ttk.Button(row6, text="AUTO", command=set_auto_font).pack(side="left", padx=4)
        ttk.Label(row6, text="  AUTO=自動選択（推奨：IPAexゴシック）").pack(side="left", padx=8)

        # NEW: OCR workers

        self.var_ocr_workers = tk.IntVar(value=DEFAULT_OCR_WORKERS)
        row6b = ttk.Frame(cfg)
        row6b.pack(fill="x", padx=8, pady=6)
        ttk.Label(row6b, text="OCR並列（プロセス）", width=18).pack(side="left")
        self.sp_ocr_workers = ttk.Spinbox(row6b, from_=0, to=4, increment=1, textvariable=self.var_ocr_workers, width=8)
        self.sp_ocr_workers.pack(side="left")
        ttk.Label(row6b, text="  0=無効 / 推奨=1（SR中にOCRを並走）").pack(side="left", padx=8)

        # NEW: Deskew toggle + max deg
        self.var_deskew = tk.BooleanVar(value=DEFAULT_ENABLE_DESKEW)
        row6c = ttk.Frame(cfg)
        row6c.pack(fill="x", padx=8, pady=6)
        ttk.Label(row6c, text="傾き補正（Deskew）", width=18).pack(side="left")
        ttk.Checkbutton(row6c, text="有効", variable=self.var_deskew, command=self._save_config).pack(side="left")
        ttk.Label(row6c, text="  最大角度").pack(side="left", padx=(20, 4))
        self.var_deskew_max = tk.DoubleVar(value=DEFAULT_DESKEW_MAX_DEG)
        ttk.Spinbox(row6c, from_=0.0, to=15.0, increment=0.5, textvariable=self.var_deskew_max, width=8).pack(side="left")
        ttk.Label(row6c, text="deg（大きいほど補正するが誤検出リスク）").pack(side="left", padx=8)

        # Portable mode
        self.var_portable = tk.BooleanVar(value=False)
        row7 = ttk.Frame(cfg)
        row7.pack(fill="x", padx=8, pady=6)
        ttk.Checkbutton(row7, text="ポータブルモード（設定をexe/スクリプトと同じフォルダに保存）",
                        variable=self.var_portable, command=self._on_toggle_portable).pack(side="left")

        # Window startup behavior
        # - fixed: use fixed size (default 900px width)
        # - restore: restore last saved geometry (size + position)
        self.var_window_restore = tk.BooleanVar(value=False)
        self.var_window_fixed_w = tk.IntVar(value=900)
        self.var_window_fixed_h = tk.IntVar(value=800)

        row7b = ttk.Frame(cfg)
        row7b.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Checkbutton(
            row7b,
            text="起動時に前回のウインドウサイズ/位置を復元",
            variable=self.var_window_restore,
            command=self._on_window_prefs_changed
        ).pack(side="left")

        row7c = ttk.Frame(cfg)
        row7c.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(row7c, text="固定モード幅", width=18).pack(side="left")
        sp_w = ttk.Spinbox(row7c, from_=500, to=2400, increment=10, textvariable=self.var_window_fixed_w, width=8)
        sp_w.pack(side="left")
        ttk.Label(row7c, text="px（復元OFFのとき）").pack(side="left", padx=8)

        # Save when user edits width via keyboard (Enter) or focus-out
        try:
            sp_w.bind("<Return>", lambda e: self._on_window_prefs_changed())
            sp_w.bind("<FocusOut>", lambda e: self._on_window_prefs_changed())
        except Exception:
            pass


        # Config path display
        self.var_cfgpath = tk.StringVar(value="")
        row8 = ttk.Frame(cfg)
        row8.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(row8, text="設定ファイル", width=18).pack(side="left")
        ttk.Label(row8, textvariable=self.var_cfgpath).pack(side="left")
        self._update_config_path_label()


        # -------------------------
        # Test Preview (任意ページ)
        # -------------------------
        test = ttk.LabelFrame(frm, text="プレビュー")
        test.pack(fill="x", pady=(0, 10))

        self.var_total_pages = tk.IntVar(value=0)
        self.var_test_page = tk.IntVar(value=1)
        self.var_test_run_ocr = tk.BooleanVar(value=True)

        rowt = ttk.Frame(test)
        rowt.pack(fill="x", padx=8, pady=6)
        ttk.Label(rowt, text="ページ", width=6).pack(side="left")
        self.sp_test_page = ttk.Spinbox(rowt, from_=1, to=1, textvariable=self.var_test_page, width=8)
        self.sp_test_page.pack(side="left")
        ttk.Label(rowt, text=" / ").pack(side="left")
        ttk.Label(rowt, textvariable=self.var_total_pages, width=6).pack(side="left")

        ttk.Checkbutton(
            rowt, text="OCRも実行（BBox確認）", variable=self.var_test_run_ocr,
            command=self._save_config
        ).pack(side="left", padx=14)

        self.btn_test = ttk.Button(rowt, text="このページをテスト", command=self._run_test_page)
        self.btn_test.pack(side="left", padx=6)

        ttk.Label(
            test,
            text="※表紙ではなく「本文ページ」を指定して、DPI/二値化/Deskew等の効きを確認できます。",
            foreground="#555"
        ).pack(anchor="w", padx=10, pady=(0, 6))


        # Buttons
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=(0, 8))
        self.btn_run = ttk.Button(btn_row, text="実行", command=self._run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(btn_row, text="中断", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.btn_reset = ttk.Button(btn_row, text="リセット（初期値）", command=self._reset_defaults)
        self.btn_reset.pack(side="left", padx=8)
        self.btn_help = ttk.Button(btn_row, text="使い方", command=self._show_help)
        self.btn_help.pack(side="left", padx=8)
        self.btn_open_log = ttk.Button(btn_row, text="ログを開く", command=self._open_log_file)
        self.btn_open_log.pack(side="left", padx=8)
        self.btn_copy_diag = ttk.Button(btn_row, text="診断情報コピー", command=self._copy_diagnostic_info)
        self.btn_copy_diag.pack(side="left", padx=8)
        self.btn_env_diag = ttk.Button(btn_row, text="環境診断(ログ)", command=self._log_environment_diagnostics)
        self.btn_env_diag.pack(side="left", padx=8)

        # Progress
        prog_row = ttk.Frame(frm)
        prog_row.pack(fill="x", pady=(0, 6))
        self.var_prog = tk.StringVar(value="待機中")
        ttk.Label(prog_row, textvariable=self.var_prog).pack(side="left")
        self.pbar = ttk.Progressbar(prog_row, mode="determinate")
        self.pbar.pack(side="left", fill="x", expand=True, padx=10)

        # Log
        ttk.Label(frm, text="ログ").pack(anchor="w")
        self.txt_log = scrolledtext.ScrolledText(frm, height=22)
        self.txt_log.pack(fill="both", expand=True)

        if DND_AVAILABLE:
            self._log("[INFO] Drag & Drop: 有効（tkinterdnd2）")
        else:
            self._log("[INFO] Drag & Drop: 無効（tkinterdnd2未導入）")

        note = (
            "※必要: pymupdf, opencv-python, pillow, torch, realesrgan, basicsr, yomitoku\n"
            f"※モデル重み: {DEFAULT_MODEL_FILENAME}（GUIで指定）\n"
            "※大規模PDFはOCR並列=1 + store_shrink強化で安定性が上がります"
        )
        ttk.Label(frm, text=note, foreground="#555").pack(anchor="w", pady=(6, 0))

        # Reflect SR/Safe mode state (disable SR-related widgets if needed)
        try:
            self._on_sr_mode_changed(save=False)
        except Exception:
            pass

    # -------------------------
    # Pickers / DnD
    # -------------------------

    def _on_sr_mode_changed(self, save: bool = True):
        """SR / 安全モードの切替ハンドラ。
        - 安全モード: SR OFF + OCR並列=0 を強制
        - SR関連UIの有効/無効を切替
        """
        try:
            safe_mode = bool(self.var_safe_mode.get()) if hasattr(self, "var_safe_mode") else False
        except Exception:
            safe_mode = False
        try:
            enable_sr = bool(self.var_enable_sr.get()) if hasattr(self, "var_enable_sr") else True
        except Exception:
            enable_sr = True

        if safe_mode:
            # Force SR off + OCR workers = 0
            try:
                if hasattr(self, "var_enable_sr"):
                    self.var_enable_sr.set(False)
                enable_sr = False
            except Exception:
                enable_sr = False
            try:
                if hasattr(self, "var_ocr_workers"):
                    self.var_ocr_workers.set(0)
            except Exception:
                pass

        # Apply widget state
        try:
            self._apply_sr_safe_ui_state(enable_sr=enable_sr, safe_mode=safe_mode)
        except Exception as e:
            _log_exception_once('L_sr_mode_ui', e)

        # Update status label
        try:
            self._update_model_status_label()
        except Exception:
            pass

        if save:
            try:
                self._save_config()
            except Exception as e:
                _log_exception_once('L_sr_mode_save', e)

    def _apply_sr_safe_ui_state(self, enable_sr: bool, safe_mode: bool):
        """SR関連UIと安全モード制約の適用（best-effort）。"""
        sr_active = bool(enable_sr) and (not bool(safe_mode))

        # SR-related widgets
        sr_widget_names = [
            "cmb_model",
            "btn_pick_model",
            "btn_reload_model",
            "btn_open_weights",
            "btn_model_autodl",
            "btn_model_guide",
            "sp_tile",
        ]
        for nm in sr_widget_names:
            w = getattr(self, nm, None)
            if w is None:
                continue
            try:
                if nm == "cmb_model":
                    w.configure(state=("readonly" if sr_active else "disabled"))
                else:
                    w.configure(state=("normal" if sr_active else "disabled"))
            except Exception:
                pass

        # SR checkbox disabled while safe mode is ON (to reduce confusion)
        try:
            if hasattr(self, "chk_enable_sr") and self.chk_enable_sr is not None:
                self.chk_enable_sr.configure(state=("disabled" if safe_mode else "normal"))
        except Exception:
            pass

        # OCR workers disabled in safe mode
        ocr_w = getattr(self, "sp_ocr_workers", None)
        if ocr_w is not None:
            try:
                ocr_w.configure(state=("disabled" if safe_mode else "normal"))
            except Exception:
                pass

    def _pick_input(self):
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            self._set_input_path(path)

    def _pick_output(self):
        # Picking a folder implies "custom" output mode.
        try:
            self._set_output_mode_code("custom")
        except Exception:
            pass

        initial = (self.var_out_custom.get() or "").strip()
        kwargs = {}
        if initial and os.path.isdir(initial):
            kwargs["initialdir"] = initial
        path = filedialog.askdirectory(**kwargs)
        if path:
            self.var_out_custom.set(path)
            self._update_output_dir_display()
            self._save_config()

    # -------------------------
    # Output destination helpers
    # -------------------------
    def _get_output_mode_code(self) -> str:
        """Return output mode code: neighbor / app_output / custom."""
        try:
            label = (self.var_out_mode.get() or "").strip()
            return str(self._out_mode_label_to_code.get(label, "neighbor"))
        except Exception:
            return "neighbor"

    def _set_output_mode_code(self, code: str) -> None:
        try:
            code = str(code or "neighbor")
            label = self._out_mode_code_to_label.get(code, self._out_mode_code_to_label.get("neighbor", "入力PDFと同じフォルダ（推奨）"))
            self.var_out_mode.set(label)
        except Exception:
            pass
        self._update_output_dir_display()

    def _on_output_mode_changed(self) -> None:
        self._update_output_dir_display()
        try:
            self._save_config()
        except Exception:
            pass

    def _recommended_output_dir(self, check_writable: bool = True) -> str:
        """Preferred 'output' dir.

        - In portable/zip usage: <app>/output
        - Fallback: LOCALAPPDATA/.../output

        check_writable=False is useful for UI display without touching disk.
        """
        base = os.path.join(self._app_base_dir(), "output")
        if not check_writable:
            return base
        if _settings_io.is_dir_writable(base):
            return base
        # fallback
        return os.path.join(self._local_config_dir(), "output")

    def _resolve_output_dir(self, in_pdf: str) -> str:
        mode = self._get_output_mode_code()

        if mode == "neighbor":
            try:
                cand = os.path.dirname(os.path.abspath(in_pdf)) if in_pdf else ""
            except Exception:
                cand = ""
            if cand and _settings_io.is_dir_writable(cand):
                return cand
            # fallback
            fb = self._recommended_output_dir()
            try:
                os.makedirs(fb, exist_ok=True)
            except Exception:
                pass
            if in_pdf:
                self._log(f"[WARN] 入力PDFと同じフォルダに書き込めないため、出力先をフォールバックしました: {fb}")
            return fb

        if mode == "app_output":
            outd = self._recommended_output_dir()
            try:
                os.makedirs(outd, exist_ok=True)
            except Exception:
                pass
            return outd

        # custom
        outd = (self.var_out_custom.get() or "").strip()
        return outd

    def _update_output_dir_display(self) -> None:
        """Update output dir entry text/state according to mode + input path."""
        mode = self._get_output_mode_code()

        if mode == "custom":
            try:
                self.ent_out.configure(state="normal")
            except Exception:
                pass
            # display custom
            try:
                self.var_out.set((self.var_out_custom.get() or "").strip())
            except Exception:
                pass
            return

        # non-custom -> readonly
        try:
            self.ent_out.configure(state="readonly")
        except Exception:
            pass

        in_pdf = (self.var_in.get() or "").strip()
        if mode == "neighbor":
            if in_pdf.lower().endswith(".pdf") and os.path.isfile(in_pdf):
                try:
                    self.var_out.set(os.path.dirname(os.path.abspath(in_pdf)))
                except Exception:
                    self.var_out.set("")
            else:
                self.var_out.set("")
            return

        if mode == "app_output":
            try:
                self.var_out.set(self._recommended_output_dir(check_writable=False))
            except Exception:
                self.var_out.set("")
            return

        # default
        try:
            self.var_out.set("")
        except Exception:
            pass

    def _open_input_folder(self):
        """Open containing folder of input PDF."""
        p = (self.var_in.get() or "").strip()
        if p.lower().endswith(".pdf") and os.path.isfile(p):
            try:
                self._open_folder(os.path.dirname(os.path.abspath(p)))
                return
            except Exception:
                pass
        # fallback: app folder
        try:
            self._open_folder(self._app_base_dir())
        except Exception:
            pass

    def _open_output_folder(self):
        """Open current effective output folder."""
        in_pdf = (self.var_in.get() or "").strip()
        outd = self._resolve_output_dir(in_pdf)
        if not outd:
            outd = self._recommended_output_dir()
        try:
            os.makedirs(outd, exist_ok=True)
        except Exception:
            pass
        try:
            self._open_folder(outd)
        except Exception:
            pass

    def _on_drop(self, event):
        data = event.data.strip()
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]
        if data.lower().endswith(".pdf"):
            self._set_input_path(data)


    # -------------------------
    # 入力PDF変更の検知（手入力対応）
    # -------------------------
    def _on_input_path_changed(self, *args):
        # 手入力時の連続イベントをデバウンス（500ms）
        try:
            if self._in_trace_job is not None:
                safe_cancel(self.root, self._in_trace_job)
        except Exception as e:
            _log_exception_once('L4373', e)
        self._in_trace_job = self.root.after(500, self._update_pdf_page_range)

    def _set_input_path(self, path: str):
        self.var_in.set(path)
        self._update_pdf_page_range()
        try:
            self._update_output_dir_display()
        except Exception:
            pass
        self._save_config()

    def _update_pdf_page_range(self):
        """入力PDFのページ数を取得し、テストページSpinboxの範囲を更新します。"""
        p = self.var_in.get().strip()
        if not p.lower().endswith(".pdf") or not os.path.isfile(p):
            self.var_total_pages.set(0)
            try:
                self.sp_test_page.config(from_=1, to=1)
            except Exception as e:
                _log_exception_once('L4389', e)
            try:
                self._update_output_dir_display()
            except Exception:
                pass
            return

        try:
            with pdf_io.open_document(p) as d:
                total = int(d.page_count)
        except Exception:
            self.var_total_pages.set(0)
            return

        total = max(1, total)
        self.var_total_pages.set(total)

        try:
            self.sp_test_page.config(from_=1, to=total)
        except Exception as e:
            _log_exception_once('L4405', e)

        # 現在値が範囲外なら補正
        cur = int(self.var_test_page.get())
        if cur < 1:
            self.var_test_page.set(1)
        elif cur > total:
            # 表紙を避けたいケースが多いので、total>=2なら2を初期提示
            self.var_test_page.set(2 if total >= 2 else total)

        try:
            self._update_output_dir_display()
        except Exception:
            pass

        try:
            self._update_output_dir_display()
        except Exception:
            pass

    # -------------------------
    # テストページ（任意）実行
    # -------------------------
    def _run_test_page(self):
        in_pdf = self.var_in.get().strip()
        if not in_pdf.lower().endswith(".pdf") or not os.path.isfile(in_pdf):
            messagebox.showwarning("確認", "入力PDFを選択してください（または正しいパスを入力してください）。")
            return

        # SR / Safe mode
        safe_mode = False
        enable_sr = True
        try:
            safe_mode = bool(self.var_safe_mode.get()) if hasattr(self, "var_safe_mode") else False
        except Exception:
            safe_mode = False
        try:
            enable_sr = bool(self.var_enable_sr.get()) if hasattr(self, "var_enable_sr") else True
        except Exception:
            enable_sr = True
        if safe_mode:
            enable_sr = False

        if enable_sr:
            if not self._validate_model_path_or_guide():
                return

        # 読み方向（縦/横）: 必ず指定（auto廃止）
        rm = (self.var_reading_mode.get() or "").strip().lower()
        if rm not in ("vertical", "horizontal"):
            messagebox.showwarning("確認", "読み方向（縦書き / 横書き）を選択してください。")
            return

        # ページ番号（1-based）
        total = int(self.var_total_pages.get() or 0)
        page_no = int(self.var_test_page.get() or 1)
        if total >= 1:
            page_no = max(1, min(total, page_no))

        run_ocr = bool(self.var_test_run_ocr.get())

        # Gather settings
        model_path = self.var_model.get().strip()
        if not enable_sr:
            model_path = ""
        dpi = int(self.var_dpi.get())
        jpeg_q = int(self.var_jpeg.get())
        bg_scale = int(self.var_bg_scale.get()) if hasattr(self, 'var_bg_scale') else int(DEFAULT_BG_SCALE_PERCENT)
        max_out_dpi = int(self.var_max_out_dpi.get()) if hasattr(self, 'var_max_out_dpi') else int(DEFAULT_MAX_OUTPUT_DPI)
        bin_s = int(self.var_bin.get())
        bold_s = int(self.var_bold.get())
        tile = int(self.var_tile.get())
        page_mode = self.var_page_mode.get().strip()
        font_path = self.var_font.get().strip()

        enable_deskew = bool(self.var_deskew.get())
        deskew_max = float(self.var_deskew_max.get())

        # [Thread-safety] weights_dir はメインスレッド側で確定させ、ワーカーへ渡す（_ensure_weights_dir→_log がGUI直更新のため）
        weights_dir = self._ensure_weights_dir()

        # [Thread-safety] Tkinter変数(.get())はメインスレッドで読み取り、ワーカースレッドには値を渡す
        _test_auto_grayscale = bool(self.var_gray_auto.get()) if hasattr(self, "var_gray_auto") else bool(DEFAULT_AUTO_GRAYSCALE)
        _test_gray_color_ratio_percent = float(self.var_gray_ratio.get()) if hasattr(self, "var_gray_ratio") else float(DEFAULT_GRAY_COLOR_RATIO_PERCENT)
        _test_gray_jpeg_quality_offset = int(self.var_gray_q_offset.get()) if hasattr(self, "var_gray_q_offset") else int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET)


        self._save_config()

        self.stop_flag.clear()
        self._set_running(True)
        self.var_prog.set(f"テスト中: ページ {page_no}/{total if total else '?'}")

        def worker():
            t0 = time.time()
            self.msg_queue.put(("test_stage", "テスト: 初期化中（モデル読み込み等）"))
            try:
                engine = PdfOcrEnhanceEngine(
                    log_cb=self._enqueue_log,
                    progress_cb=lambda a, b: None,
                    stop_flag=self.stop_flag,
                    model_path=model_path,
                    base_dpi=dpi,
                    jpeg_quality=jpeg_q,
                    auto_grayscale=_test_auto_grayscale,
                    gray_color_ratio_percent=_test_gray_color_ratio_percent,
                    gray_jpeg_quality_offset=_test_gray_jpeg_quality_offset,
                    bg_scale_percent=bg_scale,
                    max_output_dpi=max_out_dpi,
                    binarize_strength=bin_s,
                    text_boldness=bold_s,
                    esrgan_tile=tile,
                    output_page_mode=page_mode,
                    font_path=font_path,
                    ocr_workers=0,  # テストは単ページなのでMPは使わず、素直に実行（安定優先）
                    enable_deskew=enable_deskew,
                    deskew_max_deg=deskew_max,
                    store_shrink=DEFAULT_STORE_SHRINK,
                    weights_dir=weights_dir,
                    enable_sr=enable_sr
                )                # reading direction override from GUI（必ず縦/横を指定）
                engine.reading_direction_mode = rm
                # cache engine for preview actions (re-OCR etc.)
                try:
                    with self._test_engine_lock:
                        self._test_engine = engine
                except Exception:
                    try:
                        self._test_engine = engine
                    except Exception:
                        pass
                self.msg_queue.put(("test_stage", "テスト: 1) PDFレンダリング → SR中（初回は時間がかかります）"))
                result = engine.test_page(in_pdf, page_no, run_ocr=run_ocr)
                dt = time.time() - t0
                self.msg_queue.put(("log", f"[INFO] テスト処理完了: {dt:.1f}s"))
                self.msg_queue.put(("preview_result", result))
            except Exception:
                self.msg_queue.put(("preview_error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    # -------------------------
    # プレビュー表示
    # -------------------------
    def _np_to_photo(self, rgb: np.ndarray, max_w: int = 560, max_h: int = 700) -> Tuple[ImageTk.PhotoImage, Tuple[int, int]]:
        """numpy RGB -> PhotoImage（表示用にリサイズ）

        目的:
          - プレビュー3面の見た目を揃える（各パネルが同じ表示サイズになる）
          - 文字太さ/二値化の変化が分かりやすいよう、拡大も許可する

        方針:
          1) max_w/max_h の枠に収まるように等比リサイズ（拡大も可）
          2) 枠サイズ(max_w,max_h)の白背景に中央貼り付け（レターボックス）
        """
        try:
            h, w = rgb.shape[:2]
        except Exception:
            h, w = 0, 0

        if h <= 0 or w <= 0:
            pil_canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            return ImageTk.PhotoImage(pil_canvas), (max_w, max_h)

        # 等比リサイズ（拡大も許可）
        scale = min(max_w / float(w), max_h / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

        # numpy -> PIL
        try:
            pil = Image.fromarray(rgb.astype("uint8"), mode="RGB")
        except Exception:
            try:
                pil = Image.fromarray(rgb)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
            except Exception:
                pil = Image.new("RGB", (w, h), (255, 255, 255))

        if (nw, nh) != (w, h):
            pil = pil.resize((nw, nh), Image.LANCZOS)

        # レターボックス（枠に合わせて中央配置）
        pil_canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        x = (max_w - nw) // 2
        y = (max_h - nh) // 2
        pil_canvas.paste(pil, (x, y))

        return ImageTk.PhotoImage(pil_canvas), (max_w, max_h)
    def _open_zoom_viewer(self, rgb: np.ndarray, title: str, panel_kind: str = "bg", override_bold: int = None, sync_callback=None, override_bin: int = None, bin_src_rgb: np.ndarray = None, bin_zero_rgb: np.ndarray = None, sync_bin_callback=None, owner_window=None):
        """テストプレビュー等を拡大表示する（Canvas + スクロール + ズーム）。"""
        self._assert_ui('_open_zoom_viewer')
        if rgb is None:
            return
        try:
            pil = Image.fromarray(rgb.astype("uint8"), mode="RGB")
        except Exception:
            try:
                pil = Image.fromarray(rgb)
            except Exception:
                return
        enable_bold = (str(panel_kind) == "bg")
        enable_bin = (bin_src_rgb is not None)
        if enable_bin:
            try:
                binit = int(override_bin) if (override_bin is not None) else int(getattr(self.var_bin, "get", lambda: DEFAULT_BINARIZE_STRENGTH)())
            except Exception:
                binit = DEFAULT_BINARIZE_STRENGTH
            binit = max(0, min(100, binit))
        else:
            binit = DEFAULT_BINARIZE_STRENGTH
        def _apply_bold(v: int):
            try:
                v = int(v)
                v = max(-100, min(100, v))
                self.var_bold.set(v)
                try:
                    self.sc_bold.set(v)
                except Exception as e:
                    _log_exception_once('L4569', e)
                self._save_config()
                self._log(f"[INFO] 文字太さ（閲覧）を反映: {v}")
            except Exception as e:
                _log_exception_once('L4573', e)

            # 呼び出し元（プレビューウィンドウ）へ同期通知
            if sync_callback:
                try:
                    sync_callback(int(v))
                except Exception as e:
                    _log_exception_once('L4580', e)


        viewer = ZoomImageViewer(
            (owner_window if owner_window is not None else self.root),
            pil,
            title=title,
            panel_kind=panel_kind,
            enable_bold_slider=enable_bold,
            bold_init=(int(override_bold) if (override_bold is not None) else (int(self.var_bold.get()) if enable_bold else 0)),
            on_apply_bold=_apply_bold if enable_bold else None,
            enable_bin_slider=enable_bin,
            bin_init=binit,
            bin_src_rgb=(bin_src_rgb if enable_bin else None),
            bin_zero_rgb=(bin_zero_rgb if enable_bin else None),
            sync_bin_callback=sync_bin_callback,
        )

        # 子ウィンドウ（拡大ビューア）の位置は、親ウィンドウの上辺と揃える
        # ※Toplevel生成直後は幅/高さが 1 の場合があるため、配置は after_idle で遅延実行する
        if owner_window is not None:
            try:
                # まず見える状態にして前面へ
                try:
                    viewer.deiconify()
                except Exception:
                    pass
                try:
                    viewer.lift()
                    viewer.focus_force()
                except Exception:
                    pass

                def _place_zoom():
                    try:
                        viewer.update_idletasks()
                        # 親の位置（スクリーン座標）
                        px = int(owner_window.winfo_rootx())
                        py = int(owner_window.winfo_rooty())
                        pw = int(owner_window.winfo_width())
                        sw = int(viewer.winfo_screenwidth())
                        sh = int(viewer.winfo_screenheight())

                        # 現在のウィンドウサイズ（未確定なら req/geometry から拾う）
                        vw = int(viewer.winfo_width() or 0)
                        vh = int(viewer.winfo_height() or 0)
                        if vw <= 50 or vh <= 50:
                            try:
                                vw = max(vw, int(viewer.winfo_reqwidth()))
                                vh = max(vh, int(viewer.winfo_reqheight()))
                            except Exception:
                                pass
                        if vw <= 50 or vh <= 50:
                            try:
                                g0 = str(viewer.geometry() or "")
                                gsz = g0.split('+', 1)[0]
                                if 'x' in gsz:
                                    ww, hh = gsz.split('x', 1)
                                    vw = max(vw, int(ww))
                                    vh = max(vh, int(hh))
                            except Exception:
                                pass
                        vw = max(200, int(vw or 0))
                        vh = max(200, int(vh or 0))

                        # 右側に出す（入らなければ左側）
                        x = px + pw + 10
                        if x + vw > sw:
                            x = px - vw - 10
                        # 上辺を揃える（少し上に寄せる）
                        # NOTE: Windowsの装飾（タイトルバー等）や個人の好みにより、
                        #       「完全に同じy」だと少し低く感じることがあるため、
                        #       既定で少しだけ上へオフセットする。
                        y = py - 110
                        # 画面内に収める
                        x = max(0, min(x, max(0, sw - vw)))
                        y = max(0, min(y, max(0, sh - vh)))

                        # 位置だけ反映（サイズは保持）
                        viewer.geometry(f"+{x}+{y}")

                        # 画面の裏に回るケース対策（Windows）
                        try:
                            viewer.lift()
                            viewer.attributes('-topmost', True)
                            viewer.after(200, lambda: viewer.attributes('-topmost', False))
                            viewer.focus_force()
                        except Exception:
                            pass
                    except Exception as e:
                        _log_exception_once('L_zoom_place', e)

                try:
                    viewer.after_idle(_place_zoom)
                except Exception:
                    viewer.after(0, _place_zoom)
            except Exception as e:
                _log_exception_once('L_zoom_align', e)

        return viewer
    def _show_preview_window(self, result: dict):
        """テスト結果プレビュー画面（実装は preview_window.py に分離）。"""
        return _show_preview_window_impl(self, result)

        ttk.Label(box, text=tips, justify="left").pack(anchor="w", padx=8, pady=6)
    def _log(self, s: str):
        """GUIログ（起動直後など txt_log 未生成でも落ちないようにする）"""
        self._assert_ui('_log')
        try:
            if hasattr(self, "txt_log") and self.txt_log is not None:
                self.txt_log.insert("end", s + "\n")
                self.txt_log.see("end")
            else:
                # UI生成前はコンソールへ
                print(s)
        except Exception:
            try:
                print(s)
            except Exception as e:
                _log_exception_once('L5127', e)

    def _enqueue_log(self, s: str):
        self.msg_queue.put(("log", s))

    def _normalize_path_str(self, p: str) -> str:
        """設定などから受け取ったパス文字列を正規化（不可視文字・引用符・空白・区切りなどを除去）"""
        if p is None:
            return ""
        p = str(p)
        # 不可視文字（BOM, ZWSP, LRM/RLM, 方向制御）を除去
        for ch in ["\ufeff", "\u200b", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"]:
            p = p.replace(ch, "")
        # 両端の空白・改行・引用符を除去
        p = p.strip().strip('"').strip("'").strip()
        # 展開
        p = os.path.expandvars(os.path.expanduser(p))
        # 区切り統一
        p = p.replace("/", os.sep)
        try:
            p = os.path.normpath(p)
        except Exception as e:
            _log_exception_once('L5149', e)
        return p

    def _resolve_existing_file(self, p: str, fallback_dir: str = "", fallback_name: str = "") -> str:
        """与えられたパスが実在しない場合でも、正規化・候補探索で実在ファイルを見つける"""
        cand = []
        if p:
            cand.append(p)
            cand.append(self._normalize_path_str(p))
        if fallback_dir and fallback_name:
            cand.append(os.path.join(fallback_dir, fallback_name))
        try:
            if p:
                pn = self._normalize_path_str(p)
                cand.append(os.path.join(os.path.dirname(pn), os.path.basename(pn)))
        except Exception as e:
            _log_exception_once('L5165', e)

        seen = set()
        for x in cand:
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            try:
                if os.path.isfile(x):
                    return x
            except Exception:
                continue
        return ""

    def _enqueue_progress(self, cur: int, total: int):
        self.msg_queue.put(("progress", (cur, total)))

    def _poll_queue(self):
        self._assert_ui('_poll_queue')
        if getattr(self, '_closing', False):
            return
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "progress":
                    cur, total = payload
                    self.var_prog.set(f"ページ処理中: {cur}/{total}")
                    self.pbar["maximum"] = total
                    self.pbar["value"] = cur
                elif kind == "stage":
                    # サブステージ表示（例: SR中/OCR中/保存中）
                    try:
                        self.var_prog.set(str(payload))
                    except Exception as e:
                        _log_exception_once('L5200', e)
                elif kind == "test_stage":
                    try:
                        self.var_prog.set(str(payload))
                    except Exception as e:
                        _log_exception_once('L5205', e)
                elif kind == "done":
                    out_path = str(payload)
                    self.var_prog.set("完了")
                    messagebox.showinfo("完了", f"出力しました:\n{out_path}")
                    self._set_running(False)
                elif kind == "cancelled":
                    out_path = str(payload)
                    self.var_prog.set("中断")
                    if out_path:
                        messagebox.showinfo("中断", f"中断しました。途中結果PDFを出力しました:\n{out_path}")
                    else:
                        messagebox.showinfo("中断", "中断しました。出力は作成されませんでした。")
                    self._set_running(False)
                elif kind == "cancelled_no_output":
                    self.var_prog.set("中断")
                    messagebox.showinfo("中断", "中断しました。設定により出力は作成されませんでした。")
                    self._set_running(False)
                elif kind == "error":
                    self.var_prog.set("エラー")
                    messagebox.showerror("エラー", str(payload))
                    self._set_running(False)
                elif kind == "preview_result":
                    self.var_prog.set("テスト完了")
                    try:
                        self._show_preview_window(payload)
                    except Exception:
                        tb = traceback.format_exc()
                        try:
                            self._log("[ERROR] プレビュー表示に失敗:")
                            self._log(tb)
                        except Exception as e:
                            _log_exception_once('L5225', e)
                        messagebox.showerror("プレビュー表示エラー", tb)
                    finally:
                        self._set_running(False)
                elif kind == "preview_error":
                    self.var_prog.set("テストエラー")
                    messagebox.showerror("テストエラー", str(payload))
                    self._set_running(False)
        except queue.Empty:
            pass
        if not getattr(self, '_closing', False):
            try:
                self._poll_job = self.root.after(100, self._poll_queue)
            except Exception as e:
                _log_exception_once('L_poll_after', e)
    def _set_running(self, running: bool):
        self._assert_ui('_set_running')
        self.btn_run["state"] = "disabled" if running else "normal"
        self.btn_stop["state"] = "normal" if running else "disabled"
        self.btn_reset["state"] = "disabled" if running else "normal"
        try:
            self.btn_test["state"] = "disabled" if running else "normal"
        except Exception as e:
            _log_exception_once('L5244', e)

    # -------------------------
    # Settings persistence
    # -------------------------
    def _safe_int(self, v, default: int, lo: int, hi: int) -> int:
        try:
            x = int(v)
            return max(lo, min(hi, x))
        except Exception:
            return default

    def _read_config(self, path: str) -> Optional[dict]:
        cfg, err = _settings_io.read_config(path)
        if cfg is not None:
            # If recovered from backup, notify and re-save later.
            try:
                if bool(cfg.get('_recovered_from_backup', False)):
                    self._log(f"[WARN] 設定ファイルが破損していたためバックアップ(.bak)から復旧しました: {path}")
                    try:
                        prim_err = str(cfg.get('_primary_read_error', '') or '')
                        if prim_err:
                            self._log(f"[WARN] 元の設定読み込みエラー: {prim_err}")
                    except Exception:
                        pass
                    self._needs_save_config_after_load = True
            except Exception:
                pass
            return cfg
        if err is not None:
            self._log(f"[WARN] 設定読み込み失敗（無視）: {path} / {err}")
        return None

    def _apply_config_dict(self, cfg: dict):
        dpi = self._safe_int(cfg.get("dpi"), DEFAULT_DPI, 72, 600)
        jpeg_q = self._safe_int(cfg.get("jpeg_quality"), DEFAULT_JPEG_QUALITY, 40, 100)
        bin_s = self._safe_int(cfg.get("binarize_strength"), DEFAULT_BINARIZE_STRENGTH, 0, 100)
        # Migration: older builds used 60 as the default binarize strength. New default is 0.
        try:
            _bin_raw = cfg.get('binarize_strength', None)
            if _bin_raw is not None and int(float(_bin_raw)) == 60 and not bool(cfg.get('_migrated_default_bin0', False)):
                bin_s = 0
                cfg['_migrated_default_bin0'] = True
                self._needs_save_config_after_load = True
        except Exception as e:
            _log_exception_once('cfg_migrate_bin0', e)

        bold_s = self._safe_int(cfg.get("text_boldness"), DEFAULT_TEXT_BOLDNESS, -100, 100)
        tile = self._safe_int(cfg.get("esrgan_tile"), DEFAULT_ESRGAN_TILE, 0, 2048)
        bg_scale = self._safe_int(cfg.get("bg_scale_percent"), DEFAULT_BG_SCALE_PERCENT, 10, 100)
        max_out_dpi = self._safe_int(cfg.get("max_output_dpi"), DEFAULT_MAX_OUTPUT_DPI, 0, 2400)

        # グレースケール自動（ページごと）
        auto_gray = bool(cfg.get("auto_grayscale", DEFAULT_AUTO_GRAYSCALE))
        try:
            gray_ratio = float(cfg.get("gray_color_ratio_percent", DEFAULT_GRAY_COLOR_RATIO_PERCENT))
        except Exception:
            gray_ratio = float(DEFAULT_GRAY_COLOR_RATIO_PERCENT)
        gray_ratio = max(0.0, min(5.0, gray_ratio))
        gray_qoff = self._safe_int(cfg.get("gray_jpeg_quality_offset"), DEFAULT_GRAY_JPEG_QUALITY_OFFSET, -40, 0)

        page_mode = str(cfg.get("output_page_mode", DEFAULT_OUTPUT_PAGE_MODE)).strip()
        if page_mode not in ("original", "pixel"):
            page_mode = DEFAULT_OUTPUT_PAGE_MODE

        font_path = str(cfg.get("font_path", DEFAULT_FONT_PATH)).strip()
        if not font_path:
            font_path = "AUTO"
        model_path = str(cfg.get("model_path", self.var_model.get())).strip()

        ocr_workers = self._safe_int(cfg.get("ocr_workers"), DEFAULT_OCR_WORKERS, 0, 4)
        deskew = bool(cfg.get("enable_deskew", DEFAULT_ENABLE_DESKEW))
        try:
            deskew_max = float(cfg.get("deskew_max_deg", DEFAULT_DESKEW_MAX_DEG))
        except Exception:
            deskew_max = float(DEFAULT_DESKEW_MAX_DEG)

        self.var_dpi.set(dpi)
        self.var_jpeg.set(jpeg_q)
        try:
            if hasattr(self, "var_gray_auto"):
                self.var_gray_auto.set(1 if auto_gray else 0)
            if hasattr(self, "var_gray_ratio"):
                self.var_gray_ratio.set(float(gray_ratio))
            if hasattr(self, "var_gray_q_offset"):
                self.var_gray_q_offset.set(int(gray_qoff))
        except Exception as e:
            _log_exception_once('L5312', e)
        if hasattr(self, "var_bg_scale"):
            self.var_bg_scale.set(bg_scale)
        if hasattr(self, "var_max_out_dpi"):
            self.var_max_out_dpi.set(max_out_dpi)

        self.var_bin.set(bin_s)
        self.var_bold.set(bold_s)
        self.var_tile.set(tile)
        self.var_page_mode.set(page_mode)

        # 読み方向は保存しない（必ずGUIで縦/横を選択）
        self.var_reading_mode.set("")

        self.var_font.set(font_path)

        self.var_ocr_workers.set(ocr_workers)
        self.var_deskew.set(deskew)
        self.var_deskew_max.set(deskew_max)

        # SR / Safe mode
        try:
            if hasattr(self, "var_enable_sr"):
                self.var_enable_sr.set(bool(cfg.get("enable_sr", True)))
        except Exception:
            pass
        try:
            if hasattr(self, "var_safe_mode"):
                self.var_safe_mode.set(bool(cfg.get("safe_mode", False)))
        except Exception:
            pass
        try:
            if hasattr(self, "var_keep_partial_on_cancel"):
                self.var_keep_partial_on_cancel.set(bool(cfg.get("keep_partial_on_cancel", True)))
        except Exception:
            pass
        try:
            self._on_sr_mode_changed(save=False)
        except Exception:
            pass

        # テスト設定
        try:
            self.var_test_page.set(int(cfg.get("test_page", 1)))
        except Exception:
            self.var_test_page.set(1)
        try:
            self.var_test_run_ocr.set(bool(cfg.get("test_run_ocr", True)))
        except Exception:
            self.var_test_run_ocr.set(True)

        if model_path:
            self.var_model.set(model_path)

        try:
            self.sc_jpeg.set(jpeg_q)
        except Exception as e:
            _log_exception_once('L5348', e)
        try:
            self.sc_bin.set(bin_s)
        except Exception as e:
            _log_exception_once('L5352', e)

        # 出力先（互換キーも考慮）
        custom_out = str(
            cfg.get("custom_output_dir")
            or cfg.get("last_output_dir")
            or cfg.get("last_out")
            or ""
        ).strip()

        mode_raw = str(cfg.get("output_mode") or cfg.get("out_mode") or "").strip().lower()
        if mode_raw not in ("neighbor", "app_output", "custom"):
            # Backward compatibility: if old config had an explicit output dir, treat it as custom.
            if ("output_mode" not in cfg) and custom_out:
                mode_raw = "custom"
            else:
                mode_raw = "neighbor"

        geo = str(cfg.get("window_geometry") or cfg.get("geo") or cfg.get("geometry") or "").strip()

        # Window startup behavior
        restore_window = False
        try:
            if str(cfg.get("window_mode", "")).strip().lower() == "restore":
                restore_window = True
        except Exception:
            pass
        try:
            restore_window = bool(cfg.get("window_restore", restore_window) or cfg.get("restore_window", restore_window))
        except Exception:
            pass

        fixed_w = self._safe_int(cfg.get("fixed_window_width") or cfg.get("window_fixed_width") or cfg.get("main_width"), 900, 500, 2400)
        fixed_h = self._safe_int(cfg.get("fixed_window_height") or cfg.get("window_fixed_height") or cfg.get("main_height"), 800, 400, 2000)

        try:
            if hasattr(self, "var_window_restore"):
                self.var_window_restore.set(bool(restore_window))
            if hasattr(self, "var_window_fixed_w"):
                self.var_window_fixed_w.set(int(fixed_w))
            if hasattr(self, "var_window_fixed_h"):
                self.var_window_fixed_h.set(int(fixed_h))
        except Exception as e:
            _log_exception_once('cfg_winvars', e)

        if custom_out:
            try:
                self.var_out_custom.set(custom_out)
            except Exception:
                pass
        try:
            self._set_output_mode_code(mode_raw)
        except Exception:
            pass

        # Apply geometry (restore or fixed). Clamp to screen bounds to avoid off-screen windows.
        try:
            sw = int(self.root.winfo_screenwidth())
            sh = int(self.root.winfo_screenheight())
        except Exception:
            sw, sh = 99999, 99999

        def _parse_geo(_g: str):
            m = re.match(r"^\s*(\d+)x(\d+)([+-]\d+)?([+-]\d+)?\s*$", str(_g))
            if not m:
                return None
            w = int(m.group(1))
            h = int(m.group(2))
            x = int(m.group(3)) if m.group(3) else 0
            y = int(m.group(4)) if m.group(4) else 0
            has_pos = (m.group(3) is not None and m.group(4) is not None)
            return w, h, x, y, has_pos

        parsed = _parse_geo(geo) if geo else None

        geo2 = None
        try:
            if restore_window and parsed is not None:
                w, h, x, y, has_pos = parsed
                w = max(400, min(int(w), int(sw)))
                h = max(300, min(int(h), int(sh)))
                if has_pos:
                    x = max(-w + 50, min(int(x), int(sw) - 50))
                    y = max(0, min(int(y), int(sh) - 50))
                    geo2 = f"{w}x{h}{x:+d}{y:+d}"
                else:
                    geo2 = f"{w}x{h}"
            else:
                w = int(fixed_w)
                h = int(fixed_h)
                offs = ""
                if parsed is not None and parsed[4]:
                    x = max(-w + 50, min(int(parsed[2]), int(sw) - 50))
                    y = max(0, min(int(parsed[3]), int(sh) - 50))
                    offs = f"{x:+d}{y:+d}"
                geo2 = f"{w}x{h}{offs}"
        except Exception as e:
            _log_exception_once('cfg_geo_parse', e)
            geo2 = None

        if geo2:
            try:
                self.root.geometry(geo2)
            except Exception as e:
                _log_exception_once('L5366', e)

        self._refresh_model_choices(keep_current=True)

    def _detect_and_load_config(self):
        p_port = self._portable_config_path()
        p_local = self._local_config_path()

        cfg = self._read_config(p_port)
        if cfg is not None:
            self.var_portable.set(True)
            self._apply_config_dict(cfg)
            self._update_config_path_label()
            self._log(f"[INFO] 設定読み込み（ポータブル）: {p_port}")
            # One-time config migration write-back (best-effort)
            if getattr(self, '_needs_save_config_after_load', False):
                self._needs_save_config_after_load = False
                try:
                    self._save_config()
                    self._log('[INFO] 設定マイグレーション: 二値化デフォルトを0へ更新しました。')
                except Exception as e:
                    _log_exception_once('cfg_migrate_save_portable', e)
            return

        cfg = self._read_config(p_local)
        if cfg is not None:
            self.var_portable.set(False)
            self._apply_config_dict(cfg)
            self._update_config_path_label()
            self._log(f"[INFO] 設定読み込み（ローカル）: {p_local}")
            # One-time config migration write-back (best-effort)
            if getattr(self, '_needs_save_config_after_load', False):
                self._needs_save_config_after_load = False
                try:
                    self._save_config()
                    self._log('[INFO] 設定マイグレーション: 二値化デフォルトを0へ更新しました。')
                except Exception as e:
                    _log_exception_once('cfg_migrate_save_local', e)
            return

        # No config: choose a sensible default for zip distribution.
        try:
            default_portable = bool(_settings_io.detect_portable_mode(self._app_base_dir()))
            self.var_portable.set(default_portable)
        except Exception:
            pass
        self._update_config_path_label()
        self._log(f"[INFO] 設定ファイルなし（初回起動）: {self._config_path()}")

    def _save_config(self) -> bool:
        path = self._config_path()
        # Window preference values (persisted)
        restore_window = bool(self.var_window_restore.get()) if hasattr(self, "var_window_restore") else False
        fixed_w = int(self.var_window_fixed_w.get()) if hasattr(self, "var_window_fixed_w") else 900
        fixed_h = int(self.var_window_fixed_h.get()) if hasattr(self, "var_window_fixed_h") else 800

        # Geometry sanity: avoid saving bogus 1x1 geometry during early startup.
        geo_now = ""
        geo_ok = ""
        try:
            geo_now = str(self.root.winfo_geometry())
        except Exception:
            geo_now = ""
        try:
            mm = re.match(r"^\s*(\d+)x(\d+)([+-]\d+)?([+-]\d+)?\s*$", geo_now)
            if mm:
                w = int(mm.group(1))
                h = int(mm.group(2))
                if w >= 200 and h >= 200:
                    geo_ok = geo_now
        except Exception:
            geo_ok = ""

        if not geo_ok:
            # Keep previous geometry if available, else fall back to fixed geometry.
            prev = self._read_config(path)
            try:
                if isinstance(prev, dict):
                    pgeo = str(prev.get("window_geometry") or prev.get("geo") or prev.get("geometry") or "").strip()
                    if pgeo:
                        geo_ok = pgeo
            except Exception:
                pass
        if not geo_ok:
            geo_ok = f"{fixed_w}x{fixed_h}"

        # Output destination (persist custom dir separately so switching modes doesn't overwrite it)
        try:
            out_mode = str(self._get_output_mode_code())
        except Exception:
            out_mode = "neighbor"
        custom_out_dir = ""
        try:
            custom_out_dir = (self.var_out_custom.get() or "").strip()
        except Exception:
            custom_out_dir = ""
        cfg = {
            "app": APP_FULLNAME,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "portable_mode": bool(self.var_portable.get()),
            "model_path": self.var_model.get().strip(),
            "dpi": int(self.var_dpi.get()),
            "jpeg_quality": int(self.var_jpeg.get()),
            "auto_grayscale": bool(self.var_gray_auto.get()) if hasattr(self, "var_gray_auto") else bool(DEFAULT_AUTO_GRAYSCALE),
            "gray_color_ratio_percent": float(self.var_gray_ratio.get()) if hasattr(self, "var_gray_ratio") else float(DEFAULT_GRAY_COLOR_RATIO_PERCENT),
            "gray_jpeg_quality_offset": int(self.var_gray_q_offset.get()) if hasattr(self, "var_gray_q_offset") else int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET),
            "binarize_strength": int(self.var_bin.get()),
            "text_boldness": int(self.var_bold.get()),
            "esrgan_tile": int(self.var_tile.get()),
            "output_page_mode": self.var_page_mode.get().strip(),
            "bg_scale_percent": int(self.var_bg_scale.get()) if hasattr(self, "var_bg_scale") else DEFAULT_BG_SCALE_PERCENT,
            "max_output_dpi": int(self.var_max_out_dpi.get()) if hasattr(self, "var_max_out_dpi") else DEFAULT_MAX_OUTPUT_DPI,
            "font_path": self.var_font.get().strip(),
            "ocr_workers": int(self.var_ocr_workers.get()),
            "enable_sr": bool(self.var_enable_sr.get()) if hasattr(self, "var_enable_sr") else True,
            "safe_mode": bool(self.var_safe_mode.get()) if hasattr(self, "var_safe_mode") else False,
            "keep_partial_on_cancel": bool(self.var_keep_partial_on_cancel.get()) if hasattr(self, "var_keep_partial_on_cancel") else True,
            "enable_deskew": bool(self.var_deskew.get()),
            "deskew_max_deg": float(self.var_deskew_max.get()),
            "test_page": int(self.var_test_page.get()) if hasattr(self, "var_test_page") else 1,
            "test_run_ocr": bool(self.var_test_run_ocr.get()) if hasattr(self, "var_test_run_ocr") else True,
            "output_mode": out_mode,
            "custom_output_dir": custom_out_dir,
            # backward compatible keys
            "last_output_dir": custom_out_dir,
            "last_out": custom_out_dir,
            "window_mode": ("restore" if restore_window else "fixed"),
            "window_restore": bool(restore_window),
            "fixed_window_width": int(fixed_w),
            "fixed_window_height": int(fixed_h),
            "window_geometry": geo_ok,
        }

        ok, err = _settings_io.write_config(path, cfg)
        if ok:
            self._update_config_path_label()
            return True

        self._log(f"[WARN] 設定保存失敗: {path} / {err}")

        if self.var_portable.get():
            self._log("[WARN] ポータブル保存に失敗したため、LOCALAPPDATAへ切り替えます。")
            self.var_portable.set(False)
            self._update_config_path_label()
            path2 = self._config_path()
            ok2, err2 = _settings_io.write_config(path2, cfg)
            if ok2:
                return True
            self._log(f"[WARN] フォールバック保存も失敗: {err2}")
        return False


    def _on_toggle_portable(self):
        self._update_config_path_label()
        target = self._config_path()

        cfg = self._read_config(target)
        if cfg is not None:
            self._apply_config_dict(cfg)
            self._log(f"[INFO] 切替先の設定を読み込み: {target}")
        else:
            self._save_config()
            self._log(f"[INFO] 切替先に設定が無いため、現在の設定を保存: {target}")


    def _on_window_prefs_changed(self):
        """Handle window startup preference changes.

        - If restore is OFF: apply fixed width immediately (keep current height/position when possible).
        - Always persist preferences to config.
        """
        try:
            restore = bool(self.var_window_restore.get()) if hasattr(self, "var_window_restore") else False
            fixed_w = int(self.var_window_fixed_w.get()) if hasattr(self, "var_window_fixed_w") else 900

            # keep current height (user may have resized vertically)
            try:
                cur_h = int(self.root.winfo_height())
            except Exception:
                cur_h = 0
            fixed_h = int(self.var_window_fixed_h.get()) if hasattr(self, "var_window_fixed_h") else 800
            if cur_h >= 200:
                fixed_h = cur_h
            if hasattr(self, "var_window_fixed_h"):
                try:
                    self.var_window_fixed_h.set(int(fixed_h))
                except Exception:
                    pass

            if not restore:
                # preserve last offsets if possible
                offs = ""
                try:
                    g = str(self.root.winfo_geometry())
                    mm = re.match(r"^\s*\d+x\d+([+-]\d+)([+-]\d+)\s*$", g)
                    if mm:
                        x = int(mm.group(1))
                        y = int(mm.group(2))
                        offs = f"{x:+d}{y:+d}"
                except Exception:
                    offs = ""
                try:
                    self.root.geometry(f"{fixed_w}x{fixed_h}{offs}")
                except Exception as e:
                    _log_exception_once('winpref_apply', e)
        except Exception as e:
            _log_exception_once('winpref', e)

        # Persist preferences (best-effort)
        try:
            self._save_config()
        except Exception as e:
            _log_exception_once('winpref_save', e)

    # -------------------------
    # Reset defaults
    # -------------------------
    def _reset_defaults(self):
        if not messagebox.askyesno("確認", "設定を初期値に戻しますか？"):
            return

        self.var_dpi.set(DEFAULT_DPI)
        self.var_jpeg.set(DEFAULT_JPEG_QUALITY)
        try:
            if hasattr(self, "var_gray_auto"):
                self.var_gray_auto.set(1 if DEFAULT_AUTO_GRAYSCALE else 0)
            if hasattr(self, "var_gray_ratio"):
                self.var_gray_ratio.set(float(DEFAULT_GRAY_COLOR_RATIO_PERCENT))
            if hasattr(self, "var_gray_q_offset"):
                self.var_gray_q_offset.set(int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET))
        except Exception as e:
            _log_exception_once('L5471', e)
        if hasattr(self, "var_bg_scale"):
            self.var_bg_scale.set(DEFAULT_BG_SCALE_PERCENT)
        if hasattr(self, "var_max_out_dpi"):
            self.var_max_out_dpi.set(DEFAULT_MAX_OUTPUT_DPI)
        self.var_bin.set(DEFAULT_BINARIZE_STRENGTH)
        self.var_bold.set(DEFAULT_TEXT_BOLDNESS)
        self.var_tile.set(DEFAULT_ESRGAN_TILE)
        self.var_page_mode.set(DEFAULT_OUTPUT_PAGE_MODE)
        try:
            self.var_reading_mode.set("")
        except Exception as e:
            _log_exception_once('L5483', e)
        self.var_font.set(DEFAULT_FONT_PATH)
        self.var_model.set(self._default_model_path())
        try:
            if hasattr(self, "var_enable_sr"):
                self.var_enable_sr.set(True)
            if hasattr(self, "var_safe_mode"):
                self.var_safe_mode.set(False)
            if hasattr(self, "var_keep_partial_on_cancel"):
                self.var_keep_partial_on_cancel.set(True)
            self._on_sr_mode_changed(save=False)
        except Exception:
            pass

        self.var_ocr_workers.set(DEFAULT_OCR_WORKERS)
        self.var_deskew.set(DEFAULT_ENABLE_DESKEW)
        self.var_deskew_max.set(DEFAULT_DESKEW_MAX_DEG)

        try:
            self.sc_jpeg.set(DEFAULT_JPEG_QUALITY)
        except Exception as e:
            _log_exception_once('L5494', e)
        try:
            self.sc_bin.set(DEFAULT_BINARIZE_STRENGTH)
        except Exception as e:
            _log_exception_once('L5498', e)

        self._refresh_model_choices(keep_current=True)
        self._log("[INFO] 設定を初期値に戻しました。")
        self._save_config()
    # -------------------------
    # Help
    # -------------------------
    def _help_text(self) -> str:
        return f"""\
{APP_FULLNAME} 使い方（詳細）

■ 0. クイックスタート
1) 入力PDFを選択
2) 出力フォルダを選択
3) ESRGAN重み(.pth) を選択（weightsフォルダに置くと自動表示）
4) フォント（CJK対応TTF）を指定（既定: {DEFAULT_FONT_PATH}）
5) [開始] で変換

■ 1. 出力について
- 出力PDFは「背景画像(JPEG) + 透明文字(OCR)」です
- 同名ファイルがある場合は xxx(1).pdf のように自動で番号を付けます
- 出力ページモード:
  - original: 元PDFの用紙サイズ(pt)を維持（推奨）
  - pixel: 画像ピクセルをそのままptとして扱う（特殊用途）

■ 2. 画質と容量の調整（重要）
● DPI（PDF→画像化）: 既定 {DEFAULT_DPI}
- 高いほど精細ですが、SR/OCRともに重くなります
- SR(outscale=4) を使う場合、実効DPIは概ね DPI×4 になりやすいです

● 出力DPI上限（original）: 既定 {DEFAULT_MAX_OUTPUT_DPI}
- originalモードで、実効DPIが上限を超える場合に背景画像を自動縮小して容量を抑えます
- 例: DPI=150, SR×4 → 実効約600 → 上限400なら 2/3 に縮小

● 背景縮小率（出力）
- 保存前に背景を % で縮小します（透明文字も同倍率で縮小するのでズレません）
- 50%で容量は概ね1/4（ピクセル数が1/4）になります

● JPEG品質（背景）
- 70〜85がバランス
- 低いほど容量は減りますが、ブロックノイズが増えます

● グレースケール自動（ほぼ白黒ページを軽量化）
- ほぼ白黒ページを自動判定してグレー化し、容量を削減します
- 色率閾値(%) が小さいほど「色を残す」方向です
  - 推奨 0.3%: 多くの本文ページがグレー化されやすい
  - カラー保持なら 0.0%推奨（少しでも色があればカラーのまま）
- 黄ばみは捨ててグレー化する挙動です（紙色は維持しません）

■ 3. 読み方向（縦書き/横書き）
- 本バージョンでは auto（自動判定）を廃止しました。**必ず縦書き/横書きを選択**してください（誤ると読み順や配置が崩れます）。
- vertical（縦書き）:
  - 読み順は「右→左（列）」を優先して整列します（一般的な縦書き文書向け）。
  - OCRの角度情報が不安定でも「縦扱い」に寄せます（横長の見出し等は横扱いにフォールバック）。
- horizontal（横書き）:
  - 読み順は「上→下（行）」→「左→右」を優先して整列します。
  - 角度が明確に縦のトークン（欄外注など）は縦扱いになる場合があります。
- この選択は保存しません（毎回選択してください）。

■ 4. SR（Real-ESRGAN）と tile
- GPU（CUDA）が使える場合は device=cuda で高速化します。GPUが無い場合もCPUで動作します（遅くなります）
- tile は分割サイズです。大きいほどタイル数が減り速くなることがありますが、VRAM不足で失敗する場合があります
- 本ツールは、OOM（メモリ不足）発生時のみ tile を段階的に下げて自動リトライします

■ 5. OCR・画質補正
- 二値化強度: 背景を白黒寄りにして文字を浮かせます（上げすぎると潰れます）
- 太字化: 背景画像の文字を少し太くして読みやすくします（閲覧性向上）
- Deskew: 斜めスキャンの傾きを補正します（斜めが気になる場合のみON）
- OCR並列: SR（GPU/CPU）中にOCR（CPU）を並走させることで短縮することがあります（CPU負荷に注意）

■ 6. モデル重み・フォント
- ESRGAN重み(.pth) を指定してください（例: {DEFAULT_MODEL_FILENAME}）
- weightsフォルダに置くと自動検出されプルダウンに出ます
- 無い場合は「案内」ボタンで公式ページを開けます:
  - {REAL_ESRGAN_RELEASES_URL}
  - {REAL_ESRGAN_ANIME_DOC_URL}
- フォントはCJK対応TTF推奨（IPAexなど）。指定したフォントで透明文字を埋め込みます

■ 7. 安定性（大規模PDF）
- ページ数が多い場合、一定ページごとに出力PDFを保存→再オープンしてメモリを抑えます（ログに flush_every が出ます）
- fitz.TOOLS.store_shrink を適用してメモリ断片化を抑えます

■ 8. 埋め込み品質のフォールバック
- ページごとに埋め込みエラー率を計測し、必要に応じて安全な設定へ段階的に切り替えます（ログの fb=...）
- まずは埋め込みを優先し、必要な場合だけ rotate/xscale などを保守的にします

■ 9. よくある改善パターン
- 出力が大きい: DPIを下げる（例:100）/ 出力DPI上限=400 / 背景縮小率を下げる / グレー自動ON
- 遅い: DPIを下げる / GPU利用 / tile調整 / SRをOFF
- 文字選択が途切れる: 縦書きは連続文になるよう調整済み。症状が残る場合はスクショとログを共有してください
"""

    def _open_log_file(self):
        try:
            p = get_log_path()
            if not p:
                messagebox.showwarning("ログ", "ログファイルが見つかりません。")
                return
            try:
                if not os.path.isfile(p):
                    with open(p, "a", encoding="utf-8", errors="replace"):
                        pass
            except Exception:
                pass

            if sys.platform.startswith("win"):
                try:
                    os.startfile(p)  # type: ignore
                    return
                except Exception:
                    import subprocess
                    subprocess.Popen(["explorer", "/select,", p])
                    return
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", p])
                return
            else:
                import subprocess
                subprocess.Popen(["xdg-open", p])
                return
        except Exception as e:
            _log_exception_once("open_log", e)
            try:
                messagebox.showerror("ログ", f"ログを開けませんでした:\n{e}")
            except Exception:
                pass


    def _collect_environment_diagnostics(self) -> dict:
        """環境差分切り分け用の診断情報を収集（best effort）"""
        self._assert_ui('_collect_environment_diagnostics')
        info: dict = {}
        try:
            import datetime as _dt
            info["timestamp_local"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        info["app"] = APP_FULLNAME
        try:
            info["python"] = sys.version.replace("\n", " ")
            info["python_executable"] = sys.executable
        except Exception:
            pass
        try:
            info["platform"] = platform.platform()
        except Exception:
            pass
        try:
            info["log_file"] = get_log_path()
        except Exception:
            pass

        # Package versions (best effort). Prefer distribution metadata when available.
        packages: dict = {}

        def _pkgver(dist_names, import_name=None, attr="__version__"):
            try:
                import importlib.metadata as _md  # Python 3.8+
            except Exception:
                _md = None
            if _md is not None:
                for dn in dist_names:
                    try:
                        return _md.version(dn)
                    except Exception:
                        pass
            if import_name:
                try:
                    mod = __import__(import_name)
                    return getattr(mod, attr, "?")
                except Exception:
                    return None
            return None

        # numpy / opencv / fitz(PyMuPDF) / Pillow / torch / realesrgan / yomitoku
        packages["numpy"] = _pkgver(["numpy"], "numpy")
        packages["opencv"] = _pkgver(["opencv-python", "opencv-python-headless"], "cv2")
        packages["fitz"] = _pkgver(["PyMuPDF"], "fitz")
        packages["pillow"] = _pkgver(["Pillow"], "PIL")
        packages["torch"] = _pkgver(["torch"], "torch")
        packages["realesrgan"] = _pkgver(["realesrgan"], "realesrgan")
        packages["yomitoku"] = _pkgver(["yomitoku"], "yomitoku")
        info["packages"] = packages

        # Torch CUDA/device (best effort)
        try:
            import torch as _torch  # type: ignore
            tinfo: dict = {}
            tinfo["version"] = getattr(_torch, "__version__", "?")
            try:
                tinfo["cuda_available"] = bool(_torch.cuda.is_available())
                tinfo["cuda_device_count"] = int(_torch.cuda.device_count()) if _torch.cuda.is_available() else 0
                if _torch.cuda.is_available() and _torch.cuda.device_count() >= 1:
                    tinfo["cuda_device0_name"] = str(_torch.cuda.get_device_name(0))
                try:
                    tinfo["cuda_runtime_version"] = str(getattr(_torch.version, "cuda", None))
                except Exception:
                    pass
                # heuristic selected device
                tinfo["selected_device"] = ("cuda:0" if _torch.cuda.is_available() else "cpu")
            except Exception:
                pass
            info["torch_info"] = tinfo
        except Exception:
            pass

        # Weights folder status (best effort)
        weights: dict = {}
        wdir = None
        try:
            if hasattr(self, "_recommended_weights_dir"):
                wdir = self._recommended_weights_dir()
        except Exception:
            wdir = None
        if wdir:
            weights["recommended_dir"] = wdir
            try:
                weights["dir_exists"] = bool(os.path.isdir(wdir))
            except Exception:
                pass
            try:
                fdir = os.path.join(wdir, "fonts")
                weights["fonts_dir"] = fdir
                weights["fonts_dir_exists"] = bool(os.path.isdir(fdir))
            except Exception:
                pass

            # Expected Real-ESRGAN model filename
            expected_name = "RealESRGAN_x4plus_anime_6B.pth"
            try:
                from weights_manager import DEFAULT_ESRGAN  # type: ignore
                expected_name = getattr(DEFAULT_ESRGAN, "filename", expected_name)
            except Exception:
                pass
            weights["expected_model_filename"] = expected_name
            try:
                expected_path = os.path.join(wdir, expected_name)
                weights["expected_model_path"] = expected_path
                weights["expected_model_exists"] = bool(os.path.isfile(expected_path))
            except Exception:
                pass

        # Current GUI-selected paths
        try:
            if hasattr(self, "var_model"):
                weights["gui_model_path"] = self.var_model.get().strip()
                try:
                    weights["gui_model_exists"] = bool(os.path.isfile(weights["gui_model_path"]))
                except Exception:
                    pass
            if hasattr(self, "var_font"):
                weights["gui_font_path"] = self.var_font.get().strip()
                try:
                    weights["gui_font_exists"] = bool(os.path.isfile(weights["gui_font_path"]))
                except Exception:
                    pass
        except Exception:
            pass
        if weights:
            info["weights"] = weights

        # Minimal GUI settings snapshot (support-friendly)
        try:
            settings = {
                "input_pdf": self.var_in.get().strip() if hasattr(self, "var_in") else "",
                "output_mode": (self._get_output_mode_code() if hasattr(self, "_get_output_mode_code") else "?"),
                "output_dir_effective": self.var_out.get().strip() if hasattr(self, "var_out") else "",
                "output_dir_custom": self.var_out_custom.get().strip() if hasattr(self, "var_out_custom") else "",
                "reading_mode": (self.var_reading_mode.get() or "").strip() if hasattr(self, "var_reading_mode") else "",
                "dpi": int(self.var_dpi.get()) if hasattr(self, "var_dpi") else None,
                "jpeg_quality": int(self.var_jpeg.get()) if hasattr(self, "var_jpeg") else None,
            }
            # advanced toggles (existence-checked)
            try:
                settings["auto_grayscale"] = bool(self.var_gray_auto.get()) if hasattr(self, "var_gray_auto") else bool(DEFAULT_AUTO_GRAYSCALE)
            except Exception:
                pass
            try:
                settings["ocr_workers"] = int(self.var_ocr_workers.get()) if hasattr(self, "var_ocr_workers") else None
            except Exception:
                pass
            info["settings"] = settings
        except Exception:
            pass

        return info

    def _log_environment_diagnostics(self):
        """環境診断情報をログに一括出力"""
        self._assert_ui('_log_environment_diagnostics')
        try:
            info = self._collect_environment_diagnostics()
            txt = json.dumps(info, ensure_ascii=False, indent=2)
            self._log("[DIAG] ===== Environment diagnostics BEGIN =====")
            for line in txt.splitlines():
                self._log("[DIAG] " + line)
            self._log("[DIAG] ===== Environment diagnostics END =====")
            try:
                messagebox.showinfo("環境診断", "環境診断情報をログに出力しました。\n「ログを開く」から確認できます。")
            except Exception:
                pass
        except Exception as e:
            _log_exception_once("env_diag", e)
            try:
                messagebox.showerror("環境診断", f"環境診断情報の出力に失敗しました:\n{e}")
            except Exception:
                pass

    def _copy_diagnostic_info(self):
        """Support用の診断情報をクリップボードにコピー"""
        self._assert_ui('_copy_diagnostic_info')
        try:
            info = self._collect_environment_diagnostics()
            text = json.dumps(info, ensure_ascii=False, indent=2)
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                messagebox.showinfo("診断情報", "診断情報をクリップボードにコピーしました。\n不具合報告時に貼り付けてください。")
            except Exception:
                # clipboard unavailable
                messagebox.showinfo("診断情報", text)
        except Exception as e:
            _log_exception_once("copy_diag", e)
            try:
                messagebox.showerror("診断情報", f"診断情報の作成に失敗しました:\n{e}")
            except Exception:
                pass

    def _show_help(self):
        # 連打で多重起動しないよう、既存ウィンドウを再利用
        try:
            if getattr(self, "_help_win", None) is not None and self._help_win.winfo_exists():
                try:
                    self._help_win.deiconify()
                except Exception:
                    pass
                try:
                    self._help_win.lift()
                    self._help_win.focus_force()
                except Exception:
                    pass
                return
        except Exception:
            pass

        win = tk.Toplevel(self.root)
        self._help_win = win
        win.title(f"使い方 - {APP_FULLNAME}")
        win.geometry("900x640")

        def _cleanup_help_ref():
            try:
                if getattr(self, "_help_win", None) is win:
                    self._help_win = None
            except Exception:
                pass

        def _on_close():
            _cleanup_help_ref()
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", _on_close)

        def _on_destroy(e):
            try:
                if e.widget is win:
                    _cleanup_help_ref()
            except Exception:
                pass

        win.bind("<Destroy>", _on_destroy)

        top = ttk.Frame(win)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text=f"{APP_FULLNAME} 使い方（取扱説明）", font=("", 12, "bold")).pack(side="left")

        def copy_all():
            txt = self._help_text()
            win.clipboard_clear()
            win.clipboard_append(txt)
            messagebox.showinfo("コピー", "取扱説明をクリップボードにコピーしました。")

        ttk.Button(top, text="全文コピー", command=copy_all).pack(side="right")

        body = scrolledtext.ScrolledText(win, wrap="word")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        body.insert("1.0", self._help_text())
        body.configure(state="disabled")

    # -------------------------
    # Run / Stop
    # -------------------------
    def _run(self):
        in_pdf = self.var_in.get().strip()
        if not in_pdf.lower().endswith(".pdf") or not os.path.isfile(in_pdf):
            messagebox.showwarning("確認", "入力PDFを選択してください。")
            return

        # Resolve output directory based on mode
        out_dir = ""
        try:
            mode = self._get_output_mode_code()
        except Exception:
            mode = "neighbor"
        if mode == "custom":
            out_dir = (self.var_out_custom.get() or "").strip()
        else:
            out_dir = self._resolve_output_dir(in_pdf)
        # Keep display in sync (and show fallback if any)
        try:
            self.var_out.set(out_dir)
        except Exception:
            pass
        if mode == "custom":
            if not out_dir:
                messagebox.showwarning("確認", "出力先が「指定フォルダ」のため、出力フォルダを選択してください。")
                return
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showwarning("確認", "出力フォルダを作成/参照できません。\n保存先の権限を確認してください。")
            return

        # SR / Safe mode
        safe_mode = False
        enable_sr = True
        try:
            safe_mode = bool(self.var_safe_mode.get()) if hasattr(self, "var_safe_mode") else False
        except Exception:
            safe_mode = False
        try:
            enable_sr = bool(self.var_enable_sr.get()) if hasattr(self, "var_enable_sr") else True
        except Exception:
            enable_sr = True
        if safe_mode:
            enable_sr = False

        if enable_sr:
            if not self._validate_model_path_or_guide():
                return

        # 読み方向（縦/横）: 必ず指定（auto廃止）
        rm = (self.var_reading_mode.get() or "").strip().lower()
        if rm not in ("vertical", "horizontal"):
            messagebox.showwarning("確認", "読み方向（縦書き / 横書き）を選択してください。")
            return

        # Gather settings
        model_path = self.var_model.get().strip()
        if not enable_sr:
            model_path = ""
        dpi = int(self.var_dpi.get())
        jpeg_q = int(self.var_jpeg.get())
        bg_scale = int(self.var_bg_scale.get()) if hasattr(self, 'var_bg_scale') else int(DEFAULT_BG_SCALE_PERCENT)
        max_out_dpi = int(self.var_max_out_dpi.get()) if hasattr(self, 'var_max_out_dpi') else int(DEFAULT_MAX_OUTPUT_DPI)
        bin_s = int(self.var_bin.get())
        bold_s = int(self.var_bold.get())
        tile = int(self.var_tile.get())
        page_mode = self.var_page_mode.get().strip()
        font_path = self.var_font.get().strip()

        ocr_workers = int(self.var_ocr_workers.get())
        if safe_mode:
            ocr_workers = 0
        enable_deskew = bool(self.var_deskew.get())
        deskew_max = float(self.var_deskew_max.get())

        # [Thread-safety] GUI操作(Tkinter)はメインスレッドのみ。
        #  - weights_dir 作成/ログ出力（_ensure_weights_dir → _log）もメインスレッドで実施
        #  - Tk変数(.get())もメインスレッドで読み取り、ワーカーには値を渡す
        weights_dir = self._ensure_weights_dir()
        _auto_grayscale = bool(self.var_gray_auto.get()) if hasattr(self, "var_gray_auto") else bool(DEFAULT_AUTO_GRAYSCALE)
        _gray_color_ratio_percent = float(self.var_gray_ratio.get()) if hasattr(self, "var_gray_ratio") else float(DEFAULT_GRAY_COLOR_RATIO_PERCENT)
        _gray_jpeg_quality_offset = int(self.var_gray_q_offset.get()) if hasattr(self, "var_gray_q_offset") else int(DEFAULT_GRAY_JPEG_QUALITY_OFFSET)

        self._save_config()

        self.stop_flag.clear()
        self._set_running(True)
        self.var_prog.set("開始中...")

        def worker():
            engine = None
            try:
                engine = PdfOcrEnhanceEngine(
                    log_cb=self._enqueue_log,
                    progress_cb=self._enqueue_progress,
                    stop_flag=self.stop_flag,
                    model_path=model_path,
                    base_dpi=dpi,
                    jpeg_quality=jpeg_q,
                    auto_grayscale=_auto_grayscale,
                    gray_color_ratio_percent=_gray_color_ratio_percent,
                    gray_jpeg_quality_offset=_gray_jpeg_quality_offset,
                    bg_scale_percent=bg_scale,
                    max_output_dpi=max_out_dpi,
                    binarize_strength=bin_s,
                    text_boldness=bold_s,
                    esrgan_tile=tile,
                    output_page_mode=page_mode,
                    font_path=font_path,
                    ocr_workers=ocr_workers,
                    enable_deskew=enable_deskew,
                    deskew_max_deg=deskew_max,
                    store_shrink=DEFAULT_STORE_SHRINK,
                    weights_dir=weights_dir,
                    enable_sr=enable_sr
                )
                # reading direction override from GUI
                try:
                    engine.reading_direction_mode = rm
                except Exception:
                    engine.reading_direction_mode = "vertical"

                # Cancel behavior (whether to write partial output on stop)
                try:
                    engine.keep_partial_on_cancel = bool(self.var_keep_partial_on_cancel.get()) if hasattr(self, "var_keep_partial_on_cancel") else True
                except Exception:
                    engine.keep_partial_on_cancel = True

                out_path = engine.process_pdf(in_pdf, out_dir)
                if self.stop_flag.is_set():
                    if out_path:
                        self.msg_queue.put(("cancelled", out_path))
                    else:
                        self.msg_queue.put(("cancelled_no_output", ""))
                else:
                    self.msg_queue.put(("done", out_path))
            except Exception:
                self.msg_queue.put(("error", traceback.format_exc()))
            finally:
                # Best-effort release of heavy GPU/CPU resources.
                # This helps reduce VRAM usage after cancel/exception while the app stays open.
                try:
                    if engine is not None and hasattr(engine, "shutdown"):
                        engine.shutdown(aggressive=True)
                except Exception:
                    pass
                try:
                    engine = None
                except Exception:
                    pass

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_flag.set()
        self._enqueue_log("[INFO] 中断要求を送信しました。安全なタイミングで停止します。")

    def _on_close(self):
        # --- stop UI dispatcher first (prevents after callbacks on destroyed root) ---
        try:
            if getattr(self, 'ui', None) is not None:
                try:
                    self.ui.stop()
                except Exception:
                    pass
            set_default_dispatcher(None)
        except Exception as e:
            _log_exception_once('ui_dispatch_stop', e)

        # 重要: 実行中のスレッド/プロセス/after を止めずに root.destroy() すると、
        # 破棄済みウィジェット更新で TclError を連発したり、OCRワーカーが残る可能性があります。
        if getattr(self, "_closing", False):
            return
        self._closing = True

        # まず停止シグナル
        try:
            self.stop_flag.set()
        except Exception as e:
            _log_exception_once('L_onclose_stopflag', e)

        # root.after のポーリングを停止
        try:
            jid = getattr(self, "_poll_job", None)
            if jid is not None:
                try:
                    safe_cancel(self.root, jid)
                except Exception:
                    pass
                self._poll_job = None
        except Exception as e:
            _log_exception_once('L_onclose_pollcancel', e)

        # その他の after ジョブを停止（存在すれば）
        for attr, code in [("_in_trace_job", "L_onclose_intrace"),
                           ("_render_job", "L_onclose_render"),
                           ("_bold_job", "L_onclose_bold")]:
            try:
                jid = getattr(self, attr, None)
                if jid is not None:
                    try:
                        safe_cancel(self.root, jid)
                    except Exception:
                        pass
                    try:
                        setattr(self, attr, None)
                    except Exception:
                        pass
            except Exception as e:
                _log_exception_once(code, e)


        # ワーカースレッドは daemon=True のため、joinでUIをブロックしません（停止シグナルは上で送信済み）

        # マルチプロセスOCRワーカーを残さない
        try:
            import multiprocessing as _mp
            children = _mp.active_children()
            for p in children:
                try:
                    p.terminate()
                except Exception:
                    pass
            for p in children:
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass
        except Exception as e:
            _log_exception_once('L_onclose_mp', e)

        # 設定保存（失敗しても終了は続ける）
        try:
            self._save_config()
        except Exception as e:
            _log_exception_once('L_onclose_save', e)

        # 最終安全装置: 何らかのスレッド/Queue終端待ちでプロセスが残ると、
        # コンソールが閉じず、GPU(VRAM)も解放されません。GUIを閉じた後に
        # 一定時間経ってもプロセスが終了していない場合は強制終了します。
        try:
            import threading as _threading
            import time as _time
            import os as _os
            def _hard_exit_guard(app_ref=self):
                """Final safety net: if worker threads / mp children remain, free resources and (only if needed) force-exit."""
                try:
                    # Give graceful shutdown a moment.
                    _time.sleep(0.35)
                    deadline = _time.time() + 10.0
                    while _time.time() < deadline:
                        alive_worker = False
                        alive_children = False
                        alive_non_daemon = False
                        # Worker thread
                        try:
                            th = getattr(app_ref, 'worker', None)
                            if th is not None and hasattr(th, 'is_alive') and th.is_alive():
                                alive_worker = True
                                # Request stop and join briefly
                                try:
                                    ev = getattr(app_ref, 'stop_flag', None)
                                    if ev is not None and hasattr(ev, 'set'):
                                        ev.set()
                                except Exception as e:
                                    _log_exception_once('L_exit_guard_stopflag', e)
                                try:
                                    th.join(timeout=0.15)
                                except Exception as e:
                                    _log_exception_once('L_exit_guard_join_worker', e)
                        except Exception as e:
                            _log_exception_once('L_exit_guard_worker_check', e)
            
                        # Multiprocessing children
                        try:
                            import multiprocessing as _mp  # local import (Windows safe)
                            if _mp.active_children():
                                alive_children = True
                        except Exception as e:
                            _log_exception_once('L_exit_guard_mp_check', e)
            
                        # Any non-daemon threads besides main (these keep the interpreter alive)
                        try:
                            for t in _threading.enumerate():
                                if t is _threading.current_thread() or t is _threading.main_thread():
                                    continue
                                if (not getattr(t, 'daemon', False)) and getattr(t, 'is_alive', lambda: False)():
                                    alive_non_daemon = True
                                    break
                        except Exception as e:
                            _log_exception_once('L_exit_guard_thread_enum', e)
            
                        if not (alive_worker or alive_children or alive_non_daemon):
                            # Best-effort CUDA cache release (if torch exists)
                            try:
                                import gc as _gc
                                _gc.collect()
                            except Exception:
                                pass
                            try:
                                import torch as _torch  # type: ignore
                                try:
                                    if _torch.cuda.is_available():
                                        _torch.cuda.empty_cache()
                                        try:
                                            _torch.cuda.ipc_collect()
                                        except Exception:
                                            pass
                                except Exception as e:
                                    _log_exception_once('L_exit_guard_cuda_empty', e)
                            except Exception:
                                pass
                            return
            
                        _time.sleep(0.20)
            
                    # Deadline reached: terminate any remaining mp children
                    try:
                        import multiprocessing as _mp
                        for p in list(_mp.active_children()):
                            try:
                                p.terminate()
                            except Exception as e:
                                _log_exception_once('L_exit_guard_term_child', e, context={'pid': getattr(p, 'pid', None)})
                        for p in list(_mp.active_children()):
                            try:
                                p.join(timeout=0.8)
                            except Exception as e:
                                _log_exception_once('L_exit_guard_join_child', e, context={'pid': getattr(p, 'pid', None)})
                    except Exception as e:
                        _log_exception_once('L_exit_guard_mp_terminate', e)
            
                    # Best-effort CUDA cache release (if torch exists)
                    try:
                        import gc as _gc
                        _gc.collect()
                    except Exception:
                        pass
                    try:
                        import torch as _torch  # type: ignore
                        try:
                            if _torch.cuda.is_available():
                                _torch.cuda.empty_cache()
                                try:
                                    _torch.cuda.ipc_collect()
                                except Exception:
                                    pass
                        except Exception as e:
                            _log_exception_once('L_exit_guard_cuda_empty2', e)
                    except Exception:
                        pass
            
                    # Force-exit only if something is still keeping the process alive.
                    should_force = False
                    try:
                        import multiprocessing as _mp
                        if _mp.active_children():
                            should_force = True
                    except Exception:
                        pass
                    try:
                        th = getattr(app_ref, 'worker', None)
                        if th is not None and hasattr(th, 'is_alive') and th.is_alive():
                            should_force = True
                    except Exception:
                        pass
                    try:
                        for t in _threading.enumerate():
                            if t is _threading.current_thread() or t is _threading.main_thread():
                                continue
                            if (not getattr(t, 'daemon', False)) and getattr(t, 'is_alive', lambda: False)():
                                should_force = True
                                break
                    except Exception:
                        pass
            
                    if should_force:
                        try:
                            _os._exit(0)
                        except Exception:
                            pass
                except Exception as e:
                    _log_exception_once('L_exit_guard_fatal', e)
                    try:
                        _os._exit(0)
                    except Exception:
                        pass
            _threading.Thread(target=_hard_exit_guard, daemon=True).start()
        except Exception:
            pass

        # 最後にウィンドウ破棄
        try:
            self.root.destroy()
        except Exception as e:
            _log_exception_once('L_onclose_destroy', e)

    def run(self):
        self.root.mainloop()


def main():
    # If launched without run_app.py, initialize logging here too.
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        portable_mode = os.path.isfile(os.path.join(base_dir, CONFIG_FILENAME))
        setup_app_logging(portable_mode=portable_mode, base_dir=base_dir, tee_stdio=True)
        try:
            import logging
            logging.getLogger("gui").info("GUI started")
        except Exception:
            pass
    except Exception:
        pass

    try:
        mp.freeze_support()  # for PyInstaller on Windows
    except Exception as e:
        _log_exception_once('L5715', e)
    app = AppGUI()
    app.run()


if __name__ == "__main__":
    main()
