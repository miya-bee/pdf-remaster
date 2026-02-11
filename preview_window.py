# -*- coding: utf-8 -*-
from __future__ import annotations

"""Preview window implementation extracted from gui.py.

This module contains the heavy preview-window logic so that gui.py can remain leaner.
It is intentionally UI-only and may call back into AppGUI methods via `self`.
"""

import os
import math
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox

# Optional deps (keep preview usable even if some deps are missing)
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

from constants import DEFAULT_BINARIZE_STRENGTH
from log_utils import _log_exception_once
from zoom_viewer import ZoomImageViewer, binarize_preview_rgb as _binarize_preview_rgb
from image_ops import estimate_residual_deskew_angle_preview
from image_ops import apply_text_boldness_to_rgb as _apply_text_boldness_to_rgb

from ui_dispatch import safe_after, safe_cancel


def show_preview_window(self, result: dict):
    """
    テスト結果プレビュー画面
    - 背景（回転→SR後） / OCR入力（回転→SR後→前処理） / OCR結果（テキスト/BOX）
    - ★二値化スライダーを追加：リアルタイムで二値化結果を更新して確認できる
      ※OCR結果（右）は、スライダー変更で自動更新しません。必要なら「このページだけ再OCR」で更新してください。
    """
    self._assert_ui('_show_preview_window')
    page_idx = int(result.get("page_index", 0))
    total = int(result.get("page_total", 0))
    page_no = page_idx + 1

    ang = float(result.get("deskew_angle", 0.0))
    tok = int(result.get("token_count", 0))
    dev = str(result.get("device", ""))

    win = tk.Toplevel(self.root)
    # --- Race-condition guard: closing the preview window while after/jobs are pending ---
    try:
        win._closing_preview = False
        win._jobs = []  # win.after job ids to cancel on close
    except Exception:
        pass
    # --- Thread-safe UI scheduling (preview window) ---
    # Tkinter is not thread-safe: worker threads must NOT call widget methods.
    # Use the global UI dispatcher (ui_dispatch.py) so dispatch is unified across the app.
    # NOTE: safe_after() routes scheduling to the UI thread. Return value is None when called from a worker.
    def _preview_alive() -> bool:
        try:
            if getattr(win, "_closing_preview", False):
                return False
            # Avoid calling Tk methods from worker threads.
            if not is_ui_thread():
                return True
            return bool(win.winfo_exists())
        except Exception:
            return False

    def _safe_after(ms: int, func):
        try:
            return safe_after(win, ms, func, track_list=getattr(win, "_jobs", None))
        except Exception as e:
            _log_exception_once('L_prev_after_wrap', e)
            return None



    # --- Cancellation flag for worker threads (stop when closing) ---
    try:
        win._stop_event = threading.Event()
    except Exception:
        pass
    def _stop_requested() -> bool:
        try:
            if getattr(win, "_closing_preview", False):
                return True
            ev = getattr(win, "_stop_event", None)
            return bool(ev is not None and ev.is_set())
        except Exception:
            return True

    pending = {"job": None}
    pending2 = {"job": None}

    def _close_preview(_evt=None):
        # Mark closing and cancel scheduled jobs to prevent TclError
        try:
            win._closing_preview = True
        except Exception:
            pass
        try:
            ev = getattr(win, "_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass
        # cancel debounced jobs (if already created)
        try:
            if pending.get("job") is not None:
                try:
                    safe_cancel(win, pending["job"])
                except Exception:
                    pass
                pending["job"] = None
        except Exception:
            pass
        try:
            if pending2.get("job") is not None:
                try:
                    safe_cancel(win, pending2["job"])
                except Exception:
                    pass
                pending2["job"] = None
        except Exception:
            pass
        # cancel all tracked after jobs
        try:
            for jid in list(getattr(win, "_jobs", [])):
                try:
                    safe_cancel(win, jid)
                except Exception:
                    pass
            try:
                win._jobs.clear()
            except Exception:
                pass
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass

    try:
        win.protocol("WM_DELETE_WINDOW", _close_preview)
    except Exception as e:
        _log_exception_once('L_prev_protocol', e)



    def _on_destroy(_e=None):
        try:
            win._closing_preview = True
        except Exception:
            pass
        try:
            ev = getattr(win, "_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass

    try:
        win.bind("<Destroy>", _on_destroy, add="+")
    except Exception:
        pass

    # 画面の裏に回るケース対策（Windows）
    try:
        win.lift()
        win.attributes('-topmost', True)
        _safe_after(200, lambda: win.attributes('-topmost', False))
        win.focus_force()
    except Exception as e:
        _log_exception_once('L4615', e)

    win.title(f"テスト結果 - ページ {page_no}/{total}  ({dev})")
    # 画面サイズ（内容が縦に長くなりやすいので、画面に応じて大きめに確保）
    try:
        sw = int(win.winfo_screenwidth())
        sh = int(win.winfo_screenheight())
        # 横幅が広すぎると扱いにくいので控えめに（大画面でも 1350px を上限）
        w0 = min(1350, max(980, int(sw * 0.78)))
        h0 = min(900,  max(650,  int(sh * 0.80)))
        win.geometry(f"{w0}x{h0}")
    except Exception:
        w0, h0 = 1300, 900
        try:
            win.geometry("1300x900")
        except Exception as e:
            _log_exception_once('L4630', e)

    # 3面プレビューの表示サイズ（下のボタンが隠れないよう、縦を控えめにする）
    try:
        panel_max_w = max(420, int((w0 - 90) / 3))
        panel_max_h = max(360, int(h0 - 360))
        panel_max_h = min(panel_max_h, 620)
    except Exception:
        panel_max_w, panel_max_h = 560, 600

    # 最大化できない原因になりやすい transient() は使用しない（所有ウィンドウ化で最大化ボタンが無効になることがある）
    win.resizable(True, True)
    try:
        win.minsize(980, 640)
    except Exception as e:
        _log_exception_once('L4645', e)

    def _toggle_maximize(_evt=None):
        try:
            # Windows: "zoomed" が最大化
            if str(win.state()) == "zoomed":
                win.state("normal")
            else:
                win.state("zoomed")
        except Exception as e:
            _log_exception_once('L4655', e)

    # F11で最大化/復元
    try:
        win.bind("<F11>", _toggle_maximize)
    except Exception as e:
        _log_exception_once('L4661', e)

    # 参照保持（PhotoImage）
    win._photo_refs = {}
    win._zoom_viewers = {}  # panel_id -> ZoomImageViewer
    panels = {}  # panel_id -> panel辞書（プレビュー3枚管理）

    top = ttk.Frame(win)
    top.pack(fill="x", padx=10, pady=10)

    # Title line (dynamic: angle/token updates)
    var_title = tk.StringVar(value="")

    def _set_title(tokens: int = tok, applied_angle: float = None):
        try:
            a = float(ang)
        except Exception:
            a = 0.0
        try:
            if applied_angle is None:
                applied_angle = a
            var_title.set(
                f"ページ {page_no}/{total}  |  角度: {float(applied_angle):+.2f}°（自動 {a:+.2f}°）  |  tokens: {int(tokens)}"
            )
        except Exception:
            try:
                var_title.set(f"ページ {page_no}/{total}  |  tokens: {int(tokens)}")
            except Exception:
                pass

    _set_title(tokens=tok, applied_angle=ang)
    lbl_title = ttk.Label(top, textvariable=var_title, font=("", 12, "bold"))
    lbl_title.pack(side="left")
    ttk.Button(top, text="最大化/復元 (F11)", command=_toggle_maximize).pack(side="right")

    body = ttk.Frame(win)
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # images (numpy RGB)
    sr_rgb = result.get("sr_rgb")
    ocr_rgb = result.get("ocr_rgb")
    ov_rgb = result.get("overlay_rgb")

    # OCR前処理スライダーの入力画像（本番OCRと同じ入力に合わせる）
    sr_rgb_for_ocr = result.get("sr_rgb_for_ocr")
    if sr_rgb_for_ocr is None:
        sr_rgb_for_ocr = sr_rgb

    # -------------------------
    # 角度微調整（自動推定 + オフセット）
    # -------------------------
    # ここでは「3面プレビューは維持」しつつ、角度だけを微調整できるようにします。
    # ※Deskewの推定角(ang)はテスト実行時の値。オフセットで追加回転をかけます。
    state_angle = {
        "auto": float(ang) if ang is not None else 0.0,
        "off": 0.0,
        "bg_base": sr_rgb,
        "ocr_base": sr_rgb_for_ocr,
        "overlay_base": ov_rgb,
    }
    cur_imgs = {
        "bg": sr_rgb,
        "ocr_base": sr_rgb_for_ocr,
        "overlay": ov_rgb,
    }
    tok_state = {"count": int(tok)}
    var_ocr_status = tk.StringVar(value="OCR結果: テスト実行時")
    var_angle_off = tk.DoubleVar(value=0.0)
    var_angle_off_str = tk.StringVar(value="+0.00°")
    var_angle_applied_str = tk.StringVar(value=f"{state_angle['auto']:+.2f}°")

    pending_ang = {"job": None}

    def _rotate_rgb_keep(rgb_img, angle_deg: float):
        if not CV2_AVAILABLE or cv2 is None or rgb_img is None:
            return rgb_img
        try:
            h, w = rgb_img.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
            rotated = cv2.warpAffine(
                rgb_img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            return rotated
        except Exception:
            return rgb_img

    def _recompute_cur_images():
        off = float(state_angle.get("off", 0.0))
        cur_imgs["bg"] = _rotate_rgb_keep(state_angle["bg_base"], off)
        cur_imgs["ocr_base"] = _rotate_rgb_keep(state_angle["ocr_base"], off)
        cur_imgs["overlay"] = _rotate_rgb_keep(state_angle["overlay_base"], off)
        # zoom viewer に渡す参照も更新
        try:
            p = panels.get(PID_OCR)
            if p is not None:
                p["bin_src_rgb"] = cur_imgs["ocr_base"]
                p["bin_zero_rgb"] = cur_imgs["bg"]
        except Exception:
            pass

    def _update_overlay_panel(rgb_img):
        try:
            p = panels.get(PID_OV)
            if not p:
                return
            if not p["lbl"].winfo_exists():
                return
            photo, (nw, nh) = self._np_to_photo(rgb_img, max_w=int(p["max_w"]), max_h=int(p["max_h"]))
            win._photo_refs[p.get("id", p.get("title"))] = photo
            p["lbl"].configure(image=photo)
            p["lbl_dim"].configure(text=f"{nw}x{nh}")
            p["arr"] = rgb_img
            p["arr_full"] = rgb_img
        except Exception:
            pass

    def _sync_open_zoom_viewers_angle():
        """角度オフセット変更時に、開いている拡大ビューアへ最新画像を流し込む。"""
        try:
            zv = getattr(win, "_zoom_viewers", None)
        except Exception:
            zv = None
        if not zv:
            return
        # cur_imgs は _recompute_cur_images() 後の最新を想定
        for k, viewer in list(zv.items()):
            try:
                if viewer is None or (not bool(viewer.winfo_exists())):
                    zv.pop(k, None)
                    continue
            except Exception:
                zv.pop(k, None)
                continue
            try:
                kind = str(getattr(viewer, "_panel_kind", "bg"))
            except Exception:
                kind = "bg"
            try:
                if kind == "ocr":
                    # ソースだけ更新し、Zoom側の現在スライダー値で再構築
                    viewer.replace_sources_from_rgb(
                        base_rgb=None,
                        bin_src_rgb=cur_imgs.get("ocr_base"),
                        bin_zero_rgb=cur_imgs.get("bg"),
                        keep_fit_width=False,
                        suppress_callbacks=True,
                    )
                elif kind == "overlay":
                    viewer.replace_sources_from_rgb(
                        base_rgb=cur_imgs.get("overlay"),
                        bin_src_rgb=None,
                        bin_zero_rgb=None,
                        keep_fit_width=False,
                        suppress_callbacks=True,
                    )
                else:
                    # 背景（SR）
                    viewer.replace_sources_from_rgb(
                        base_rgb=cur_imgs.get("bg"),
                        bin_src_rgb=None,
                        bin_zero_rgb=None,
                        keep_fit_width=False,
                        suppress_callbacks=True,
                    )
            except Exception:
                continue

    def _apply_angle_now(off: float):
        try:
            off = float(off)
        except Exception:
            off = 0.0
        # clamp
        if off < -3.0:
            off = -3.0
        if off > 3.0:
            off = 3.0

        state_angle["off"] = off
        var_angle_off_str.set(f"{off:+.2f}°")
        var_angle_applied_str.set(f"{(state_angle['auto'] + off):+.2f}°")

        # 画像更新
        _recompute_cur_images()

        # OCR結果は古くなる（再OCRで更新）
        try:
            var_ocr_status.set("OCR結果: 旧（角度/前処理変更後）→ 再OCRで更新")
        except Exception:
            pass

        # タイトル更新
        try:
            _set_title(tokens=tok_state.get("count", 0), applied_angle=state_angle["auto"] + off)
        except Exception:
            pass

        # 右パネルは画像自体を回転して追随（OCRの再実行はボタンで）
        _update_overlay_panel(cur_imgs["overlay"])

        # 左/中央は、現在スライダー値で再描画
        try:
            _apply_boldness(int(var_preview_bold.get() or 0))
        except Exception:
            pass
        try:
            _apply_strength(int(var_preview_bin.get() or 0))
        except Exception:
            pass

        # 角度補正を、開いている拡大ビューアにも反映
        try:
            _sync_open_zoom_viewers_angle()
        except Exception:
            pass

    def _on_angle_scale(_v=None):
        # デバウンスして更新
        try:
            safe_cancel(pending_ang.get("job"))
        except Exception:
            pass

        def _do():
            try:
                _apply_angle_now(var_angle_off.get())
            except Exception:
                pass

        pending_ang["job"] = _safe_after(80, _do)

    # ここでいう「残差」は、テスト実行時の deskew（自動推定）後にまだ残る微小な傾き。
    # プレビューは微調整用途のため、軽い推定（低解像度 + エッジ + Hough）で安全に行います。
    var_angle_resid = tk.StringVar(value="残差: --")

    # 自動補正の信頼度（低いときは自動適用しない）
    state_angle["resid_ok"] = False
    state_angle["resid_method"] = ""
    state_angle["resid_mad"] = 0.0
    state_angle["resid_count"] = 0
    state_angle["resid_sumlen"] = 0.0

    def _estimate_residual_angle(rgb_img) -> float:
        """現在の画像に対して、追加で必要な回転角（deskew角）を推定する。

        重要: プレビューの自動補正は「誤補正しない」ことを最優先にします。
        実装は image_ops.estimate_residual_deskew_angle_preview に集約し、
        GUI側は結果を表示するだけにします。
        """
        # reset (UI state)
        state_angle["resid_ok"] = False
        state_angle["resid_method"] = ""
        state_angle["resid_mad"] = 0.0
        state_angle["resid_count"] = 0
        state_angle["resid_sumlen"] = 0.0
        try:
            delta, info = estimate_residual_deskew_angle_preview(rgb_img, max_abs_deg=3.0)
            try:
                state_angle["resid_method"] = str(info.get("method", ""))
                state_angle["resid_mad"] = float(info.get("mad", 0.0))
                state_angle["resid_count"] = int(info.get("count", 0))
                state_angle["resid_sumlen"] = float(info.get("sumlen", 0.0))
            except Exception:
                pass
            ok = bool(info.get("ok", False)) and abs(float(delta)) > 1e-6
            state_angle["resid_ok"] = bool(ok)
            if not ok:
                return 0.0
            return float(delta)
        except Exception:
            state_angle["resid_ok"] = False
            return 0.0

    def _auto_correct_angle(force: bool = False):
        """残差を推定してオフセットへ反映。

        - force=False: 信頼度が低い場合は“適用しない”（誤補正防止）
        - force=True : ボタン操作など、ユーザーの意図が明確な場合は適用
        """
        if not _preview_alive():
            return
        try:
            delta = float(_estimate_residual_angle(cur_imgs.get("ocr_base")))
        except Exception:
            delta = 0.0

        try:
            ok = bool(state_angle.get("resid_ok", False))
            method = str(state_angle.get("resid_method", ""))
            mad = float(state_angle.get("resid_mad", 0.0))
            cnt = int(state_angle.get("resid_count", 0))
            sl = float(state_angle.get("resid_sumlen", 0.0))
            tag = ("OK" if ok else "低信頼")
            var_angle_resid.set(f"残差推定: {delta:+.3f}° [{tag}] {method} n={cnt} L={sl:.0f} mad={mad:.2f}")
        except Exception:
            pass

        if abs(delta) < 1e-6:
            return
        if not force:
            # 自動実行（起動時など）は「誤補正しない」を最優先
            if not bool(state_angle.get("resid_ok", False)):
                return
            # まっすぐページを微小角で回してしまうケースがあるため、
            # 自動適用は十分に大きい残差のみ（ボタンは常に適用）
            if abs(delta) < 0.45:
                return

        try:
            new_off = float(state_angle.get("off", 0.0)) + float(delta)
        except Exception:
            new_off = float(delta)
        new_off = float(max(-3.0, min(3.0, new_off)))
        try:
            var_angle_off.set(new_off)
        except Exception:
            pass
        _apply_angle_now(new_off)

    def _auto_correct_angle_force():
        _auto_correct_angle(force=True)

    # angle UI
    angle_ctrl = ttk.LabelFrame(body, text="角度微調整（自動推定 + オフセット）")
    angle_ctrl.pack(fill="x", pady=(0, 8))
    ttk.Label(angle_ctrl, text=f"自動推定: {state_angle['auto']:+.2f}°").pack(side="left", padx=(8, 10))
    ttk.Label(angle_ctrl, textvariable=var_angle_resid, foreground="#555").pack(side="left", padx=(0, 10))
    ttk.Label(angle_ctrl, text="オフセット").pack(side="left")

    # ttk.Scale は幅が狭いと刻みが粗く感じやすいので、resolution を持つ tk.Scale を使い、
    # さらに横幅に合わせて伸びるようにする（微調整しやすくする）。
    sc_ang = tk.Scale(
        angle_ctrl,
        from_=-3.0,
        to=3.0,
        orient="horizontal",
        variable=var_angle_off,
        resolution=0.01,
        showvalue=False,
        command=_on_angle_scale,
    )
    sc_ang.pack(side="left", fill="x", expand=True, padx=(8, 6))

    ttk.Label(angle_ctrl, textvariable=var_angle_off_str, width=8).pack(side="left")
    ttk.Label(angle_ctrl, text="適用角").pack(side="left", padx=(10, 4))
    ttk.Label(angle_ctrl, textvariable=var_angle_applied_str, width=9).pack(side="left")

    def _nudge(d: float):
        try:
            v = float(var_angle_off.get()) + float(d)
        except Exception:
            v = float(d)
        v = float(max(-3.0, min(3.0, v)))
        try:
            var_angle_off.set(v)
        except Exception:
            pass
        _on_angle_scale()

    # 微調整ボタン（スライダーが粗く感じる場合の補助）
    ttk.Button(angle_ctrl, text="-0.05", width=6, command=lambda: _nudge(-0.05)).pack(side="left", padx=(8, 2))
    ttk.Button(angle_ctrl, text="+0.05", width=6, command=lambda: _nudge(+0.05)).pack(side="left", padx=(0, 6))
    ttk.Button(angle_ctrl, text="-0.01", width=6, command=lambda: _nudge(-0.01)).pack(side="left", padx=(0, 2))
    ttk.Button(angle_ctrl, text="+0.01", width=6, command=lambda: _nudge(+0.01)).pack(side="left", padx=(0, 6))

    ttk.Button(angle_ctrl, text="0に戻す", command=lambda: (var_angle_off.set(0.0), _on_angle_scale())).pack(side="left", padx=(0, 6))
    ttk.Button(angle_ctrl, text="自動補正", command=_auto_correct_angle_force).pack(side="left", padx=(0, 8))
    ttk.Label(angle_ctrl, textvariable=var_ocr_status, foreground="#555").pack(side="left", padx=8)

    if not CV2_AVAILABLE:
        try:
            sc_ang.configure(state="disabled")
            var_ocr_status.set("OCR結果: OpenCVが無いため角度微調整は無効")
        except Exception:
            pass


    # -------------------------
    # 二値化スライダー（リアルタイム）
    # -------------------------
    ctrl = ttk.LabelFrame(body, text="二値化（リアルタイム確認）")
    ctrl.pack(fill="x", pady=(0, 8))

    # 初期値：メイン設定の二値化強度を採用（テスト時と一致させる）
    try:
        init_strength = int(self.var_bin.get())
    except Exception:
        init_strength = DEFAULT_BINARIZE_STRENGTH
    init_strength = max(0, min(100, init_strength))

    var_preview_bin = tk.IntVar(value=init_strength)

    ttk.Label(ctrl, text="二値化強度:", width=10).pack(side="left", padx=(8, 2))
    lbl_val = ttk.Label(ctrl, textvariable=var_preview_bin, width=4)
    lbl_val.pack(side="left")


    def _preprocess_preview(rgb_img: np.ndarray, strength: int) -> np.ndarray:
        """二値化画像（RGB）を作る（OpenCVなしでも動作するプレビュー用）"""
        try:
            return _binarize_preview_rgb(rgb_img, strength)
        except Exception as e:
            _log_exception_once('L_prev_bin', e)
            return rgb_img


    def _get_zoom_rgb(pan: dict):
        """拡大表示で使う「現在の」画像を返す（角度オフセットを反映）。

        - 背景: cur_imgs['bg']（角度反映済み）を渡し、Zoom側の太字スライダーで見た目を一致させる
        - OCR入力: 現在の二値化強度で生成（角度反映済み）
        - OCR結果: cur_imgs['overlay']（角度反映済み）
        """
        try:
            kind = str(pan.get("kind", "bg"))
        except Exception:
            kind = "bg"

        # 背景（左）
        if kind == "bg":
            return cur_imgs.get("bg") if cur_imgs.get("bg") is not None else pan.get("arr_full", pan.get("arr"))

        # OCR入力（二値化・中央）
        if kind == "ocr":
            try:
                strength = int(var_preview_bin.get() or 0)
            except Exception:
                strength = 0
            strength = max(0, min(100, strength))

            base = cur_imgs.get("ocr_base") if cur_imgs.get("ocr_base") is not None else pan.get("bin_src_rgb")
            if strength <= 0:
                zero = cur_imgs.get("bg") if cur_imgs.get("bg") is not None else pan.get("bin_zero_rgb")
                return zero if zero is not None else (base if base is not None else pan.get("arr_full", pan.get("arr")))

            if base is None:
                return pan.get("arr_full", pan.get("arr"))
            return _preprocess_preview(base, strength)

        # OCR結果（右）
        if kind == "overlay":
            return cur_imgs.get("overlay") if cur_imgs.get("overlay") is not None else pan.get("arr_full", pan.get("arr"))

        return pan.get("arr_full", pan.get("arr"))


    def add_panel(parent, panel_id: str, title: str, arr: np.ndarray, kind: str = "bg", max_w: int = None, max_h: int = None, get_bold_fn=None, sync_fn=None, get_bin_fn=None, sync_bin_fn=None, bin_src_rgb=None, bin_zero_rgb=None):
        p = ttk.LabelFrame(parent, text=title)
        p.pack(side="left", fill="both", expand=True, padx=6)

        if arr is None:
            ttk.Label(p, text="(画像なし)").pack()
            return None

        # max_w/max_h が指定されていない場合は、ウィンドウに合わせた共通サイズを使う

        if max_w is None:

            try:

                max_w = int(panel_max_w)

            except Exception:

                max_w = 560

        if max_h is None:

            try:

                max_h = int(panel_max_h)

            except Exception:

                max_h = 600
        photo, (nw, nh) = self._np_to_photo(arr, max_w=max_w, max_h=max_h)
        win._photo_refs[panel_id] = photo

        lbl = ttk.Label(p, image=photo)
        lbl.pack(padx=6, pady=(6, 2))

        lbl_dim = ttk.Label(p, text=f"{arr.shape[1]}x{arr.shape[0]} px → 表示 {nw}x{nh}px")
        lbl_dim.pack(pady=(0, 4))

        btnrow = ttk.Frame(p)
        btnrow.pack(fill="x", padx=6, pady=(0, 6))
        # arrは可変の可能性があるため、panel dict から参照する
        panel = {
            "id": panel_id,
            "title": title,
            "kind": kind,
            "arr": arr,
            "arr_full": arr,
            "lbl": lbl,
            "lbl_dim": lbl_dim,
            "max_w": max_w,
            "max_h": max_h,
            "bin_src_rgb": bin_src_rgb,
            "bin_zero_rgb": bin_zero_rgb,
        }
        def _open_zoom_and_track(pan=panel):
            viewer = self._open_zoom_viewer(
                _get_zoom_rgb(pan),
                f"拡大プレビュー: {pan['title']}",
                panel_kind=str(pan.get("kind", "bg")),
                override_bold=(int(get_bold_fn()) if get_bold_fn else None),
                sync_callback=sync_fn,
                override_bin=(int(get_bin_fn()) if get_bin_fn else None),
                bin_src_rgb=pan.get("bin_src_rgb"),
                bin_zero_rgb=pan.get("bin_zero_rgb"),
                sync_bin_callback=sync_bin_fn,
                owner_window=win,
            )
            try:
                if viewer is None:
                    return None
                zv = getattr(win, "_zoom_viewers", None)
                if zv is None:
                    win._zoom_viewers = {}
                    zv = win._zoom_viewers
                zv[pan.get("id", pan.get("title"))] = viewer
                def _cleanup(_evt=None, key=pan.get("id", pan.get("title"))):
                    try:
                        zv2 = getattr(win, "_zoom_viewers", None)
                        if zv2 is not None:
                            zv2.pop(key, None)
                    except Exception:
                        pass
                try:
                    viewer.bind("<Destroy>", _cleanup)
                except Exception:
                    pass
            except Exception:
                pass
            return viewer

        ttk.Button(
            btnrow,
            text="拡大表示",
            command=_open_zoom_and_track,
        ).pack(side="left")
        ttk.Label(btnrow, text="Ctrl+ホイールで拡大/縮小・ドラッグで移動").pack(side="left", padx=8)

        panels[panel_id] = panel
        return panel

    cols = ttk.Frame(body)
    cols.pack(fill="both", expand=True)

    # プレビュー用: 文字太さ（デフォルト0）。拡大表示へもこの値を引き継ぐ
    var_preview_bold = tk.IntVar(value=0)
    sc2 = None  # [Guard] assigned later; protect early callbacks
    sc_bin_preview = None  # [Guard] assigned later; protect early callbacks
    suppress_bin = {"flag": False}

    # Panel titles (keep the 3-pane layout, clarify pipeline stages)
    P_BG = "背景（回転→SR後）"
    P_OCR = "OCR入力（回転→SR後→前処理）"
    P_OV = "OCR結果（テキスト/BOX）"
    # Panel ids (internal keys; titles may change in future)
    PID_BG = "bg"
    PID_OCR = "ocr"
    PID_OV = "overlay"


    # ★追加: 拡大画面から呼ばれる同期用関数
    def _sync_bold_from_zoom(v):
        try:
            v = int(v)
        except Exception:
            return
        v = max(-100, min(100, v))
        try:
            var_preview_bold.set(v)
        except Exception as e:
            _log_exception_once('L4808', e)
        try:
            lbl_bold_val.config(text=f"{v:+4d}")
        except Exception as e:
            _log_exception_once('L4812', e)
        if sc2 is not None:
            try:
                sc2.set(v)
            except Exception as e:
                _log_exception_once('L4816', e)
        try:
            _apply_boldness(v)
        except Exception as e:
            _log_exception_once('L4820', e)


    # ★追加: 拡大（二値化）画面から呼ばれる同期用関数
    def _sync_bin_from_zoom(v):
        try:
            v = int(v)
        except Exception:
            return
        v = max(0, min(100, v))
        try:
            var_preview_bin.set(v)
        except Exception as e:
            _log_exception_once('L_bin_sync1', e)

        # プレビュー側スライダーも合わせる（コマンド再入を抑制）
        if sc_bin_preview is not None:
            try:
                suppress_bin["flag"] = True
                sc_bin_preview.set(v)
            except Exception as e:
                _log_exception_once('L_bin_sync2', e)
            finally:
                suppress_bin["flag"] = False

        # 中央（二値化）パネルを更新
        try:
            _apply_strength(v)
        except Exception as e:
            _log_exception_once('L_bin_sync3', e)



    add_panel(cols, PID_BG, P_BG, sr_rgb, kind="bg", max_w=int(panel_max_w), max_h=int(panel_max_h), get_bold_fn=var_preview_bold.get, sync_fn=_sync_bold_from_zoom)
    add_panel(cols, PID_OCR, P_OCR, ocr_rgb, kind="ocr", max_w=int(panel_max_w), max_h=int(panel_max_h), get_bin_fn=var_preview_bin.get, sync_bin_fn=_sync_bin_from_zoom, bin_src_rgb=cur_imgs.get("ocr_base"), bin_zero_rgb=cur_imgs.get("bg"))
    add_panel(cols, PID_OV, P_OV, ov_rgb, kind="overlay", max_w=int(panel_max_w), max_h=int(panel_max_h))

    # プレビュー表示直後に一度だけ「残差自動補正」を実行（必要な場合のみ）
    # ※角度UI/パネル構築後に呼ぶことで、内部更新（_apply_boldness/_apply_strength）が安全に動きます。
    if CV2_AVAILABLE:
        try:
            _safe_after(150, _auto_correct_angle)
        except Exception:
            pass

    # OCR前処理パネルがある場合だけスライダーを有効化
    def _apply_strength(strength: int):
        if not _preview_alive():
            return
        if cur_imgs.get("ocr_base") is None:
            return
        p = panels.get(PID_OCR)
        if not p:
            return
        try:
            if int(strength) <= 0:
                new_ocr = (cur_imgs.get("bg") if cur_imgs.get("bg") is not None else cur_imgs.get("ocr_base"))
            else:
                new_ocr = _preprocess_preview(cur_imgs.get("ocr_base"), strength)
        except Exception as e:
            # 失敗時は何もしない（ログだけ）
            try:
                self._log(f"[WARN] プレビュー二値化の更新に失敗: {e}")
            except Exception as e:
                _log_exception_once('L4841', e)
            return

        # UI更新（PhotoImage差し替え）
        if not _preview_alive():
            return
        try:
            if not p.get('lbl') or not p['lbl'].winfo_exists():
                return
        except Exception:
            return
        photo, (nw, nh) = self._np_to_photo(new_ocr, max_w=p.get('max_w', int(panel_max_w)), max_h=p.get('max_h', int(panel_max_h)))
        win._photo_refs[p.get("id", p.get("title"))] = photo
        p["lbl"].configure(image=photo)
        p["arr"] = new_ocr
        p["arr_full"] = new_ocr
        p["lbl_dim"].configure(text=f"{new_ocr.shape[1]}x{new_ocr.shape[0]} px → 表示 {nw}x{nh}px")

    def _on_scale(v):
        if suppress_bin.get("flag"):
            return
        if not _preview_alive():
            return
        # ttk.Scale は文字列で来る
        try:
            strength = int(float(v))
        except Exception:
            strength = int(var_preview_bin.get() or DEFAULT_BINARIZE_STRENGTH)
        strength = max(0, min(100, strength))
        var_preview_bin.set(strength)
        try:
            var_ocr_status.set("OCR結果: 旧（角度/前処理変更後）→ 再OCRで更新")
        except Exception:
            pass

        # 連続ドラッグ時に重くならないようデバウンス
        if pending["job"] is not None:
            try:
                safe_cancel(win, pending["job"])
            except Exception as e:
                _log_exception_once('L4865', e)
            pending["job"] = None
        pending["job"] = _safe_after(80, lambda s=strength: _apply_strength(s))

    sc = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal", command=_on_scale)
    sc_bin_preview = sc
    sc.set(init_strength)
    sc.pack(side="left", fill="x", expand=True, padx=8)

    def _apply_to_main():
        """プレビューで気に入った値をメイン設定へ反映（次回以降にも有効）"""
        try:
            v = int(var_preview_bin.get())
            v = max(0, min(100, v))
            self.var_bin.set(v)
            try:
                self.sc_bin.set(v)
            except Exception as e:
                _log_exception_once('L4882', e)
            self._save_config()
            self._log(f"[INFO] 二値化強度を反映: {v}")
        except Exception as e:
            _log_exception_once('L4886', e)

    ttk.Button(ctrl, text="この値をメイン設定へ反映", command=_apply_to_main).pack(side="left", padx=(0, 8))

    # --- このページだけ再OCR（右パネルのBOXを更新） ---
    def _rerun_ocr_this_page():
        if not _preview_alive():
            return
        if not CV2_AVAILABLE:
            try:
                messagebox.showwarning("再OCR", "OpenCVが利用できないため再OCRできません。")
            except Exception:
                pass
            return

        try:
            strength = int(var_preview_bin.get() or 0)
        except Exception:
            strength = 0
        strength = max(0, min(100, strength))

        # UI: disable while running
        try:
            btn_reocr.config(state="disabled")
        except Exception:
            pass
        try:
            var_ocr_status.set("OCR結果: 再OCR中…")
        except Exception:
            pass

        def _worker():
            if _stop_requested():
                return
            try:
                with self._test_engine_lock:
                    eng = self._test_engine
                if eng is None:
                    raise RuntimeError("テストエンジンが未初期化です。先に『このページをテスト』を実行してください。")

                # preprocess (real pipeline) with temporary binarize strength
                old_bs = getattr(eng, "binarize_strength", None)
                try:
                    eng.binarize_strength = int(strength)
                    ocr_bgr2 = eng._preprocess_for_ocr(cur_imgs.get("ocr_base"))
                finally:
                    try:
                        if old_bs is not None:
                            eng.binarize_strength = old_bs
                    except Exception:
                        pass

                if _stop_requested():
                    return

                tokens = eng._run_ocr_singleproc(ocr_bgr2)

                if _stop_requested():
                    return

                overlay_rgb2 = cur_imgs.get("bg").copy() if cur_imgs.get("bg") is not None else None
                if overlay_rgb2 is not None and cv2 is not None:
                    for t in tokens:
                        try:
                            r = t.rect
                            x0, y0, x1, y1 = int(r.x0), int(r.y0), int(r.x1), int(r.y1)
                            cv2.rectangle(overlay_rgb2, (x0, y0), (x1, y1), (0, 0, 255), 2)
                        except Exception:
                            continue

            except Exception as err:
                def _ui_err():
                    if not _preview_alive():
                        return
                    try:
                        messagebox.showerror("再OCRエラー", str(err))
                    except Exception:
                        pass
                    try:
                        btn_reocr.config(state="normal")
                    except Exception:
                        pass
                    try:
                        var_ocr_status.set("OCR結果: 失敗（再OCRエラー）")
                    except Exception:
                        pass

                _safe_after(0, _ui_err)
                return

            def _ui_ok():
                if not _preview_alive():
                    return
                try:
                    tok_state["count"] = int(len(tokens))
                except Exception:
                    pass
                try:
                    _set_title(tokens=tok_state.get("count", 0), applied_angle=state_angle["auto"] + float(state_angle.get("off", 0.0)))
                except Exception:
                    pass
                if overlay_rgb2 is not None:
                    try:
                        # overlay_baseを更新（角度変更後も追随できるように 0°基準へ戻して保存）
                        state_angle["overlay_base"] = _rotate_rgb_keep(overlay_rgb2, -float(state_angle.get("off", 0.0)))
                        cur_imgs["overlay"] = overlay_rgb2
                    except Exception:
                        pass
                    _update_overlay_panel(overlay_rgb2)
                try:
                    var_ocr_status.set("OCR結果: 最新（再OCR済み）")
                except Exception:
                    pass
                try:
                    btn_reocr.config(state="normal")
                except Exception:
                    pass

            _safe_after(0, _ui_ok)

        threading.Thread(target=_worker, daemon=True).start()

    btn_reocr = ttk.Button(ctrl, text="このページだけ再OCR（BOX更新）", command=_rerun_ocr_this_page)
    btn_reocr.pack(side="left", padx=(0, 8))
    ttk.Label(ctrl, text="※スライダー変更後は「このページだけ再OCR」で右パネル（BOX）を更新できます。").pack(side="left", padx=6)

    # -------------------------
    # 太字化スライダー（リアルタイム）
    # -------------------------
    ctrl2 = ttk.LabelFrame(body, text="文字太さ（閲覧用・リアルタイム確認）")
    ctrl2.pack(fill="x", pady=(0, 8))

    try:
        init_bold = int(self.var_bold.get())
    except Exception:
        init_bold = 0
    init_bold = max(-100, min(100, init_bold))

    ttk.Label(ctrl2, text="文字太さ:", width=10).pack(side="left", padx=(8, 2))
    lbl_bold_val = ttk.Label(ctrl2, text=f"{init_bold:+4d}", width=5)
    lbl_bold_val.pack(side="left")

    sr_base = sr_rgb  # 元のSR画像（累積適用を避ける）

    # ★表示用の縮小画像を先に作成（縮小表示でも太字化の差を見える化）
    def _resize_for_panel(rgb_img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        try:
            if rgb_img is None:
                return None
            h0, w0 = rgb_img.shape[:2]
            scale = min(max_w / float(w0), max_h / float(h0), 1.0)
            if scale >= 1.0:
                return rgb_img.copy()
            nw, nh = int(w0 * scale), int(h0 * scale)
            return cv2.resize(rgb_img, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            return rgb_img

    sr_disp_base = _resize_for_panel(sr_base, int(panel_max_w), int(panel_max_h))


    def _bold_preview(rgb_img: np.ndarray, strength: int) -> np.ndarray:
        """閲覧用の文字太さ調整（プレビュー版）。

        重要: 実出力と同じ処理を使う（image_ops.apply_text_boldness_to_rgb）。
        これにより「プレビューと実出力で太さがズレる」問題を抑制します。
        """
        try:
            if _apply_text_boldness_to_rgb is None:
                return rgb_img
            return _apply_text_boldness_to_rgb(rgb_img, int(strength))
        except Exception:
            return rgb_img

    # --------------------------------------------------
    # 太字化プレビュー（A方式）
    #   重要: フル解像度で太字化 → 表示用に縮小
    #   ZoomImageViewer と同じ順序で処理するため、見た目の差が最小になります。
    # --------------------------------------------------
    bold_state = {"seq": 0}

    def _apply_boldness(strength: int):
        if not _preview_alive():
            return
        """背景(SR)パネルの表示を更新（A方式: フル解像度→表示用縮小）。"""
        if cur_imgs.get("bg") is None:
            return
        p = panels.get(PID_BG)
        if not p:
            return
        try:
            strength = int(strength)
        except Exception:
            strength = 0
        strength = max(-100, min(100, strength))

        # 最新リクエストだけ反映（古い計算結果は捨てる）
        bold_state["seq"] += 1
        my_seq = bold_state["seq"]

        def _worker():
            if _stop_requested():
                return
            try:
                if strength == 0:
                    out_full = cur_imgs.get("bg")
                else:
                    # ★フル解像度で太字化（ここがA方式の要点）
                    out_full = _bold_preview(cur_imgs.get("bg"), strength)
            except Exception as e:
                out_full = cur_imgs.get("bg")
                try:
                    self._enqueue_log(f"[WARN] プレビュー太字化の更新に失敗: {e}")
                except Exception as e:
                    _log_exception_once('L5009', e)

            if _stop_requested():
                return

            def _ui():
                if not _preview_alive():
                    return
                try:
                    if not p.get('lbl') or not p['lbl'].winfo_exists():
                        return
                except Exception:
                    return
                # 途中でスライダーが動いていたら結果を捨てる
                if my_seq != bold_state["seq"]:
                    return
                try:
                    photo, (nw, nh) = self._np_to_photo(
                        out_full,
                        max_w=p.get("max_w", 560),
                        max_h=p.get("max_h", 700),
                    )
                    win._photo_refs[p.get("id", p.get("title"))] = photo
                    p["lbl"].configure(image=photo)
                    # p["arr"] は巨大配列を保持しない（メモリ節約）
                    try:
                        p["lbl_dim"].configure(text=f"{out_full.shape[1]}x{out_full.shape[0]} px → 表示 {nw}x{nh}px")
                    except Exception as e:
                        _log_exception_once('L5027', e)
                except Exception as e2:
                    try:
                        self._log(f"[WARN] 画像更新に失敗: {e2}")
                    except Exception as e:
                        _log_exception_once('L5032', e)

            try:
                _safe_after(0, _ui)
            except Exception as e:
                _log_exception_once('L5037', e)

        threading.Thread(target=_worker, daemon=True).start()


    def _on_bold_scale(v):
        if not _preview_alive():
            return
        try:
            strength = int(float(v))
        except Exception:
            try:
                strength = int(var_preview_bold.get() or 0)
            except Exception:
                strength = 0
        strength = max(-100, min(100, strength))
        var_preview_bold.set(strength)
        lbl_bold_val.config(text=f"{strength:+4d}")

        if pending2["job"] is not None:
            try:
                safe_cancel(win, pending2["job"])
            except Exception as e:
                _log_exception_once('L5059', e)
            pending2["job"] = None
        pending2["job"] = _safe_after(120, lambda s=strength: _apply_boldness(s))

    sc2 = ttk.Scale(ctrl2, from_=-100, to=100, orient="horizontal", command=_on_bold_scale)
    sc2.set(init_bold)
    sc2.pack(side="left", fill="x", expand=True, padx=8)

    def _apply_bold_to_main():
        """プレビューで気に入った太字化をメイン設定へ反映"""
        try:
            v = int(var_preview_bold.get() or 0)
            v = max(-100, min(100, v))
            self.var_bold.set(v)
            try:
                self.sc_bold.set(v)
            except Exception as e:
                _log_exception_once('L5076', e)
            self._save_config()
            self._log(f"[INFO] 文字太さ（閲覧）を反映: {v}")
        except Exception as e:
            _log_exception_once('L5080', e)

    ttk.Button(ctrl2, text="この値をメイン設定へ反映", command=_apply_bold_to_main).pack(side="left", padx=(0, 8))
    ttk.Label(ctrl2, text="※背景画像のみ更新（OCR前処理/BOXはテスト実行時の結果）").pack(side="left", padx=6)

    # 初期状態でも、念のため同期
    try:
        _apply_boldness(init_bold)
    except Exception as e:
        _log_exception_once('L5095', e)

    # 初期状態でも、念のため同期
    try:
        _apply_strength(init_strength)
    except Exception as e:
        _log_exception_once('L5101', e)

    # Tips
    tips = (
        "調整のコツ:\n"
        "  - 二値化強度を上げる: 背景が消えるが文字が欠けやすい\n"
        "  - DeskewをON: 斜めスキャンのOCR精度が上がりやすい（max角度は5deg前後推奨）\n"
        "  - DPIを上げる: 精細になるが処理/メモリが重くなる\n"
        "  - BBoxの追従も見たい場合: このウィンドウを閉じて、二値化強度を反映→再度「このページをテスト」を実行\n"
    )
    box = ttk.LabelFrame(body, text="ヒント")
    box.pack(fill="x", pady=(8, 0))
    ttk.Label(box, text=tips, justify="left").pack(anchor="w", padx=8, pady=6)