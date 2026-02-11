# -*- coding: utf-8 -*-
from __future__ import annotations

"""Zoom viewer (separated from gui.py).

This module contains:
- binarize_preview_rgb(): a lightweight, stable binarization preview for UI tuning
- ZoomImageViewer: the zoom/inspect window used by the GUI

Keeping this in a dedicated module reduces gui.py size and lowers the risk of
accidental UI/threading regressions when editing other parts of the app.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk

# Optional deps (viewer can exist even if these are missing; features will degrade gracefully)
try:
    from PIL import Image, ImageTk, ImageFilter  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
    ImageFilter = None  # type: ignore

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

# Shared ops (keep import here for readability; failures fall back to PIL path)
try:
    from image_ops import apply_text_boldness_to_rgb as _apply_text_boldness_to_rgb
except Exception:
    _apply_text_boldness_to_rgb = None  # type: ignore

from log_utils import _log_exception_once
from image_ops import binarize_preview_rgb as _binarize_preview_rgb_impl

# -------------------------
# Preview binarization helper (strength 0-100)
# -------------------------
def binarize_preview_rgb(rgb_img, strength: int):
    """UI preview helper wrapper.

    Implementation lives in image_ops.binarize_preview_rgb so preview behavior
    is defined in a single place.
    """
    try:
        return _binarize_preview_rgb_impl(rgb_img, strength)
    except Exception as e:
        _log_exception_once('zoom_viewer.binarize_preview_rgb', e, context={'strength': strength})
        return rgb_img


class ZoomImageViewer(tk.Toplevel):
    """画像を拡大・縮小しながら詳細確認できるビューア（Canvas + スクロール）。
    - Ctrl+ホイール: ズーム
    - ホイール: 縦スクロール / Shift+ホイール: 横スクロール
    - ドラッグ: パン（掴んで移動）
    - （任意）文字太さ（擬似ボールド）スライダー: プレビュー用に背景画像を太らせて確認
    """

    @staticmethod
    def _fmt_signed(v: int) -> str:
        """Return signed integer string: '0' or '+25' / '-25'."""
        try:
            vv = int(v)
        except Exception:
            vv = 0
        return "0" if vv == 0 else f"{vv:+d}"

    def __init__(
        self,
        parent,
        pil_image: Image.Image,
        title: str = "プレビュー（拡大）",
        panel_kind: str = "bg",
        enable_bold_slider: bool = False,
        bold_init: int = 0,
        on_apply_bold=None,
        enable_bin_slider: bool = False,
        bin_init: int = 0,
        bin_src_rgb=None,
        bin_zero_rgb=None,
        sync_bin_callback=None,
    ):
        super().__init__(parent)
        # Guard: ignore slider callbacks during widget construction
        self._init_guard = True
        self._first_render = True

        # Panel kind (bg / ocr / overlay) for external sync logic
        try:
            self._panel_kind = str(panel_kind) if (panel_kind is not None) else "bg"
        except Exception:
            self._panel_kind = "bg"

        self._did_initial_fit = False
        self.title(title)

        # デフォルトは「全画面に近い」サイズを避け、横幅が広すぎない範囲で見やすい大きさにする
        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            # 横幅が広すぎると扱いにくいので控えめに（大画面でも 1050px を上限）
            w0 = min(1050, max(760, int(sw * 0.58)))
            h0 = min(900,  max(650, int(sh * 0.75)))
            self.geometry(f"{w0}x{h0}")
        except Exception:
            # 例外時も横幅が広すぎない値にする
            self.geometry("980x740")


        # -------------------------
        # 二値化（OCR前処理）スライダー（任意）
        # -------------------------
        self._enable_bin_slider = bool(enable_bin_slider and (bin_src_rgb is not None) and NUMPY_AVAILABLE)
        try:
            self._bin_strength = int(max(0, min(100, int(bin_init))))
        except Exception:
            self._bin_strength = 0
        self._bin_job = None
        self._bin_src_rgb = bin_src_rgb
        self._bin_zero_rgb = bin_zero_rgb
        self._sync_bin_callback = sync_bin_callback

        # 二値化スライダーが有効なら、初期画像を strength に合わせて作る
        if self._enable_bin_slider:
            try:
                src0 = self._bin_zero_rgb if (int(self._bin_strength) <= 0 and self._bin_zero_rgb is not None) else self._bin_src_rgb
                arr0 = binarize_preview_rgb(src0, self._bin_strength)
                pil_image = Image.fromarray(arr0.astype("uint8"), mode="RGB")
            except Exception as e:
                _log_exception_once('L_bin_init', e)
                self._enable_bin_slider = False

        # ベース画像（RGB）
        self._orig_base = pil_image.convert("RGB")
        self._base = self._orig_base  # 現在表示用（太字反映後など）

        # Large images can exceed Tk PhotoImage limits at 100% before the first "fit" render.
        # Start a bit smaller, then we will "fit width" once the canvas size is ready.
        try:
            _max_side = max(self._orig_base.size)
            self._initial_zoom_percent = 10 if int(_max_side) >= 5000 else 100
        except Exception:
            self._initial_zoom_percent = 100
        self._photo = None
        self._img_id = None

        # ズーム
        self._zoom = 1.0
        self._render_job = None
        self._hq_render_job = None
        self._fast_render = False

        # 太字（任意）
        self._enable_bold_slider = bool(enable_bold_slider)
        self._bold_strength = int(max(-100, min(100, int(bold_init)))) if self._enable_bold_slider else 0
        self._on_apply_bold = on_apply_bold
        self._bold_job = None


        # 太字（閲覧）スライダーの画像生成は重いので、別スレッドで処理してGUIフリーズを避ける
        self._bold_req_seq = 0
        self._bold_pending = False
        self._bold_pending_seq = -1
        self._bold_req_q = None
        self._bold_res_q = None
        self._bold_worker_stop = None
        self._bold_worker_thread = None
        self._bold_poll_job = None
        if self._enable_bold_slider:
            try:
                self._bold_req_q = queue.Queue(maxsize=1)  # 最新リクエストのみ保持
                self._bold_res_q = queue.Queue()           # 完了結果
                self._bold_worker_stop = threading.Event()
                self._bold_worker_thread = threading.Thread(target=self._bold_worker_loop, daemon=True)
                self._bold_worker_thread.start()
                # 結果ポーリング（tkは別スレッドから触らない）
                self._bold_poll_job = self.after(60, self._poll_bold_results)
            except Exception as e:
                _log_exception_once('L3360b', e)
                self._bold_req_q = None


        # 二値化（OCR前処理）スライダーの画像生成も重いので、別スレッドで処理してGUIフリーズを避ける
        self._bin_req_seq = 0
        self._bin_pending = False
        self._bin_pending_seq = -1
        self._bin_req_q = None
        self._bin_res_q = None
        self._bin_worker_stop = None
        self._bin_worker_thread = None
        self._bin_poll_job = None
        if self._enable_bin_slider:
            try:
                self._bin_req_q = queue.Queue(maxsize=1)  # 最新リクエストのみ保持
                self._bin_res_q = queue.Queue()           # 完了結果
                self._bin_worker_stop = threading.Event()
                self._bin_worker_thread = threading.Thread(target=self._bin_worker_loop, daemon=True)
                self._bin_worker_thread.start()
                # 結果ポーリング（tkは別スレッドから触らない）
                self._bin_poll_job = self.after(60, self._poll_bin_results)
            except Exception as e:
                _log_exception_once('bin_worker_init', e)
                self._bin_req_q = None
                self._bin_res_q = None
                self._bin_worker_stop = None
                self._bin_worker_thread = None
                self._bin_poll_job = None
        # -------------------------
        # ツールバー（ズーム）
        # -------------------------
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(6, 4))

        ttk.Label(bar, text="ズーム:").pack(side="left")
        self.sld_zoom = ttk.Scale(bar, from_=10, to=400, orient="horizontal", command=self._on_zoom_slider)
        self.sld_zoom.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_zoom = ttk.Label(bar, text="100%")
        self.lbl_zoom.pack(side="left", padx=(0, 8))
        # 初期値設定時に command が発火すると lbl_zoom 等が未作成のタイミングがあり得るため、後から command を設定する
        try:
            self.sld_zoom.set(float(getattr(self, "_initial_zoom_percent", 100)))
        except Exception:
            pass

        ttk.Button(bar, text="－", width=4, command=lambda: self._step_zoom(-10)).pack(side="left")
        ttk.Button(bar, text="＋", width=4, command=lambda: self._step_zoom(+10)).pack(side="left", padx=(4, 8))
        ttk.Button(bar, text="100%", command=lambda: self._set_zoom(1.0)).pack(side="left")
        ttk.Button(bar, text="幅に合わせる", command=self._fit_width).pack(side="left", padx=6)
        ttk.Button(bar, text="ウィンドウに合わせる", command=self._fit_window).pack(side="left")


        # -------------------------
        # ツールバー（二値化）
        # -------------------------
        if self._enable_bin_slider:
            bar_bin = ttk.Frame(self)
            # 要望：ズームスライダーの下に配置
            bar_bin.pack(fill="x", padx=8, pady=(0, 6))
            ttk.Label(bar_bin, text="二値化:").pack(side="left")
            self.sld_bin = ttk.Scale(bar_bin, from_=0, to=100, orient="horizontal", command=self._on_bin_slider)
            self.sld_bin.pack(side="left", fill="x", expand=True, padx=8)
            self.lbl_bin = ttk.Label(bar_bin, text=str(self._bin_strength), width=4)
            self.lbl_bin.pack(side="left")
            # 初期値設定時の command 発火を避ける（UI生成順の都合）
            try:
                self.sld_bin.set(self._bin_strength)
            except Exception:
                pass

            ttk.Button(bar_bin, text="0", width=4, command=lambda: self._set_bin(0)).pack(side="left", padx=(10, 2))
            ttk.Button(bar_bin, text="50", width=4, command=lambda: self._set_bin(50)).pack(side="left", padx=2)
            ttk.Button(bar_bin, text="100", width=4, command=lambda: self._set_bin(100)).pack(side="left", padx=2)

        # -------------------------
        # ツールバー（太字）
        # -------------------------
        if self._enable_bold_slider:
            bar2 = ttk.Frame(self)
            bar2.pack(fill="x", padx=8, pady=(0, 6))
            ttk.Label(bar2, text="文字太さ（閲覧）:").pack(side="left")
            self.sld_bold = ttk.Scale(bar2, from_=-100, to=100, orient="horizontal", command=self._on_bold_slider)
            self.sld_bold.pack(side="left", fill="x", expand=True, padx=8)
            self.lbl_bold = ttk.Label(bar2, text=self._fmt_signed(self._bold_strength))
            self.lbl_bold.pack(side="left")
            # 初期値設定時の command 発火を避ける（UI生成順の都合）
            try:
                self.sld_bold.set(self._bold_strength)
            except Exception:
                pass

            ttk.Button(bar2, text=self._fmt_signed(0), width=4, command=lambda: self._set_bold(0)).pack(side="left", padx=(10, 2))
            ttk.Button(bar2, text=self._fmt_signed(25), width=4, command=lambda: self._set_bold(25)).pack(side="left", padx=2)
            ttk.Button(bar2, text=self._fmt_signed(50), width=4, command=lambda: self._set_bold(50)).pack(side="left", padx=2)
            ttk.Button(bar2, text=self._fmt_signed(75), width=4, command=lambda: self._set_bold(75)).pack(side="left", padx=2)
            ttk.Button(bar2, text=self._fmt_signed(100), width=4, command=lambda: self._set_bold(100)).pack(side="left", padx=2)

        # -------------------------
        # Canvas + Scrollbar
        # -------------------------
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(frm, bg="#222222", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.vbar = ttk.Scrollbar(frm, orient="vertical", command=self.canvas.yview)
        self.vbar.pack(side="right", fill="y")
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.hbar.pack(side="bottom", fill="x")

        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        # パン操作
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)

        # マウスホイール（スクロール/ズーム）
        self._bind_mousewheel()

        # リサイズ時にフィット操作が安定するよう更新（レンダリングはしない）
        self.canvas.bind("<Configure>", lambda e: None)

        self._init_guard = False

        # 初期：太字適用があれば反映 → 描画
        if self._enable_bold_slider and callable(self._on_apply_bold):
            self._apply_bold_now(self._bold_strength)

        # 画面サイズが確定するのを少し待ってから「幅に合わせる」を実行
        def _startup_fit(attempt: int = 0):
            # Wait until the canvas gets a real size, then fit+render.
            try:
                if not bool(self.winfo_exists()):
                    return
            except Exception:
                return
            try:
                self.update_idletasks()
                cw = int(self.canvas.winfo_width())
                ch = int(self.canvas.winfo_height())
            except Exception:
                cw, ch = 0, 0

            if cw < 50 or ch < 50:
                if attempt < 60:
                    try:
                        self.after(25, lambda: _startup_fit(attempt + 1))
                    except Exception:
                        pass
                else:
                    try:
                        self._fit_width()
                    except Exception:
                        try:
                            self._schedule_render(fast=False, delay_ms=1)
                            self._schedule_hq_render()
                        except Exception:
                            pass
                return

            try:
                self._fit_width()
            except Exception:
                try:
                    self._schedule_render(fast=False, delay_ms=1)
                    self._schedule_hq_render()
                except Exception:
                    pass

        try:
            self.after_idle(lambda: _startup_fit(0))
        except Exception:
            self.after(150, lambda: _startup_fit(0))
    # -------------------------
    # Input bindings
    # -------------------------
    def _bind_mousewheel(self):
        """マウスホイールをこのウィンドウ単位でバインド（bind_allは使わない）"""
        try:
            self.bind("<MouseWheel>", self._on_mousewheel)
            self.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
            self.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        except Exception as e:
            _log_exception_once('L3372', e)

    def destroy(self):
        """閉じる（太字ワーカー等を停止）"""
        try:
            # 太字処理結果ポーリング停止
            if getattr(self, "_bold_poll_job", None) is not None:
                try:
                    self.after_cancel(self._bold_poll_job)
                except Exception as e:
                    _log_exception_once('L3380b', e)
                self._bold_poll_job = None

            # 太字ワーカー停止
            try:
                if getattr(self, "_bold_worker_stop", None) is not None:
                    self._bold_worker_stop.set()
                if getattr(self, "_bold_req_q", None) is not None:
                    try:
                        # ブロック解除用のダミー投入（満杯なら捨てて投入）
                        self._bold_req_q.put_nowait((-1, 0))
                    except Exception:
                        try:
                            while True:
                                self._bold_req_q.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._bold_req_q.put_nowait((-1, 0))
                        except Exception:
                            pass
                th = getattr(self, "_bold_worker_thread", None)
                if th is not None and th.is_alive():
                    try:
                        th.join(timeout=0.2)
                    except Exception as e:
                        _log_exception_once('L3380c', e)
            except Exception as e:
                _log_exception_once('L3380d', e)

            # 二値化処理結果ポーリング停止
            if getattr(self, "_bin_poll_job", None) is not None:
                try:
                    self.after_cancel(self._bin_poll_job)
                except Exception as e:
                    _log_exception_once('bin_poll_cancel', e)
                self._bin_poll_job = None

            # 二値化ワーカー停止
            try:
                if getattr(self, "_bin_worker_stop", None) is not None:
                    self._bin_worker_stop.set()
                if getattr(self, "_bin_req_q", None) is not None:
                    try:
                        self._bin_req_q.put_nowait((-1, 0, None, None))
                    except Exception:
                        try:
                            while True:
                                self._bin_req_q.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._bin_req_q.put_nowait((-1, 0, None, None))
                        except Exception:
                            pass
                th2 = getattr(self, "_bin_worker_thread", None)
                if th2 is not None and th2.is_alive():
                    try:
                        th2.join(timeout=0.2)
                    except Exception as e:
                        _log_exception_once('bin_worker_join', e)
            except Exception as e:
                _log_exception_once('bin_worker_stop', e)

        except Exception as e:
            _log_exception_once('L3380e', e)
        super().destroy()

    def _on_mousewheel(self, e):
        # Windowsは e.delta (120単位)が多い
        delta = -1 * (e.delta // 120) if getattr(e, "delta", 0) else 0
        if delta == 0:
            return
        self.canvas.yview_scroll(delta, "units")

    def _on_shift_mousewheel(self, e):
        delta = -1 * (e.delta // 120) if getattr(e, "delta", 0) else 0
        if delta == 0:
            return
        self.canvas.xview_scroll(delta, "units")

    def _on_ctrl_mousewheel(self, e):
        delta = 1 if e.delta > 0 else -1
        self._step_zoom(10 * delta)

    def _on_pan_start(self, e):
        self.canvas.scan_mark(e.x, e.y)

    def _on_pan_move(self, e):
        self.canvas.scan_dragto(e.x, e.y, gain=1)

    # -------------------------
    # Zoom control
    # -------------------------
    def _on_zoom_slider(self, _):

        if getattr(self, "_init_guard", False):
            return
        if getattr(self, "_suppress_zoom_cmd", False):
            return


        try:
            v = float(self.sld_zoom.get())
        except Exception:
            v = float(getattr(self, "_zoom", 1.0)) * 100.0
        v = max(10.0, min(400.0, v))
        try:
            self.lbl_zoom.configure(text=f"{int(v)}%")
        except Exception as e:
            _log_exception_once('L3419', e)
        self._zoom = float(v) / 100.0
        self._schedule_render(fast=True)
        self._schedule_hq_render()

    def _step_zoom(self, step_pct: int):

        try:
            cur = float(self.sld_zoom.get())
        except Exception:
            cur = float(self._zoom) * 100.0
        nxt = max(10.0, min(400.0, cur + float(step_pct)))
        self.sld_zoom.set(nxt)
        # tkinterでは .set() だけでは command が発火しないため明示的に更新
        self._on_zoom_slider(None)


    def _set_zoom(self, z: float):
        """Set zoom to given scale (1.0=100%). Used by preset buttons.

        ttk.Scale.set() does not reliably fire the slider command across platforms,
        so we explicitly route through _on_zoom_slider() to update the view.
        """
        try:
            z = float(z)
        except Exception:
            z = 1.0
        z = float(max(0.10, min(4.00, z)))
        pct = z * 100.0
        try:
            self._suppress_zoom_cmd = True
            try:
                self.sld_zoom.set(pct)
            except Exception as e:
                _log_exception_once('L3446', e)
        finally:
            try:
                self._suppress_zoom_cmd = False
            except Exception:
                pass
        # Ensure identical update path as slider drag
        try:
            self._on_zoom_slider(None)
        except Exception as e:
            _log_exception_once('L3450', e)


    def _initial_fit_width_then_render(self):
        """拡大プレビューの初期表示を『幅に合わせる』にする。"""
        try:
            self.update_idletasks()
        except Exception as e:
            _log_exception_once('L3458', e)
        try:
            self._did_initial_fit = True
            self._fit_width()   # スライダー経由でズームを更新
        except Exception:
            # 失敗しても描画は継続
            self._did_initial_fit = True
        # 念のため描画要求（_fit_widthが効かない環境でも表示されるように）
        try:
            self._schedule_render(fast=True)
            self._schedule_hq_render()
        except Exception as e:
            _log_exception_once('L3469', e)

    def _fit_width(self):

        try:
            # 画面サイズが確定していないケースに備えて update_idletasks
            self.update_idletasks()
            cw = max(1, int(self.canvas.winfo_width()))
            w = int(self._base.size[0])
            z = cw / float(max(1, w))
            self.sld_zoom.set(max(10.0, min(400.0, z * 100.0)))
            self._on_zoom_slider(None)
        except Exception:
            try:
                self._on_zoom_slider(None)
            except Exception as e:
                _log_exception_once('L3485', e)

    def _fit_window(self):

        try:
            self.update_idletasks()
            cw = max(1, int(self.canvas.winfo_width()))
            ch = max(1, int(self.canvas.winfo_height()))
            w, h = self._base.size
            z = min(cw / float(max(1, w)), ch / float(max(1, h)))
            self.sld_zoom.set(max(10.0, min(400.0, z * 100.0)))
            self._on_zoom_slider(None)
        except Exception:
            try:
                self._on_zoom_slider(None)
            except Exception as e:
                _log_exception_once('L3501', e)


    def _on_bin_slider(self, _):
        if getattr(self, "_init_guard", False):
            return
        if getattr(self, "_suppress_bin_cmd", False):
            return
        if not getattr(self, "_enable_bin_slider", False):
            return
        try:
            v = int(float(self.sld_bin.get()))
        except Exception:
            v = int(getattr(self, "_bin_strength", 50))
        v = max(0, min(100, v))
        self._bin_strength = v
        try:
            self.lbl_bin.configure(text=str(v))
        except Exception as e:
            _log_exception_once('L3499b', e)

        # 連続ドラッグでも重くならないようデバウンス
        if getattr(self, "_bin_job", None) is not None:
            try:
                self.after_cancel(self._bin_job)
            except Exception as e:
                _log_exception_once('L3500b', e)
        self._bin_job = self.after(80, lambda: self._apply_bin_now(v))


    def _set_bin(self, v: int):
        """Preset buttons for binarization.

        Make the UI respond immediately: cancel any pending debounce job and
        apply the change right away (slider.set alone won't update the image).
        """
        if not getattr(self, "_enable_bin_slider", False):
            return
        try:
            v = int(v)
        except Exception:
            v = 50
        v = int(max(0, min(100, v)))
        self._bin_strength = v
        # Cancel pending debounce job (if any)
        if getattr(self, "_bin_job", None) is not None:
            try:
                self.after_cancel(self._bin_job)
            except Exception:
                pass
            self._bin_job = None
        # Update slider without re-entering the command handler
        try:
            self._suppress_bin_cmd = True
            try:
                self.sld_bin.set(v)
            except Exception:
                pass
        finally:
            try:
                self._suppress_bin_cmd = False
            except Exception:
                pass
        try:
            self.lbl_bin.configure(text=str(v))
        except Exception:
            pass
        # Apply now (no extra debounce)
        try:
            self._apply_bin_now(v)
        except Exception:
            # fallback
            try:
                self._on_bin_slider(None)
            except Exception:
                pass




    def _apply_bin_now(self, v: int):
        """二値化スライダーの値を、拡大プレビューへリアルタイム反映する（重処理は別スレッド）"""
        self._bin_job = None
        if not getattr(self, "_enable_bin_slider", False):
            return
        try:
            v = int(max(0, min(100, int(v))))
        except Exception:
            v = 50

        # store
        try:
            self._bin_strength = int(v)
        except Exception:
            pass

        src = getattr(self, "_bin_src_rgb", None)
        src_zero = getattr(self, "_bin_zero_rgb", None)
        if src is None and src_zero is None:
            return

        # 0) コールバック（同期通知）は即時（描画が遅れても値は同期させる）
        cb = getattr(self, "_sync_bin_callback", None)
        if cb:
            try:
                cb(int(v))
            except Exception as e:
                _log_exception_once('bin_sync_cb', e)

        # 1) リクエスト番号更新（取り違え防止）
        if getattr(self, "_bin_req_q", None) is not None and getattr(self, "_bin_worker_thread", None) is not None:
            try:
                self._bin_req_seq = int(getattr(self, "_bin_req_seq", 0)) + 1
            except Exception:
                self._bin_req_seq = 1
            seq = int(self._bin_req_seq)
        else:
            seq = None

        # 2) 即時応答：0なら即戻す（ワーカー不要）
        if v == 0:
            try:
                use_src = src_zero if (src_zero is not None) else src
                if use_src is not None:
                    pil = Image.fromarray(use_src.astype("uint8"), mode="RGB")
                    self._orig_base = pil.convert("RGB")
                    self._base = self._orig_base
                else:
                    self._base = self._orig_base
            except Exception:
                try:
                    self._base = self._orig_base
                except Exception:
                    pass
            # fit width and redraw
            try:
                self._fit_width()
            except Exception:
                try:
                    self._schedule_render(fast=True)
                    self._schedule_hq_render()
                except Exception:
                    pass
            return

        # 3) 非0はワーカーへ（ワーカーが無い場合は従来同期でフォールバック）
        if seq is not None:
            try:
                self._bin_pending = True
                self._bin_pending_seq = int(seq)
            except Exception:
                pass

            try:
                self._enqueue_bin_request(seq, v, src, src_zero)
            except Exception as e:
                _log_exception_once('bin_enqueue', e)

            # show current base immediately (avoid "no response" feel)
            try:
                self._schedule_render(fast=True)
            except Exception:
                pass

            # Kick polling quickly so preset buttons feel instant
            try:
                self._kick_bin_poll(soon=True)
            except Exception:
                pass
            return

        # --- Fallback (synchronous) ---
        try:
            use_src = (src_zero if (int(v) <= 0 and src_zero is not None) else src)
            arr = binarize_preview_rgb(use_src, v)
            pil = Image.fromarray(arr.astype("uint8"), mode="RGB")
            self._orig_base = pil.convert("RGB")
            self._base = self._orig_base
        except Exception as e:
            _log_exception_once('bin_fallback', e)
            return

        # If bold is enabled and currently non-zero, re-apply it on top of the new binarized base.
        try:
            if getattr(self, "_enable_bold_slider", False) and int(getattr(self, "_bold_strength", 0)) != 0:
                self._apply_bold_now(int(getattr(self, "_bold_strength", 0)), notify=False)
        except Exception as e:
            _log_exception_once('bin_then_bold', e)

        # enforce fit width
        try:
            self._fit_width()
        except Exception:
            try:
                self._schedule_render()
                self._schedule_hq_render()
            except Exception:
                pass

    def _on_bold_slider(self, _):
        if getattr(self, "_init_guard", False):
            return
        if getattr(self, "_suppress_bold_cmd", False):
            return
        if not self._enable_bold_slider:
            return
        v = int(float(self.sld_bold.get()))
        v = max(-100, min(100, v))
        self._bold_strength = v
        try:
            self.lbl_bold.configure(text=self._fmt_signed(v))
        except Exception as e:
            _log_exception_once('L3512', e)
        # 連続ドラッグでも重くならないようデバウンス
        if self._bold_job is not None:
            try:
                self.after_cancel(self._bold_job)
            except Exception as e:
                _log_exception_once('L3518', e)
        self._bold_job = self.after(140, lambda: self._apply_bold_now(v))


    def _set_bold(self, v: int):
        """Preset buttons for boldness (viewer only).

        Preset buttons should feel instant: cancel any pending debounce job and
        enqueue/apply immediately.
        """
        if not getattr(self, "_enable_bold_slider", False):
            return
        try:
            v = int(v)
        except Exception:
            v = 0
        v = int(max(-100, min(100, v)))
        self._bold_strength = v
        # Cancel pending debounce job (if any)
        if getattr(self, "_bold_job", None) is not None:
            try:
                self.after_cancel(self._bold_job)
            except Exception:
                pass
            self._bold_job = None
        # Update slider without re-entering the command handler
        try:
            self._suppress_bold_cmd = True
            try:
                self.sld_bold.set(v)
            except Exception:
                pass
        finally:
            try:
                self._suppress_bold_cmd = False
            except Exception:
                pass
        try:
            self.lbl_bold.configure(text=self._fmt_signed(v))
        except Exception:
            pass
        # Apply now (heavy work will still be in worker thread)
        try:
            self._apply_bold_now(v)
        except Exception:
            try:
                self._on_bold_slider(None)
            except Exception:
                pass


    def _apply_bold_preview_pil(self, pil_img, strength: int):
        """閲覧用の文字太さ調整（拡大ビューア内）を適用した PIL 画像を返す。

        strength: -100..+100
          0   : 無効（原画像）
          +値 : 太く（ストロークを膨張させ、追加領域を少し暗くする）
          -値 : 細く（ストロークを収縮させ、削れた領域を背景に近づける）
        """
        if not PIL_AVAILABLE or pil_img is None:
            return pil_img

        try:
            s = int(float(strength))
        except Exception:
            s = 0
        s = int(max(-100, min(100, s)))
        if s == 0:
            try:
                return pil_img.convert("RGB")
            except Exception:
                return pil_img

        # --- Prefer OpenCV path (match FINAL output exactly) ---
        if CV2_AVAILABLE and NUMPY_AVAILABLE and _apply_text_boldness_to_rgb is not None:
            try:
                img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                out = _apply_text_boldness_to_rgb(img, int(s))
                return Image.fromarray(out, mode="RGB")
            except Exception:
                pass

        # --- PIL fallback (approximate) ---
        try:
            img = pil_img.convert("RGB")
            strength_abs = abs(int(s))
            k = 3 if strength_abs < 60 else 5
            alpha = 0.20 + (float(strength_abs) / 100.0) * 0.55  # 0.20..0.75

            if s > 0:
                # 太く（暗い方へ寄せる）
                out = img.filter(ImageFilter.MinFilter(k))
            else:
                # 細く（明るい方へ寄せる）
                out = img.filter(ImageFilter.MaxFilter(k))
            return Image.blend(img, out, alpha)
        except Exception:
            return pil_img

    def _enqueue_bold_request(self, seq: int, strength: int) -> None:
        """太字処理要求をワーカーへ送る（常に最新のみ保持）"""
        q = getattr(self, "_bold_req_q", None)
        if q is None:
            return
        try:
            q.put_nowait((seq, strength))
            return
        except Exception:
            # 満杯なら古い要求を捨てて最新だけ入れる
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait((seq, strength))
            except Exception:
                pass



    def _enqueue_bin_request(self, seq: int, strength: int, src, src_zero) -> None:
        """二値化処理要求をワーカーへ送る（常に最新のみ保持）"""
        q = getattr(self, "_bin_req_q", None)
        if q is None:
            return
        try:
            q.put_nowait((seq, strength, src, src_zero))
            return
        except Exception:
            # 満杯なら古い要求を捨てて最新だけ入れる
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait((seq, strength, src, src_zero))
            except Exception:
                pass

    def _bin_worker_loop(self):
        """二値化の重処理はここで実行（GUIスレッドをブロックしない）"""
        stop = getattr(self, "_bin_worker_stop", None)
        req_q = getattr(self, "_bin_req_q", None)
        res_q = getattr(self, "_bin_res_q", None)
        if stop is None or req_q is None or res_q is None:
            return

        while not stop.is_set():
            try:
                seq, strength, src, src_zero = req_q.get(timeout=0.2)
            except Exception:
                continue

            if stop.is_set():
                break

            # 連続要求は最後の1つだけ処理（中間は捨てる）
            try:
                while True:
                    seq2, strength2, src2, src_zero2 = req_q.get_nowait()
                    seq, strength, src, src_zero = seq2, strength2, src2, src_zero2
            except Exception:
                pass

            if stop.is_set():
                break

            try:
                strength = int(max(0, min(100, int(strength))))
            except Exception:
                strength = 0

            try:
                use_src = src_zero if (int(strength) <= 0 and src_zero is not None) else src
            except Exception:
                use_src = src

            out_img = None
            try:
                if use_src is None:
                    out_img = None
                else:
                    arr = binarize_preview_rgb(use_src, int(strength))
                    out_img = Image.fromarray(arr.astype("uint8"), mode="RGB")
            except Exception as e:
                _log_exception_once('bin_worker_apply', e)
                out_img = None

            try:
                res_q.put_nowait((seq, strength, out_img))
            except Exception:
                # キューが詰まっても最新優先
                try:
                    while True:
                        res_q.get_nowait()
                except Exception:
                    pass
                try:
                    res_q.put_nowait((seq, strength, out_img))
                except Exception:
                    pass

    def _kick_bin_poll(self, *, soon: bool = True):
        """Ensure bin-result polling is scheduled promptly."""
        try:
            job = getattr(self, "_bin_poll_job", None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                self._bin_poll_job = None
        except Exception:
            pass

        try:
            delay = 8 if soon else 60
            self._bin_poll_job = self.after(max(1, int(delay)), self._poll_bin_results)
        except Exception:
            self._bin_poll_job = None

    def _poll_bin_results(self):
        """ワーカー結果をGUIスレッドで回収して描画へ反映（二値化）"""
        self._bin_poll_job = None

        stop = getattr(self, "_bin_worker_stop", None)
        if stop is not None and stop.is_set():
            return

        res_q = getattr(self, "_bin_res_q", None)
        got_current = False
        cur_seq = int(getattr(self, "_bin_req_seq", 0))
        if res_q is not None:
            try:
                while True:
                    seq, strength, out_img = res_q.get_nowait()
                    if seq == cur_seq:
                        got_current = True
                        try:
                            if out_img is None:
                                # fallback to non-binarized base
                                self._base = self._orig_base
                            elif PIL_AVAILABLE and isinstance(out_img, Image.Image):
                                self._orig_base = out_img.convert("RGB")
                                self._base = self._orig_base
                            else:
                                self._base = self._orig_base
                        except Exception:
                            self._base = self._orig_base

                        # Bold overlay (if enabled and non-zero) should be re-applied on top of new base.
                        try:
                            if getattr(self, "_enable_bold_slider", False) and int(getattr(self, "_bold_strength", 0)) != 0:
                                self._apply_bold_now(int(getattr(self, "_bold_strength", 0)), notify=False)
                        except Exception as e:
                            _log_exception_once('bin_then_bold', e)

                        # enforce "fit width" on update
                        try:
                            self._fit_width()
                        except Exception:
                            try:
                                self._schedule_render()
                                self._schedule_hq_render()
                            except Exception:
                                pass
            except Exception:
                pass

        # pending flag clear
        try:
            if got_current and int(getattr(self, "_bin_pending_seq", -1)) == cur_seq:
                self._bin_pending = False
        except Exception as e:
            _log_exception_once('bin_pending_clear', e)

        # continue polling
        try:
            delay = 20 if bool(getattr(self, "_bin_pending", False)) else 90
            self._bin_poll_job = self.after(int(delay), self._poll_bin_results)
        except Exception:
            self._bin_poll_job = None
    def _bold_worker_loop(self):
        """太字化の重処理はここで実行（GUIスレッドをブロックしない）"""
        stop = getattr(self, "_bold_worker_stop", None)
        req_q = getattr(self, "_bold_req_q", None)
        res_q = getattr(self, "_bold_res_q", None)
        if stop is None or req_q is None or res_q is None:
            return

        while not stop.is_set():
            try:
                seq, strength = req_q.get(timeout=0.2)
            except Exception:
                continue

            if stop.is_set():
                break

            # 連続要求は最後の1つだけ処理（中間は捨てる）
            try:
                while True:
                    seq2, strength2 = req_q.get_nowait()
                    seq, strength = seq2, strength2
            except Exception:
                pass

            if stop.is_set():
                break

            try:
                strength = int(max(-100, min(100, int(strength))))
            except Exception:
                strength = 0

            if strength == 0:
                out_img = self._orig_base
            else:
                try:
                    base_ref = self._orig_base
                    out_img = self._apply_bold_preview_pil(base_ref, strength)
                except Exception:
                    out_img = self._orig_base


            try:
                res_q.put_nowait((seq, strength, out_img))
            except Exception:
                # キューが詰まっても最新優先
                try:
                    while True:
                        res_q.get_nowait()
                except Exception:
                    pass
                try:
                    res_q.put_nowait((seq, strength, out_img))
                except Exception:
                    pass
    def _kick_bold_poll(self, *, soon: bool = True):
        """Ensure bold-result polling is scheduled promptly.
    
        Preset buttons should feel instant. If a slow (idle) poll is already scheduled,
        cancel it and reschedule sooner.
        """
        try:
            job = getattr(self, "_bold_poll_job", None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                self._bold_poll_job = None
        except Exception:
            pass
    
        try:
            delay = 8 if soon else 60
            self._bold_poll_job = self.after(max(1, int(delay)), self._poll_bold_results)
        except Exception:
            self._bold_poll_job = None
    
    def _poll_bold_results(self):
        """ワーカー結果をGUIスレッドで回収して描画へ反映"""
        self._bold_poll_job = None

        stop = getattr(self, "_bold_worker_stop", None)
        if stop is not None and stop.is_set():
            return

        res_q = getattr(self, "_bold_res_q", None)
        got_current = False
        cur_seq = int(getattr(self, "_bold_req_seq", 0))
        if res_q is not None:
            try:
                while True:
                    seq, strength, out_img = res_q.get_nowait()
                    # 最新要求のみ反映
                    if seq == cur_seq:
                        got_current = True
                        try:
                            if out_img is None:
                                self._base = self._orig_base
                            elif PIL_AVAILABLE and isinstance(out_img, Image.Image):
                                self._base = out_img
                            elif NUMPY_AVAILABLE:
                                try:
                                    arr = np.array(out_img)
                                    if arr.ndim == 2:
                                        arr = np.stack([arr, arr, arr], axis=-1)
                                    if arr.ndim == 3 and arr.shape[-1] >= 4:
                                        arr = arr[..., :3]
                                    self._base = Image.fromarray(arr.astype(np.uint8), mode="RGB")
                                except Exception:
                                    self._base = self._orig_base
                            else:
                                self._base = self._orig_base
                        except Exception:
                            self._base = self._orig_base
                            self._base = self._orig_base
                        try:
                            self._schedule_render()
                        except Exception as e:
                            _log_exception_once('ZoomImageViewer.schedule_render', e)
            except Exception:
                pass

        # If the latest request has been applied, clear pending flag
        try:
            if got_current and int(getattr(self, "_bold_pending_seq", -1)) == cur_seq:
                self._bold_pending = False
        except Exception as e:
            _log_exception_once('ZoomImageViewer.bold_pending_clear', e)

        # 継続ポーリング
        try:
            delay = 20 if bool(getattr(self, "_bold_pending", False)) else 90
            self._bold_poll_job = self.after(int(delay), self._poll_bold_results)
        except Exception:
            # ウィンドウ破棄済みなど
            self._bold_poll_job = None

    def _apply_bold_now(self, v: int, *, notify: bool = True):
        """太字スライダーの値を、拡大プレビューへリアルタイム反映する（重処理は別スレッド）"""
        self._bold_job = None
        try:
            v = int(max(-100, min(100, int(v))))
        except Exception:
            v = 0

        # 0) リクエスト番号更新（進行中の計算結果の取り違え防止）
        if getattr(self, "_bold_req_q", None) is not None and getattr(self, "_bold_worker_thread", None) is not None:
            self._bold_req_seq = int(getattr(self, "_bold_req_seq", 0)) + 1
            seq = self._bold_req_seq
        else:
            seq = None

        # 1) 画面の即時応答：0なら即戻す
        if v == 0:
            try:
                self._base = self._orig_base
            except Exception:
                pass
            # Immediately redraw (preset buttons may not trigger any other redraw)
            try:
                self._schedule_render(fast=True)
                self._schedule_hq_render()
            except Exception:
                pass
        else:
            # 2) 非0はワーカーへ（ワーカーが無い場合は従来同期でフォールバック）
            if seq is not None:
                try:
                    self._bold_pending = True
                    self._bold_pending_seq = int(seq)
                except Exception:
                    pass
                self._enqueue_bold_request(seq, v)
                # Kick polling quickly so preset buttons feel instant
                try:
                    self._kick_bold_poll(soon=True)
                except Exception:
                    pass
            else:
                # フォールバック（古い環境など）
                try:
                    self._base = self._apply_bold_preview_pil(self._orig_base, v)
                except Exception:
                    self._base = self._orig_base
                # Fallback is synchronous; redraw now
                try:
                    self._schedule_render(fast=True)
                    self._schedule_hq_render()
                except Exception:
                    pass
        # 3) 設定側へも反映（コールバックがあれば呼ぶ）
        # 値が変わっていないのに再通知すると同期ループの原因になるため、
        # 内部再適用（例：二値化変更後に太字を再適用）では notify=False を使う。
        if notify and callable(getattr(self, '_on_apply_bold', None)):
            try:
                # (strength) または (image, strength) のどちらでもOK
                try:
                    self._on_apply_bold(v)
                except TypeError:
                    self._on_apply_bold(None, v)
            except Exception as e:
                _log_exception_once('L3550b', e)


    def replace_sources_from_rgb(
        self,
        base_rgb=None,
        bin_src_rgb=None,
        bin_zero_rgb=None,
        keep_fit_width: bool = False,
        suppress_callbacks: bool = False,
    ):
        """外部（プレビュー等）から、表示元画像/二値化元画像を差し替える。

        目的:
          - プレビュー側で角度オフセット等が変わったとき、既に開いている拡大ビューアにも反映する
          - 反映後の表示を維持（倍率固定）。fitに戻したい場合は keep_fit_width=True を指定

        suppress_callbacks=True のときは、同期コールバック（メイン側スライダー更新）を抑制します。
        """
        try:
            if not bool(self.winfo_exists()):
                return
        except Exception:
            return

        # callbacks can cause feedback loops; optionally suppress
        cb_bold = getattr(self, "_on_apply_bold", None)
        cb_bin = getattr(self, "_sync_bin_callback", None)
        if suppress_callbacks:
            try:
                self._on_apply_bold = None
            except Exception:
                pass
            try:
                self._sync_bin_callback = None
            except Exception:
                pass

        try:
            if bin_src_rgb is not None:
                self._bin_src_rgb = bin_src_rgb
            if bin_zero_rgb is not None:
                self._bin_zero_rgb = bin_zero_rgb

            if base_rgb is not None:
                # base_rgb is expected RGB uint8; best-effort convert
                try:
                    pil = Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")
                except Exception:
                    pil = Image.fromarray((np.array(base_rgb)).astype(np.uint8), mode="RGB")
                self._orig_base = pil.convert("RGB")
                self._base = self._orig_base

            # rebuild according to current slider state
            if getattr(self, "_enable_bin_slider", False) and (getattr(self, "_bin_src_rgb", None) is not None or getattr(self, "_bin_zero_rgb", None) is not None):
                try:
                    self._apply_bin_now(int(getattr(self, "_bin_strength", 0)))
                except Exception:
                    try:
                        self._schedule_render()
                    except Exception:
                        pass
            elif getattr(self, "_enable_bold_slider", False) and int(getattr(self, "_bold_strength", 0)) != 0:
                try:
                    self._apply_bold_now(int(getattr(self, "_bold_strength", 0)))
                except Exception:
                    try:
                        self._schedule_render()
                    except Exception:
                        pass
            else:
                try:
                    self._schedule_render()
                except Exception:
                    pass

            if keep_fit_width:
                try:
                    self._fit_width()
                except Exception:
                    pass
        finally:
            # restore callbacks
            if suppress_callbacks:
                try:
                    self._on_apply_bold = cb_bold
                except Exception:
                    pass
                try:
                    self._sync_bin_callback = cb_bin
                except Exception:
                    pass
    # ------------------------------------------------------------------
    # 表示位置: 初期表示を中央に合わせる（スクロール領域の中央へ）
    # ------------------------------------------------------------------
    def _center_view(self):
        try:
            self.update_idletasks()
            cw = max(1, int(self.canvas.winfo_width()))
            ch = max(1, int(self.canvas.winfo_height()))

            # 現在描画している表示画像サイズ（px）
            iw = None
            ih = None
            if hasattr(self, "_disp_w") and hasattr(self, "_disp_h"):
                try:
                    iw = int(self._disp_w)
                    ih = int(self._disp_h)
                except Exception:
                    iw = None
                    ih = None
            if (iw is None or ih is None) and hasattr(self, "_tkimg") and self._tkimg is not None:
                try:
                    iw = int(self._tkimg.width())
                    ih = int(self._tkimg.height())
                except Exception:
                    iw = None
                    ih = None
            if iw is None or ih is None:
                return

            # moveto は「左上位置 / 全体サイズ」の割合
            if iw <= cw:
                xfrac = 0.0
            else:
                xfrac = max(0.0, min(1.0, ((iw - cw) / 2.0) / float(iw)))
            if ih <= ch:
                yfrac = 0.0
            else:
                yfrac = max(0.0, min(1.0, ((ih - ch) / 2.0) / float(ih)))

            self.canvas.xview_moveto(xfrac)
            self.canvas.yview_moveto(yfrac)
        except Exception as e:
            _log_exception_once('L3666', e)


    def _schedule_render(self, *, fast: bool = False, delay_ms: int = None):
        """Schedule a render on the UI thread.

        fast=True is used for interactive zooming (mousewheel/slider drag). It avoids
        expensive high-quality resampling while the user is actively changing zoom.
        A high-quality render is scheduled separately by _schedule_hq_render().
        """
        if self._render_job is not None:
            try:
                self.after_cancel(self._render_job)
            except Exception as e:
                _log_exception_once('L3673', e)
        try:
            self._fast_render = bool(fast)
        except Exception:
            self._fast_render = False

        d = None
        try:
            d = int(delay_ms) if (delay_ms is not None) else None
        except Exception:
            d = None
        if d is None:
            d = 12 if fast else 40
        self._render_job = self.after(max(1, int(d)), self._render)

    def _schedule_hq_render(self):
        """Schedule a high-quality render after zoom settles (debounced)."""
        if getattr(self, "_hq_render_job", None) is not None:
            try:
                self.after_cancel(self._hq_render_job)
            except Exception as e:
                _log_exception_once('L3674', e)
        self._hq_render_job = self.after(220, self._render_hq)

    def _render_hq(self):
        self._hq_render_job = None
        try:
            if not int(self.winfo_exists()):
                return
        except Exception:
            return
        try:
            self._fast_render = False
        except Exception:
            pass
        # Render soon; still cancellable if another zoom event happens.
        try:
            self._schedule_render(fast=False, delay_ms=1)
        except Exception as e:
            _log_exception_once('L3675', e)


    def _render(self):
        self._render_job = None
        self._hq_render_job = None
        self._fast_render = False
        try:
            base_img = self._base
            # Ensure base is a PIL.Image (defensive: worker outputs or external calls might pass array-like)
            if not hasattr(base_img, "size"):
                try:
                    if NUMPY_AVAILABLE:
                        arr = np.array(base_img)
                        if arr.ndim == 2:
                            arr = np.stack([arr, arr, arr], axis=-1)
                        if arr.ndim == 3 and arr.shape[-1] >= 4:
                            arr = arr[..., :3]
                        base_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
                    else:
                        base_img = self._orig_base
                except Exception:
                    base_img = self._orig_base
                self._base = base_img

            w0, h0 = base_img.size
            # Desired size at current zoom (may be too large for Tk PhotoImage on some environments)
            w = int(max(1, min(16000, round(w0 * self._zoom))))
            h = int(max(1, min(16000, round(h0 * self._zoom))))
            # Resample strategy:
            # - General images: LANCZOS (best quality), switch to lighter filters at huge zoom.
            # - OCR/binarization preview: avoid ringing/halo around text (縁取り) by using BOX/BILINEAR,
            #   and use NEAREST when in binary mode.
            resample = Image.LANCZOS
            try:
                if getattr(self, "_enable_bin_slider", False):
                    bs = int(getattr(self, "_bin_strength", 0))
                    if bs > 0:
                        # binary preview: keep edges crisp (no gray halo)
                        resample = Image.NEAREST
                    else:
                        # original preview under bin slider: downsample with BOX, upsample with BILINEAR
                        if float(self._zoom) < 1.0:
                            resample = Image.BOX
                        else:
                            resample = Image.BILINEAR
                else:
                    # 200%超の巨大ズームではLANCZOSが重くなりやすいので軽量フィルタに切替
                    if float(self._zoom) > 2.0:
                        resample = Image.BILINEAR
                    if float(self._zoom) > 3.0:
                        resample = Image.NEAREST
            except Exception as e:
                _log_exception_once('L3690', e)

            # Fast interactive rendering: avoid expensive high-quality resampling while zoom is changing.
            try:
                if bool(getattr(self, "_fast_render", False)):
                    if resample == Image.LANCZOS:
                        resample = Image.BILINEAR if float(self._zoom) >= 1.0 else Image.BOX
            except Exception as e:
                _log_exception_once('L3691', e)

            # Create PhotoImage; if it fails (very large images / Tk limits), reduce zoom and retry.
            last_err = None
            target_zoom = float(self._zoom)
            photo = None
            for _attempt in range(5):
                w = int(max(1, min(16000, round(w0 * target_zoom))))
                h = int(max(1, min(16000, round(h0 * target_zoom))))
                try:
                    img = base_img.resize((w, h), resample)
                    photo = ImageTk.PhotoImage(img)
                    break
                except Exception as e:
                    last_err = e
                    target_zoom *= 0.75

            if photo is None:
                raise last_err if last_err is not None else RuntimeError("render failed")

            # If we had to reduce zoom to render, keep UI consistent.
            if abs(target_zoom - float(self._zoom)) > 1e-6:
                try:
                    self._zoom = float(target_zoom)
                except Exception:
                    pass
                try:
                    self._suppress_zoom_cmd = True
                    self.sld_zoom.set(max(10.0, min(400.0, float(target_zoom) * 100.0)))
                except Exception:
                    pass
                finally:
                    try:
                        self._suppress_zoom_cmd = False
                    except Exception:
                        pass

            self._photo = photo
            if self._img_id is None:
                self._img_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
            else:
                self.canvas.itemconfigure(self._img_id, image=self._photo)

            self.canvas.configure(scrollregion=(0, 0, w, h))
            try:
                if getattr(self, '_first_render', False) and getattr(self, '_did_initial_fit', False):
                    self._center_view()
                    self._first_render = False
            except Exception as e:
                _log_exception_once('L3705', e)
        except Exception as e:
            # preview: do not crash; log once
            _log_exception_once('ZoomImageViewer.render', e, context={'zoom': getattr(self,'_zoom',None)})
            return
