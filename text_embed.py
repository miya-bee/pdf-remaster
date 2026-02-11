# -*- coding: utf-8 -*-
from __future__ import annotations

"""Text embedding utilities (invisible text + font embedding) for PDF Remaster.

This module isolates PyMuPDF text insertion logic to reduce coupling in engine.py
and improve maintainability & safety.
"""

import os
import sys
import glob

from typing import List

import fitz  # PyMuPDF

from log_utils import _log_exception_once


class TextEmbedMixin:

    # -------------------------
    # Font auto-selection (AUTO)
    # -------------------------
    def _resolve_font_path(self, font_spec: str, weights_dir: str) -> str:
        """透明テキスト用のCJKフォント（日本語対応）を決定する。

        優先順:
          1) font_spec が実在するフォントファイル(ttf/otf/ttc)ならそれを採用
          2) font_spec が空/'AUTO'なら、同梱フォントを優先探索して採用
             - (A) weights_dir/fonts（UIで weights_dir を指定している場合）
             - (B) <app_root>/weights/fonts, <app_root>/fonts, <app_root>/assets/fonts
                 * PyInstaller: sys._MEIPASS と sys.executable 近傍も探索
             - (C) ./weights/fonts, ./fonts（カレント）
             - (D) Windows Fonts（Meiryo / Yu Gothic 等）
          3) 見つからない場合は空文字（挿入側で 'japan' -> 'helv' にフォールバック）

        同梱先の推奨:
          - weights/fonts/ に IPAexGothic / Noto Sans CJK JP / Source Han Sans 等を入れる
        """
        spec = (font_spec or "").strip().strip('"').strip("'")
        spec = os.path.expandvars(os.path.expanduser(spec))
        spec = os.path.normpath(spec) if spec else ""

        # 明示指定（ファイルパス）
        if spec and os.path.isfile(spec):
            try:
                self.log(f"[INFO] フォント(指定): {spec}")
            except Exception:
                pass
            return spec
        if spec and spec.upper() != "AUTO":
            try:
                self.log(f"[WARN] 指定フォントが見つかりません。AUTOへフォールバック: {spec}")
            except Exception:
                pass

        search_dirs: List[str] = []

        def add_dir(p: str):
            try:
                if p and os.path.isdir(p) and p not in search_dirs:
                    search_dirs.append(p)
            except Exception as e:
                _log_exception_once('L_font_add_dir', e)

        # (A) weights_dir/fonts
        wd = (weights_dir or "").strip().strip('"').strip("'")
        wd = os.path.expandvars(os.path.expanduser(wd))
        wd = os.path.normpath(wd) if wd else ""
        if wd:
            add_dir(os.path.join(wd, "fonts"))

        # (B) app roots
        roots: List[str] = []
        try:
            roots.append(os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            _log_exception_once('L_font_root1', e)
        try:
            roots.append(os.getcwd())
        except Exception as e:
            _log_exception_once('L_font_root2', e)
        try:
            roots.append(os.path.dirname(os.path.abspath(sys.executable)))
        except Exception as e:
            _log_exception_once('L_font_root3', e)
        try:
            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                roots.append(os.path.abspath(meipass))
        except Exception as e:
            _log_exception_once('L_font_root4', e)

        uniq_roots: List[str] = []
        for r in roots:
            if r and r not in uniq_roots:
                uniq_roots.append(r)

        for r in uniq_roots:
            add_dir(os.path.join(r, "weights", "fonts"))
            add_dir(os.path.join(r, "fonts"))
            add_dir(os.path.join(r, "assets", "fonts"))

        # (C) current explicit
        add_dir(os.path.join(".", "weights", "fonts"))
        add_dir(os.path.join(".", "fonts"))

        for d in search_dirs:
            cand = self._pick_font_from_dir(d)
            if cand:
                try:
                    self.log(f"[INFO] フォント(AUTO:{os.path.normpath(d)}): {cand}")
                except Exception:
                    pass
                return cand

        # (D) Windows Fonts
        cand = self._pick_windows_font()
        if cand:
            try:
                self.log(f"[INFO] フォント(AUTO:WindowsFonts): {cand}")
            except Exception:
                pass
            return cand

        try:
            self.log("[WARN] フォント(AUTO): 利用可能なCJKフォントが見つかりません。標準フォールバックを使用します。")
        except Exception:
            pass
        return ""

    def _pick_font_from_dir(self, dir_path: str) -> str:
        if not dir_path or not os.path.isdir(dir_path):
            return ""
        exts = (".ttf", ".otf", ".ttc")
        files = []
        try:
            for fn in os.listdir(dir_path):
                p = os.path.join(dir_path, fn)
                if os.path.isfile(p) and fn.lower().endswith(exts):
                    files.append(p)
        except Exception as e:
            _log_exception_once('TextEmbedMixin.pick_font_from_dir', e, context={'dir': dir_path})
            return ""
        if not files:
            return ""

        def score(p: str) -> int:
            name = os.path.basename(p).lower()
            ext = os.path.splitext(name)[1]
            ext_penalty = 0 if ext in (".ttf", ".otf") else 3  # .ttc は環境差が出やすいので軽く不利に
            # 最優先: IPAexゴシック
            if "ipaexg" in name:
                return 0 + ext_penalty
            # 次点: IPAex / IPA 系
            if "ipaex" in name:
                return 5 + ext_penalty
            if name.startswith("ipag") or "ipag" in name:
                return 10 + ext_penalty
            if name.startswith("ipam") or "ipam" in name:
                return 11 + ext_penalty
            # Source Han (Adobe)
            if ("sourcehan" in name or "source han" in name) and ("sans" in name or "serif" in name):
                return 19 + ext_penalty
            # Noto CJK / Noto JP
            if "noto" in name and ("cjk" in name or "jp" in name):
                return 20 + ext_penalty
            # Meiryo / Yu Gothic っぽい
            if "meiryo" in name:
                return 30 + ext_penalty
            if ("yugoth" in name) or ("yu" in name and "goth" in name):
                return 31 + ext_penalty
            return 100

        files.sort(key=lambda p: (score(p), os.path.basename(p).lower()))
        return files[0]

    def _pick_windows_font(self) -> str:
        windir = os.environ.get("WINDIR", r"C:\\Windows")
        fonts_dir = os.path.join(windir, "Fonts")
        if not os.path.isdir(fonts_dir):
            return ""

        # まずは決め打ち候補（存在すれば即採用）
        candidates = [
            # IPA がシステムに入っている場合
            "ipaexg.ttf", "ipaexg.ttc", "ipaexgothic.ttf",
            "ipag.ttf", "ipam.ttf", "ipagp.ttf",
            # Meiryo
            "meiryo.ttc", "meiryob.ttc", "meiryo.ttf",
            # Yu Gothic（Windows 10/11）
            "YuGothR.ttc", "YuGothM.ttc", "YuGothB.ttc", "YuGothL.ttc",
            "YuGothic.ttc", "yugothic.ttf",
        ]
        for fn in candidates:
            p = os.path.join(fonts_dir, fn)
            if os.path.isfile(p):
                return p

        # 決め打ちが無い場合はパターン探索
        patterns = [
            "ipaexg*.ttf", "ipaexg*.ttc",
            "meiryo*.ttc", "meiryo*.ttf",
            "YuGoth*.ttc", "YuGothic*.ttc", "yugoth*.ttc",
        ]
        try:
            hits: List[str] = []
            for pat in patterns:
                hits.extend(glob.glob(os.path.join(fonts_dir, pat)))
            if hits:
                # IPAexGothic を優先しつつ、名称でソート
                hits.sort(key=lambda p: (0 if "ipaexg" in os.path.basename(p).lower() else 1,
                                         os.path.basename(p).lower()))
                return hits[0]
        except Exception as e:
            _log_exception_once('L_font_glob', e)
        return ""


    def _ensure_embed_fontname(self, doc: fitz.Document) -> str:
        """透明テキスト用フォントを出力PDFへ埋め込み、fontname を返す。

        - self.font_path が有効なら、そのフォントを doc に埋め込み (insert_font) して固定 fontname で使う。
        - 失敗/未指定の場合は 'japan' を試し、さらに失敗したら 'helv' へフォールバック。
        """
        if getattr(self, "_embed_font_ready", False) and getattr(self, "_embed_fontname", None):
            return self._embed_fontname

        # PyMuPDFのバージョン差対策: Document.insert_font が無い環境では、fontfile直指定を優先する。
        # docへの埋め込みを試さず、挿入側で fontfile 指定へ回す。
        if not hasattr(doc, "insert_font"):
            self._embed_fontname = "__FONTFILE__"
            self._embed_font_ready = True
            try:
                self._font_method_preferred = 'fontfile'
                self._fontname_preferred = str(getattr(self, 'font_path', '') or '')
                self._log_font_strategy_once()
            except Exception:
                pass
            return self._embed_fontname

        fp = getattr(self, "font_path", "") or ""
        if fp and os.path.isfile(fp):
            fname = "CJK0"
            try:
                doc.insert_font(fontname=fname, fontfile=fp)
                self._embed_fontname = fname
                self._embed_font_ready = True
                try:
                    self._font_method_preferred = 'embedded'
                    self._fontname_preferred = str(fname)
                    self._log_font_strategy_once()
                except Exception:
                    pass
                return fname
            except Exception as e:
                _log_exception_once('L520', e)
                try:
                    self.log(f"[WARN] フォント埋め込み失敗: {fp} ({e})。フォールバックします。")
                except Exception as e2:
                    _log_exception_once('L528', e2)

        # フォールバック: PyMuPDF組み込みCIDフォント（環境によっては 'japan' が無い場合もあるので、
        # 挿入時の最終フォールバックは _insert_invisible_text 側で担保する）
        self._embed_fontname = "japan"
        self._embed_font_ready = True
        try:
            self._font_method_preferred = 'builtin'
            self._fontname_preferred = 'japan'
            self._log_font_strategy_once()
        except Exception:
            pass
        return self._embed_fontname

    def _log_font_strategy_once(self):
        """フォント適用方式（preferred）をGUIログに1回だけ出す。"""
        if getattr(self, "_font_method_logged", False):
            return
        self._font_method_logged = True
        try:
            self.log(
                f"[INFO] font_method_preferred={getattr(self, '_font_method_preferred', '')} "
                f"font_spec={getattr(self, 'font_spec', '')} "
                f"font_path={getattr(self, 'font_path', '')} "
                f"fontname_preferred={getattr(self, '_embed_fontname', '') or getattr(self, '_fontname_preferred', '')}"
            )
        except Exception:
            pass

    def _note_font_usage(self, method: str, fontname: str = ""):
        """実際に挿入に成功したフォント方式をカウントし、最後に使った情報を保持する。"""
        try:
            if not hasattr(self, "_font_method_counts") or self._font_method_counts is None:
                self._font_method_counts = {"embedded": 0, "fontfile": 0, "builtin": 0, "helv": 0}
            if method not in self._font_method_counts:
                self._font_method_counts[method] = 0
            self._font_method_counts[method] += 1
            if fontname:
                self._fontname_last_used = str(fontname)
        except Exception:
            pass

    def _insert_invisible_text(self, page: fitz.Page, point: fitz.Point, text: str, fontsize: float, rotate: int, xscale: float = 1.0):
        """不可視テキスト挿入（render_mode=3）"""
        morph = None
        rot = rotate
        if xscale and abs(float(xscale) - 1.0) > 0.002:
            try:
                mat = fitz.Matrix(float(xscale), 1.0).prerotate(int(rotate))
                morph = (point, mat)
                rot = 0
            except Exception:
                morph = None
                rot = rotate

        fontname = "helv"
        try:
            fontname = self._ensure_embed_fontname(page.parent)
        except Exception:
            fontname = "helv"

        use_fontfile_primary = (str(fontname) == "__FONTFILE__")

        # (A) 埋め込み/組み込みフォント
        if not use_fontfile_primary:
            method = "embedded"
            if str(fontname) == "japan":
                method = "builtin"
            elif str(fontname) == "helv":
                method = "helv"
            try:
                page.insert_text(
                    point, text,
                    fontname=fontname,
                    fontsize=fontsize,
                    render_mode=3,
                    rotate=rot,
                    morph=morph,
                    overlay=True
                )
                self._note_font_usage(method, str(fontname))
                return
            except Exception as e:
                _log_exception_once('L1360', e)
                self._font_fallback_used = True

        # (B) fontfile
        fp = getattr(self, "font_path", "") or ""
        if fp and os.path.isfile(fp):
            try:
                page.insert_text(
                    point, text,
                    fontfile=fp,
                    fontsize=fontsize,
                    render_mode=3,
                    rotate=rot,
                    morph=morph,
                    overlay=True
                )
                self._note_font_usage("fontfile", fp)
                return
            except Exception as e:
                self._font_fallback_used = True
                try:
                    self.log(f"[WARN] fontfile挿入に失敗。フォールバックします: {e}")
                except Exception:
                    pass

        # (C) 最終フォールバック
        try:
            page.insert_text(
                point, text,
                fontname="japan",
                fontsize=fontsize,
                render_mode=3,
                rotate=rot,
                morph=morph,
                overlay=True
            )
            self._note_font_usage("builtin", "japan")
            self._font_fallback_used = True
            return
        except Exception as e:
            _log_exception_once('L1394', e)

        page.insert_text(
            point, text,
            fontname="helv",
            fontsize=fontsize,
            render_mode=3,
            rotate=rot,
            morph=morph,
            overlay=True
        )
        self._note_font_usage("helv", "helv")
        self._font_fallback_used = True

    def _insert_invisible_text_shape(self, shape: fitz.Shape, page: fitz.Page, point: fitz.Point, text: str, fontsize: float, rotate: int, xscale: float = 1.0):
        """不可視テキストを Shape に追加する（commit で一括反映）。"""
        morph = None
        rot = int(rotate)
        if xscale and abs(float(xscale) - 1.0) > 0.002:
            try:
                mat = fitz.Matrix(float(xscale), 1.0).prerotate(int(rotate))
                morph = (point, mat)
                rot = 0
            except Exception:
                morph = None
                rot = int(rotate)

        fontname = "helv"
        try:
            fontname = self._ensure_embed_fontname(page.parent)
        except Exception:
            fontname = "helv"

        use_fontfile_primary = (str(fontname) == "__FONTFILE__")

        # (A) 埋め込み/組み込みフォント
        if not use_fontfile_primary:
            method = "embedded"
            if str(fontname) == "japan":
                method = "builtin"
            elif str(fontname) == "helv":
                method = "helv"
            try:
                shape.insert_text(
                    point, text,
                    fontsize=float(fontsize),
                    fontname=str(fontname),
                    render_mode=3,
                    rotate=int(rot),
                    morph=morph,
                    color=None,
                    fill=None,
                )
                self._note_font_usage(method, str(fontname))
                return
            except Exception as e:
                _log_exception_once('L1447', e)
                self._font_fallback_used = True

        # (B) fontfile
        fp = getattr(self, "font_path", "") or ""
        if fp and os.path.isfile(fp):
            try:
                shape.insert_text(
                    point, text,
                    fontsize=float(fontsize),
                    fontfile=fp,
                    fontname="CJK0",
                    render_mode=3,
                    rotate=int(rot),
                    morph=morph,
                    color=None,
                    fill=None,
                )
                self._note_font_usage("fontfile", fp)
                return
            except Exception as e:
                _log_exception_once('L1466', e)
                self._font_fallback_used = True

        # (C) 最終フォールバック
        try:
            shape.insert_text(
                point, text,
                fontsize=float(fontsize),
                fontname="japan",
                render_mode=3,
                rotate=int(rot),
                morph=morph,
                color=None,
                fill=None,
            )
            self._note_font_usage("builtin", "japan")
        except Exception:
            shape.insert_text(
                point, text,
                fontsize=float(fontsize),
                fontname="helv",
                render_mode=3,
                rotate=int(rot),
                morph=morph,
                color=None,
                fill=None,
            )
            self._note_font_usage("helv", "helv")
        self._font_fallback_used = True
