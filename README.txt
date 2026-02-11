PDF Remaster_v1_0_0 (split prototype)

How to run:
  python run_app.py

Files:
  constants.py   - shared constants (app name/version, defaults)
  settings_io.py  - config paths + JSON read/write (side-effect free)
  pdf_io.py       - PDF open/save + metadata helpers (PyMuPDF safe wrappers)
  pdf_compose.py  - page composition (background + invisible text embedding)
  image_ops.py    - image processing mixin (SR/deskew/OCR preproc/boldness)
  embed.py        - OCR/embedding data-structures (first extraction step)
  ocr_worker.py   - multiprocessing OCR worker loop (spawn-safe)
  ocr_pipeline.py - OCR pipeline helpers (singleproc + mp helpers, token conversion, fallback)
  engine.py       - core pipeline (PdfOcrEnhanceEngine; orchestrator)
  gui.py          - Tk UI (explicit imports; no engine namespace injection)
  run_app.py      - entry point


[Step9] Added process_runner.py (process_pdf main loop extracted) and log_utils.py (throttled exception logging for runner).

- text_embed.py: 透明テキスト埋め込み／フォント埋め込み（PyMuPDF依存の隔離）

[Step11] text_embed.py にフォント自動選択（AUTO解決）を移動し、engine.py から分離しました。


[Step12] log_utils 集約: engine/gui の _log_exception_once を log_utils.py に統一し、依存を縮小。

[Step40] A-2 강화: preview panels keyed by internal panel_id (bg/ocr/overlay) instead of title strings; Zoom viewer tracking also uses panel_id to avoid breakage when UI titles change.

[Step48] Bold stroke darkening tuned: when strengthening text (+ side), the newly-added stroke region is darkened more (closer to the older v97 look) while keeping the step46+ smooth high-end behavior.


---
[STEP58] 優先度A: OCRマルチプロセスの終了処理を堅牢化
- ocr_pipeline._shutdown_mp_ocr_workers を強化（sentinel送信は非ブロッキング、graceful→terminate、Queue close/cancel_join_thread）
- process_runner の finally から result_q も渡して確実に終了
---
--------------------------------
公開準備（A1: 依存固定 / A2: weights配布・案内）
--------------------------------

[A1] 依存関係（requirements）を同梱しました
- requirements.txt         : 基本（NumPy 1.26.4固定）
- requirements-gpu.txt     : GPU向け手順（torchは公式手順で先に入れる想定）
- install_cpu.bat          : CPU環境での簡易インストール
- install_gpu_after_torch.bat : GPU環境での簡易インストール（torch導入後に実行）

※ GPU版の PyTorch は CUDA により導入方法が異なるため、torch/torchvision は requirements に固定せず、
   公式の案内どおりに先に入れることを推奨します。

[A2] SR（Real-ESRGAN）の重み(.pth)を「アプリから取得」できるようにしました
- 設定欄の「SRモデル」行に「自動DL」ボタンを追加
- 既定ファイル: weights/RealESRGAN_x4plus_anime_6B.pth
- ダウンロード中は進捗を表示し、完了後に自動で選択・保存します
- 失敗時は「案内」から手動入手の手順も確認できます

補足（OCRモデルのダウンロード）
- yomitoku は初回実行時にモデルを Hugging Face から取得することがあります。
  速度制限が気になる場合は、環境変数 HF_TOKEN を設定してください（ログに警告が出ます）。


--------------------------------
[Step78] zip配布（展開して実行）向けのUI/保存先改善
--------------------------------

- 出力先を「入力PDFの隣（推奨） / アプリ内 output / 指定フォルダ」から選択できるようにしました。
  - 「指定フォルダ」の場合のみ、出力フォルダを参照して選択します。
  - 「入力PDFの隣」が書き込み不可の場合は、自動的に output へフォールバックします（ログに警告）。
- 入力PDF行に「フォルダ」ボタンを追加（入力PDFの入っているフォルダを開きます）。
- 出力先行に「開く」ボタンを追加（現在の出力先フォルダを開きます）。
- weights周りの配置を2段に整理しました。
  - 上段: SRモデル / 自動DL（推奨） / 案内
  - 下段: パス / 参照... / weightsフォルダ / 再読み込み
- 初回起動時、アプリフォルダに書き込める場合は「ポータブル設定」をデフォルトONにします。

--------------------------------
[Step88] OCR埋め込み座標ズレ修正（確定スケール方式）
--------------------------------
- OCR入力画像サイズ (ocr_w_px/ocr_h_px) と、背景画像サイズ (w_px/h_px) の比率から
  token座標のスケール補正 (tok_sx/tok_sy) を *推定ではなく確定* する方式へ変更。
- 縦書き・余白が大きいページで発生する「右に行くほど座標がズレる」問題を解消。

--------------------------------
[Step89] 配布用フォルダ名の整理
--------------------------------
- zip内のフォルダ名を PDF_Remaster_v1_0_0 に統一（step番号はzip名で管理）。

--------------------------------
[Step90] A1+A2 適用（依存関係整備 / プレビュー二値化・残差Deskewの単一ソース化）
--------------------------------
- requirements.txt / requirements-gpu.txt を整備し、README の前提（NumPy固定など）と矛盾しないように調整。
- プレビュー用の二値化と残差Deskew推定を image_ops.py に集約し、zoom_viewer / preview_window から呼び出す形へ統一。
  （プレビュー側と実処理側の分岐によるズレ再発を抑制）

--------------------------------
[Step91] Deskew推定器のさらなる共通化（Hough推定ロジック外出し）
--------------------------------
- HoughLinesP を用いた deskew 角度推定を module-level の共通関数 estimate_deskew_angle_by_hough() に集約。
  - 出力Deskew（_deskew_rgb_impl）とプレビュー残差Deskew（estimate_residual_deskew_angle_preview）が同一推定器を共有。
  - プレビュー側は axis_tol / mad_limit を厳しめにして保守的に（誤回転を避ける）。
- estimate_residual_deskew_angle_preview の info.update 行の破損を修正（SyntaxError 回避）。

