# -*- coding: utf-8 -*-
from __future__ import annotations

"""weights_manager.py

A small helper to manage external model weights for PDF Remaster.

Goals
- Keep the main app runnable even if weights are missing.
- Provide a safe, resumable-ish download path with integrity checks.
- Avoid bringing in heavy dependencies just for downloading.

Currently managed:
- Real-ESRGAN model: RealESRGAN_x4plus_anime_6B.pth
"""

import hashlib
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


ProgressCallback = Callable[[int, int, str], None]  # (downloaded, total, status)


@dataclass(frozen=True)
class WeightSpec:
    name: str
    filename: str
    urls: List[str]
    sha256: Optional[str] = None
    min_bytes: int = 1_000_000  # sanity: at least 1MB


# NOTE: primary URL uses a pinned commit on Hugging Face (stable).
# Fallback URL points to the official Real-ESRGAN GitHub release asset.
DEFAULT_ESRGAN = WeightSpec(
    name="Real-ESRGAN (anime) x4",
    filename="RealESRGAN_x4plus_anime_6B.pth",
    urls=[
        # Hugging Face (pinned commit)
        "https://huggingface.co/nateraw/real-esrgan/resolve/44ad8adf6069185b86df22349b12f255821c86ab/RealESRGAN_x4plus_anime_6B.pth?download=true",
        # GitHub release asset (fallback)
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ],
    # SHA256 found on the Hugging Face file pointer page.
    sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    min_bytes=10_000_000,
)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _is_valid_file(path: str, spec: WeightSpec) -> Tuple[bool, str]:
    if not path or not os.path.isfile(path):
        return False, "missing"
    try:
        size = os.path.getsize(path)
        if size < spec.min_bytes:
            return False, f"too_small({size})"
        if spec.sha256:
            got = _sha256_file(path)
            if got.lower() != spec.sha256.lower():
                return False, "sha256_mismatch"
        return True, "ok"
    except Exception as e:
        return False, f"error({e})"


def download_weight(
    spec: WeightSpec,
    dst_path: str,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    timeout_sec: int = 30,
) -> str:
    """Download a weight file to dst_path with basic integrity checks.

    - Downloads to a temporary file and atomically replaces on success.
    - Verifies min_bytes and sha256 (if provided).
    - Tries URLs in order.

    Raises RuntimeError on failure.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    last_err = None
    for url in spec.urls:
        tmp_path = None
        try:
            # Create a unique temp file in the destination directory to avoid collisions.
            # Keeping it in the same directory ensures os.replace is atomic.
            dst_dir = os.path.dirname(dst_path) or "."
            base = os.path.basename(dst_path) or "weight"
            fd, tmp_path = tempfile.mkstemp(prefix=base + ".", suffix=".tmp", dir=dst_dir)
            try:
                os.close(fd)
            except Exception:
                pass
            if progress_cb:
                progress_cb(0, 0, f"connecting: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "PDFRemaster/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                total = int(r.headers.get("Content-Length") or 0)
                downloaded = 0
                t0 = time.time()
                with open(tmp_path, "wb") as f:
                    while True:
                        if cancel_flag and cancel_flag():
                            raise RuntimeError("cancelled")
                        chunk = r.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            # throttle: update at most ~30fps
                            if (time.time() - t0) > 0.03:
                                progress_cb(downloaded, total, "downloading")
                                t0 = time.time()

                if progress_cb:
                    progress_cb(downloaded, total, "verifying")

            ok, reason = _is_valid_file(tmp_path, spec)
            if not ok:
                raise RuntimeError(f"invalid_download: {reason}")

            # atomic replace
            try:
                os.replace(tmp_path, dst_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            if progress_cb:
                progress_cb(0, 0, "done")
            return dst_path

        except Exception as e:
            last_err = e
            # cleanup temp
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            continue

    raise RuntimeError(f"download failed: {spec.filename} ({last_err})")


def ensure_default_esrgan(
    weights_dir: str,
    filename: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
) -> str:
    """Ensure the default ESRGAN model exists in weights_dir."""
    fname = filename or DEFAULT_ESRGAN.filename
    dst = os.path.join(weights_dir, fname)
    ok, _ = _is_valid_file(dst, DEFAULT_ESRGAN)
    if ok:
        return dst
    return download_weight(DEFAULT_ESRGAN, dst, progress_cb=progress_cb, cancel_flag=cancel_flag)
