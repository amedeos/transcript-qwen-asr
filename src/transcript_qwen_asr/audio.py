"""Decode arbitrary video/audio files into 16 kHz mono float32 PCM via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install it (e.g. `emerge media-video/ffmpeg` "
            "on Gentoo, `apt install ffmpeg` on Debian/Ubuntu) and retry."
        )


def extract_pcm16k_mono(media: Path) -> np.ndarray:
    """Decode `media` to 16 kHz mono float32 PCM in [-1, 1]."""
    if not media.exists():
        raise FileNotFoundError(media)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(media),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {media}:\n{stderr}")

    pcm_int16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm_int16.size == 0:
        raise RuntimeError(f"No audio decoded from {media} (silent or no audio stream?)")
    return pcm_int16.astype(np.float32) / 32768.0
