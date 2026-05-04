#!/usr/bin/env python3
"""Convenience launcher: run the CLI without installing the package.

Usage:
    ./transcript-qwen-asr.py video.mp4 --txt --srt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transcript_qwen_asr.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
