"""Write transcription results to .txt / .srt / .json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Subtitle layout heuristics
MAX_CUE_SECONDS = 7.0
MAX_CUE_CHARS = 84
MAX_LINE_CHARS = 42
PUNCTUATION = ".!?,;:"
LONG_SILENCE_S = 0.8


@dataclass
class _Cue:
    start: float
    end: float
    text: str


def write_txt(result: Any, base: Path) -> Path:
    out = base.with_suffix(".txt")
    out.write_text(result.text.strip() + "\n", encoding="utf-8")
    return out


def write_json(result: Any, base: Path, model_repo: str) -> Path:
    out = base.with_suffix(".json")
    words: list[dict[str, Any]] | None = None
    if result.time_stamps is not None:
        words = [
            {"text": item.text, "start": float(item.start_time), "end": float(item.end_time)}
            for item in result.time_stamps.items
        ]
    payload = {
        "model": model_repo,
        "language": result.language,
        "text": result.text,
        "words": words,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def write_srt(result: Any, base: Path) -> Path:
    if result.time_stamps is None:
        raise ValueError("SRT output requires return_time_stamps=True (use --srt).")
    cues = list(_group_cues(result.time_stamps.items))
    out = base.with_suffix(".srt")
    out.write_text(_format_srt(cues), encoding="utf-8")
    return out


def _group_cues(items: Iterable[Any]) -> Iterable[_Cue]:
    items = list(items)
    if not items:
        return

    cur_start = float(items[0].start_time)
    cur_end = float(items[0].end_time)
    cur_text = items[0].text
    prev_end = cur_end

    for item in items[1:]:
        text = item.text
        start = float(item.start_time)
        end = float(item.end_time)

        candidate = _join(cur_text, text)
        gap = start - prev_end
        prev_ends_with_punct = cur_text.rstrip()[-1:] in PUNCTUATION

        should_break = (
            len(candidate) > MAX_CUE_CHARS
            or (end - cur_start) > MAX_CUE_SECONDS
            or gap > LONG_SILENCE_S
            or prev_ends_with_punct
        )

        if should_break:
            yield _Cue(cur_start, cur_end, _wrap_lines(cur_text))
            cur_start, cur_end, cur_text = start, end, text
        else:
            cur_text = candidate
            cur_end = end
        prev_end = end

    yield _Cue(cur_start, cur_end, _wrap_lines(cur_text))


def _join(left: str, right: str) -> str:
    """Join two transcription units with a space unless `right` is bare punctuation."""
    if not left:
        return right
    if right and right[0] in PUNCTUATION:
        return left + right
    return left + " " + right


def _wrap_lines(text: str) -> str:
    """Break a cue into at most two lines, each ≤ MAX_LINE_CHARS."""
    text = text.strip()
    if len(text) <= MAX_LINE_CHARS:
        return text
    words = text.split(" ")
    line1: list[str] = []
    line2: list[str] = []
    target = len(text) // 2
    cur = 0
    for w in words:
        if cur + len(w) <= target:
            line1.append(w)
            cur += len(w) + 1
        else:
            line2.append(w)
    if not line2:
        return text
    return " ".join(line1) + "\n" + " ".join(line2)


def _format_srt(cues: list[_Cue]) -> str:
    out: list[str] = []
    for i, cue in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_ts(cue.start)} --> {_ts(cue.end)}")
        out.append(cue.text)
        out.append("")
    return "\n".join(out) + "\n"


def _ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
