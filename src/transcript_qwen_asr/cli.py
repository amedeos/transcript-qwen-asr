"""Command-line entry point for transcript-qwen-asr."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from transcript_qwen_asr import audio, glossary, output

console = Console(stderr=True)
log = logging.getLogger("transcript_qwen_asr")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript-qwen-asr",
        description="Transcribe video files locally with Qwen3-ASR on an NVIDIA GPU.",
    )
    p.add_argument("videos", nargs="+", type=Path, help="One or more video/audio files.")
    p.add_argument(
        "--model", choices=["0.6B", "1.7B"], default="1.7B",
        help="Qwen3-ASR model size (default: 1.7B).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output basename. Extensions are appended per format. "
             "When multiple inputs are given, this is treated as a directory. "
             "Default: same directory as the input, with the input's stem.",
    )
    p.add_argument("--txt", action="store_true", help="Write plain-text transcript (.txt).")
    p.add_argument("--srt", action="store_true", help="Write subtitles (.srt). Loads forced aligner.")
    p.add_argument("--json", action="store_true", help="Write structured JSON (.json).")
    p.add_argument("--prompt", default=None, help="Free-form context biasing string (system prompt).")
    p.add_argument("--glossary", type=Path, default=None,
                   help="Newline-separated technical-terms file. Combined with --prompt if both given.")
    p.add_argument("--language", default=None,
                   help="Force a single language (e.g. 'English', 'Italian'). "
                        "Omit for per-chunk auto-detection (handles IT/EN mixing).")
    p.add_argument("--batch-size", type=int, default=4, help="ASR inner batch size (default: 4).")
    p.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0).")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Model dtype (default: bf16).")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase verbosity (-v INFO, -vv DEBUG).")
    return p


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, show_time=False)],
    )


def _resolve_output_base(out_arg: Path | None, video: Path, multi: bool) -> Path:
    if out_arg is None:
        return video.with_suffix("")
    if multi:
        out_arg.mkdir(parents=True, exist_ok=True)
        return out_arg / video.stem
    out_arg.parent.mkdir(parents=True, exist_ok=True)
    return out_arg


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _setup_logging(args.verbose)

    if not (args.txt or args.srt or args.json):
        args.txt = True

    try:
        audio.ensure_ffmpeg()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    context = glossary.build_context(args.prompt, args.glossary)
    if context:
        log.info("Context biasing active (%d chars).", len(context))

    from transcript_qwen_asr import transcribe  # heavy import deferred until after --help / arg parsing

    model_repo = transcribe.MODEL_REPOS[args.model]
    with console.status(f"Loading {args.model} model…"):
        model = transcribe.load(
            model_size=args.model,
            want_srt=args.srt,
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
        )

    multi = len(args.videos) > 1
    failures = 0
    for video in args.videos:
        try:
            console.print(f"[bold cyan]→ {video}[/bold cyan]")
            with console.status(f"Decoding audio from {video.name}…"):
                pcm = audio.extract_pcm16k_mono(video)
            duration_min = len(pcm) / audio.SAMPLE_RATE / 60.0
            log.info("Decoded %.1f min of audio.", duration_min)

            with console.status(f"Transcribing {video.name} ({duration_min:.1f} min)…"):
                result = transcribe.run(
                    model=model,
                    pcm=pcm,
                    context=context,
                    language=args.language,
                    want_srt=args.srt,
                )
            log.info("Detected language: %s", result.language)

            base = _resolve_output_base(args.output, video, multi)
            written: list[Path] = []
            if args.txt:
                written.append(output.write_txt(result, base))
            if args.srt:
                written.append(output.write_srt(result, base))
            if args.json:
                written.append(output.write_json(result, base, model_repo))

            for path in written:
                console.print(f"  [green]✓[/green] {path}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            console.print(f"  [red]✗ {video}: {e}[/red]")
            log.debug("Traceback", exc_info=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
