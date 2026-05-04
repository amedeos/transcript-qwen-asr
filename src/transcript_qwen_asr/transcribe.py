"""Thin wrapper around the qwen-asr Qwen3ASRModel."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

MODEL_REPOS = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load(
    model_size: str,
    want_srt: bool,
    dtype: str = "bf16",
    device: str = "cuda:0",
    batch_size: int = 1,
    beam_size: int = 4,
) -> Any:
    """Load Qwen3ASRModel (and forced aligner if want_srt)."""
    from qwen_asr import Qwen3ASRModel

    repo = MODEL_REPOS[model_size]
    log.info("Loading ASR model %s on %s (%s)…", repo, device, dtype)
    model = Qwen3ASRModel.from_pretrained(
        repo,
        forced_aligner=ALIGNER_REPO if want_srt else None,
        forced_aligner_kwargs={"dtype": DTYPES[dtype], "device_map": device} if want_srt else None,
        max_inference_batch_size=batch_size,
        dtype=DTYPES[dtype],
        device_map=device,
    )

    if beam_size > 1:
        gen = model.model.generation_config
        gen.num_beams = beam_size
        gen.do_sample = False
        gen.early_stopping = True
        gen.no_repeat_ngram_size = 3
        gen.length_penalty = 1.0
        gen.repetition_penalty = 1.1
        log.info(
            "Beam search enabled (num_beams=%d, no_repeat_ngram_size=3, repetition_penalty=1.1).",
            beam_size,
        )

    return model


def run(
    model: Any,
    pcm: np.ndarray,
    context: str,
    language: str | None,
    want_srt: bool,
) -> Any:
    """Transcribe a single audio array. Returns an ASRTranscription."""
    from transcript_qwen_asr.audio import SAMPLE_RATE

    try:
        results = model.transcribe(
            audio=(pcm, SAMPLE_RATE),
            context=context,
            language=language,
            return_time_stamps=want_srt,
        )
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA out of memory during transcription. Retry with `--model 0.6B`, "
            "lower `--beam-size`, lower `--batch-size`, or `--dtype fp16`."
        ) from e
    return results[0]
