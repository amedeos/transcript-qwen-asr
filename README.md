# transcript-qwen-asr

Local Python CLI to transcribe video files with [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) running on an NVIDIA GPU. No cloud API. Designed for technical content in Italian and English with mixed terminology (e.g. OpenShift, OVN-Kubernetes), with optional system-prompt biasing for spelling preservation.

## Requirements

- Python ≥ 3.12
- An NVIDIA GPU with recent CUDA drivers (≥ CUDA 12.8 for Blackwell sm_120; cu130 recommended for CUDA 13.x drivers)
- `ffmpeg` in `PATH` (audio extraction from video)
- ~4 GB free disk for the 0.6B model, ~10 GB for 1.7B + forced aligner

| Variant | Model VRAM | + `--srt` aligner |
| --- | --- | --- |
| `--model 0.6B` | ≈ 2-3 GB | + ≈ 2 GB |
| `--model 1.7B` (default) | ≈ 5-7 GB | + ≈ 2 GB |

## Install

Two equivalent paths — pick whichever you prefer.

### Option A: `uv` (recommended)

```bash
git clone <this-repo>
cd transcript-qwen-asr
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

`pyproject.toml` defaults to the **CUDA 13.0** PyTorch wheel index (`cu130`, PyTorch ≥ 2.11), which is the right choice for **Blackwell GPUs (RTX 50-series, B100/B200, sm_100/sm_120)** running CUDA 13.x drivers. If you have a different setup, edit the `pytorch-cu130` entry:

| Your hardware / driver | Use index |
| --- | --- |
| Blackwell + CUDA 13.x driver | `pytorch-cu130` (default) |
| Blackwell + CUDA 12.8/12.9 driver | `pytorch-cu128` |
| Hopper / Ada / older + CUDA 12.6 | `pytorch-cu126` |
| Hopper / Ada / older + CUDA 12.4 | `pytorch-cu124` |

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

### Option B: classic `pip` + `requirements.txt`

```bash
git clone <this-repo>
cd transcript-qwen-asr
python3.12 -m venv .venv
source .venv/bin/activate

# 1. Install torch from the CUDA wheel matching your driver.
#    Blackwell GPUs (RTX 50-series, B100/B200) need cu128 or cu130 — older indexes won't work.
#    See the table above for which to pick.
pip install --index-url https://download.pytorch.org/whl/cu130 torch

# 2. Install the rest:
pip install -r requirements.txt

# 3. (Optional) Also install the package itself so `transcript-qwen-asr` and
#    `python -m transcript_qwen_asr` work as commands. Skip this if you only
#    want to use the `./transcript-qwen-asr.py` launcher script.
pip install --no-deps -e .
```

### Pre-download the models (optional)

Avoids a 1-3 GB download on the first transcription:

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B   # only if you'll use --srt
```

## Usage

You can launch the tool three ways, all equivalent:

```bash
# 1. Top-level launcher script (works without installing the package)
./transcript-qwen-asr.py video.mp4 --txt

# 2. As a module
python -m transcript_qwen_asr video.mp4 --txt

# 3. Installed console script (only after `uv pip install -e .` or `pip install -e .`)
transcript-qwen-asr video.mp4 --txt
```

```text
transcript-qwen-asr VIDEO [VIDEO ...]
  --model {0.6B,1.7B}      default: 1.7B
  -o, --output PATH        output basename (or directory if multiple inputs)
  --txt --srt --json       pick output formats (--txt is default if none given)
  --prompt STR             free-form context biasing (system prompt)
  --glossary PATH          newline-separated technical-terms file
  --language LANG          force language; omit for auto-detection (handles IT/EN mixing)
  --beam-size N            beam-search width (default: 4; 1 = greedy)
  --preprocess             apply a conservative ffmpeg filter chain before ASR (off by default)
  --batch-size N           default: 1 (VRAM scales with batch-size × beam-size)
  --device DEV             default: cuda:0
  --dtype {bf16,fp16,fp32} default: bf16
  -v, --verbose
```

### Examples

Plain text transcript with the default 1.7B model:
```bash
transcript-qwen-asr talk.mp4 --txt
# → talk.txt next to the input
```

All three formats with technical-term biasing via a glossary file:
```bash
cat > k8s.txt <<'EOF'
# Lines starting with '#' are comments; blanks are ignored.
OpenShift
OVN-Kubernetes
PostgreSQL
kubectl
FFmpeg
EOF

transcript-qwen-asr talk.mp4 \
    --glossary k8s.txt \
    --prompt "Technical talk about Kubernetes networking." \
    --txt --srt --json \
    -o transcripts/talk1
# → transcripts/talk1.txt, transcripts/talk1.srt, transcripts/talk1.json
```

Batch over multiple videos (model loaded once):
```bash
transcript-qwen-asr ~/Videos/*.mp4 --txt --srt -o ~/transcripts/
```

### Glossary file format

Plain text. One term per line. Lines starting with `#` and blank lines are ignored. Terms are concatenated into the system prompt as:

> Preserve spelling and capitalization for the following technical terms: A, B, C.

If `--prompt` is also given, the free-form prompt comes first, then a blank line, then the terms sentence.

### Notes & caveats

- **`--srt` is significantly slower** than `--txt` / `--json`. The forced aligner forces the model into 180-second chunks (vs. 1200 s for ASR-only), so a 1-hour video produces ~6× more model invocations. Italian and English are both supported by the aligner.
- **Mixed-language video** (IT + EN switching mid-talk) works automatically; omit `--language`. The detected language is reported in the JSON output as a comma-separated string, e.g. `"Italian,English"`.
- **CUDA OOM**: re-run with `--model 0.6B`, lower `--beam-size`, lower `--batch-size`, or `--dtype fp16`. The decoder allocates KV cache proportional to `batch-size × beam-size`, so the two flags together set the VRAM ceiling.
- **Quality vs. speed**: `--beam-size 4` (default) explores four hypothesis sequences in parallel during decoding, which reduces hallucinations and improves rare/technical term recognition compared to greedy decoding (`--beam-size 1`). It costs roughly 1.5–2× the wall time on a 1.7B model. `--preprocess` adds a light ffmpeg cleanup (high-pass at 80 Hz, low-pass at 7.5 kHz, dynamic normalization) — almost free in time and helpful when the source has uneven volume or low-frequency rumble.
- **Context biasing is a soft hint**, not a guarantee. The model is more likely to produce the desired spelling but may still err on rare or out-of-distribution terms.
- **Long-form single-language content is not the sweet spot.** The underlying `qwen-asr` wrapper caps generation at `max_new_tokens=512` and chunks audio at 1200 s. On dense Italian speech (~150 wpm), this means roughly **85% of every full chunk is silently dropped** because the audio carries far more tokens than the cap allows. In failure mode the model can also leak the system prompt verbatim into the transcript. For long meetings or talks in a single language (especially Italian), a Whisper-large-v3 setup with VAD will produce better and more complete transcripts. Qwen3-ASR-1.7B remains useful for short clips, heavy code-switching, and languages where Whisper is weaker.

## Architecture

```
video → ffmpeg → 16 kHz mono PCM (numpy)
                     │
                     ▼
            Qwen3ASRModel.transcribe(audio, context, return_time_stamps)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
       .txt        .srt         .json
```

All ASR happens via the official `qwen-asr` PyPI wrapper: the wrapper handles internal chunking at energy-minimum boundaries, batching, and (when `--srt` is on) forced alignment for word-level timestamps. The system-prompt `context` is injected into every chunk's chat template automatically.

## License

See [LICENSE](LICENSE).
