"""Assemble the system-prompt context passed to Qwen3-ASR for biasing."""

from __future__ import annotations

from pathlib import Path


def _read_terms(path: Path) -> list[str]:
    terms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


def build_context(prompt: str | None, glossary: Path | None) -> str:
    """Return the system prompt that biases ASR decoding.

    - `prompt` is taken verbatim.
    - `glossary` file: one term per line, `#`-prefixed lines and blanks ignored.
      Wrapped into a "preserve spelling" sentence.
    - Both given: free-form prompt first, blank line, then the terms sentence.
    - Neither given: empty string (no biasing).
    """
    parts: list[str] = []

    if prompt:
        parts.append(prompt.strip())

    if glossary is not None:
        terms = _read_terms(glossary)
        if terms:
            parts.append(
                "Preserve spelling and capitalization for the following technical terms: "
                + ", ".join(terms)
                + "."
            )

    return "\n\n".join(parts)
