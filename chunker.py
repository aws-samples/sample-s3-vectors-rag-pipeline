# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pure, dependency-free text chunker for the Amazon S3 Vectors RAG pipeline sample.

Splits document text into overlapping, token-bounded chunks so that each
embedding captures a coherent slice of context. Tokenization is intentionally
approximate (whitespace-based) to keep the module deterministic and free of
external dependencies -- this is sample-grade, not a production tokenizer.
"""

from __future__ import annotations


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into approximate tokens by whitespace.

    This sample-grade tokenizer treats each whitespace-separated word as a single
    token. Leading, trailing, and repeated whitespace is collapsed, so empty or
    whitespace-only input yields an empty list.

    Args:
        text: The raw text to tokenize.

    Returns:
        A list of whitespace-delimited tokens, in their original order.
    """
    return text.split()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split ``text`` into ~``chunk_size``-token chunks with token overlap.

    A sliding window advances by ``chunk_size - overlap`` tokens each step, so
    every adjacent pair of chunks shares exactly ``overlap`` tokens (the last
    ``overlap`` tokens of one chunk equal the first ``overlap`` tokens of the
    next). Each chunk is the space-joined form of its token window.

    Args:
        text: The document text to split.
        chunk_size: The maximum number of tokens per chunk. Defaults to 500.
        overlap: The number of tokens shared between adjacent chunks. Must be
            less than ``chunk_size``. Defaults to 50.

    Returns:
        A list of chunk strings. Empty or whitespace-only input returns ``[]``.
        Input of ``chunk_size`` tokens or fewer returns exactly one chunk
        containing all of the provided tokens.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or if ``overlap`` is
            negative or not strictly less than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    tokens = _tokenize(text)
    if not tokens:
        return []

    # Short text: a single chunk containing every token.
    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(" ".join(window))
        # Stop once this window reaches the end of the token sequence.
        if start + chunk_size >= len(tokens):
            break
        start += step

    return chunks
