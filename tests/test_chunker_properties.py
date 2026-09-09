# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the text chunker (`chunker.py`).

Each test corresponds to a numbered correctness property from the
`sample-s3-vectors-rag-pipeline` design document and runs a minimum of 100 examples.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from chunker import chunk_text

# Default chunker configuration (mirrors chunk_text's defaults / Requirement 2).
CHUNK_SIZE = 500
OVERLAP = 50

# Non-whitespace tokens so that space-joining and re-splitting round-trips
# exactly (text.split() collapses on any whitespace).
_token = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
)

# Enough tokens to guarantee two or more chunks with the default settings.
_multi_chunk_tokens = st.lists(_token, min_size=CHUNK_SIZE + 1, max_size=1200)


# Feature: sample-s3-vectors-rag-pipeline, Property 3: Adjacent chunks share the configured overlap
@settings(max_examples=100)
@given(tokens=_multi_chunk_tokens)
def test_property_3_adjacent_overlap(tokens: list[str]) -> None:
    """Adjacent chunks share exactly ``overlap`` tokens.

    For any input text that produces two or more chunks, the last ``overlap``
    tokens of each chunk equal the first ``overlap`` tokens of the next.

    Validates: Requirements 2.2
    """
    text = " ".join(tokens)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    # This generator is sized to always yield 2+ chunks; assert the precondition
    # so a regression in chunk counting cannot silently skip the overlap check.
    assert len(chunks) >= 2

    for current, following in zip(chunks, chunks[1:]):
        current_tokens = current.split()
        following_tokens = following.split()
        assert current_tokens[-OVERLAP:] == following_tokens[:OVERLAP]


# Feature: sample-s3-vectors-rag-pipeline, Property 2: Chunks never exceed the target token size
@settings(max_examples=100)
@given(
    text=st.text(),
    chunk_size=st.integers(min_value=1, max_value=50),
    overlap=st.integers(min_value=0, max_value=49),
)
def test_property_2_max_chunk_size(text: str, chunk_size: int, overlap: int) -> None:
    """Every chunk produced contains at most ``chunk_size`` tokens.

    Validates: Requirements 2.1
    """
    # ``overlap`` must be strictly less than ``chunk_size`` for a valid call.
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    for chunk in chunks:
        assert len(chunk.split()) <= chunk_size


# Feature: sample-s3-vectors-rag-pipeline, Property 4: Short text yields a single complete chunk
@settings(max_examples=100)
@given(tokens=st.lists(_token, min_size=1, max_size=CHUNK_SIZE))
def test_property_4_short_text_single_chunk(tokens: list[str]) -> None:
    """Non-empty text of ``chunk_size`` tokens or fewer yields one full chunk.

    For any non-empty input of at most ``chunk_size`` tokens, the chunker
    returns exactly one chunk, and that chunk contains all of the provided
    tokens in order.

    Validates: Requirements 2.3
    """
    text = " ".join(tokens)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    assert len(chunks) == 1
    assert chunks[0].split() == tokens


# Feature: sample-s3-vectors-rag-pipeline, Property 5: Chunking preserves the original content
@settings(max_examples=100)
@given(
    tokens=st.lists(_token, max_size=1200),
    blank=st.sampled_from(["", "   ", "\t", "\n", "  \t\n "]),
)
def test_property_5_content_preservation(tokens: list[str], blank: str) -> None:
    """Chunking preserves the original token sequence, and blank input yields no chunks.

    For any input text, concatenating the unique (non-overlapping) region of
    each chunk -- the first chunk in full, followed by each subsequent chunk
    with its leading ``overlap`` tokens removed -- reproduces the original
    token sequence. Empty or whitespace-only input produces zero chunks.

    Note: ``text.split()`` collapses whitespace, so the reconstruction is
    compared against the tokenized (split) form of the input rather than the
    raw string.

    Validates: Requirements 2.4, 2.5
    """
    # Empty and whitespace-only input must produce zero chunks (Requirement 2.4).
    assert chunk_text(blank, chunk_size=CHUNK_SIZE, overlap=OVERLAP) == []

    text = " ".join(tokens)
    expected = text.split()  # equivalent to _tokenize(text)

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    # No tokens -> no chunks (covers the empty-token-list case).
    if not expected:
        assert chunks == []
        return

    # Reconstruct: first chunk in full, then each subsequent chunk minus its
    # leading ``overlap`` tokens (the region shared with the previous chunk).
    reconstructed: list[str] = list(chunks[0].split())
    for chunk in chunks[1:]:
        reconstructed.extend(chunk.split()[OVERLAP:])

    assert reconstructed == expected
