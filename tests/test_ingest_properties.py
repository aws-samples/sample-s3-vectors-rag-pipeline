# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the ingestor (`ingest.py`).

Each test corresponds to a numbered correctness property from the
`sample-s3-vectors-rag-pipeline` design document and runs a minimum of 100 examples.
"""

from __future__ import annotations

import string
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from ingest import SUPPORTED_EXTENSIONS, discover_documents

# Windows reserved device names: a file whose base name matches one of these
# cannot be created regardless of extension, so we exclude them from generation.
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Base names use only lowercase letters and digits so that (a) they are
# filesystem-safe on every OS and (b) they never collide under the
# case-insensitive filesystem semantics of Windows/macOS.
_base_name = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=12)

# A mix of supported (.md/.txt) and unsupported extensions, plus the
# no-extension case. Extensions are lowercase to keep filenames collision-free.
_extension = st.sampled_from(["", ".md", ".txt", ".pdf", ".doc", ".markdown", ".text", ".mdx"])

# A generated filename is a base name concatenated with an extension.
_filename = st.builds(lambda base, ext: base + ext, _base_name, _extension)


# Feature: sample-s3-vectors-rag-pipeline, Property 6: Document discovery selects exactly the supported extensions
@settings(max_examples=100)
@given(filenames=st.lists(_filename, max_size=20))
def test_property_6_document_discovery(filenames: list[str]) -> None:
    """`discover_documents` returns exactly the `.md`/`.txt` files and no others.

    For any set of files placed in the sample directory, discovery selects
    exactly those whose extension is one of the supported document types and
    excludes every other file.

    Validates: Requirements 5.1
    """
    # Drop Windows-reserved base names and deduplicate (the filesystem is a set
    # of unique names; the same generated name written twice is still one file).
    unique_names = {
        name
        for name in filenames
        if Path(name).stem.lower() not in _WINDOWS_RESERVED
    }

    # A fresh temp directory per example prevents state leaking between runs.
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in unique_names:
            (Path(tmpdir) / name).touch()

        discovered = {path.name for path in discover_documents(tmpdir)}

    expected = {
        name for name in unique_names if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
    }

    assert discovered == expected


from unittest.mock import MagicMock, patch

from config import load_config
from ingest import BATCH_SIZE, ChunkMetadata, VectorRecord, build_vectors, store_vectors
from chunker import chunk_text

# Arbitrary document text, including empty/whitespace-only inputs so that the
# zero-chunk case is exercised alongside multi-chunk documents.
_document_text = st.text(max_size=4000)


# Feature: sample-s3-vectors-rag-pipeline, Property 7: Ingestion attaches correct metadata to every chunk
@settings(max_examples=100)
@given(text=_document_text, base=_base_name)
def test_property_7_ingestion_metadata(text: str, base: str) -> None:
    """`build_vectors` attaches correct metadata to every produced chunk.

    For any document, each resulting :class:`VectorRecord` carries the document
    filename as ``source_file``, the exact chunk text as ``source_text``, and a
    contiguous ``0..n-1`` ``chunk_index`` sequence in chunk order.

    The embedder is mocked so the property runs offline and focuses purely on
    metadata assembly rather than embedding behavior.

    Validates: Requirements 5.4
    """
    filename = base + ".txt"
    config = load_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / filename
        doc_path.write_text(text, encoding="utf-8")

        with patch("ingest.embed_text", return_value=[0.0] * 1024):
            records = build_vectors(doc_path, config)

    # The chunker is the source of truth for how the document splits; the
    # ingestor must produce exactly one record per chunk, in order.
    expected_chunks = chunk_text(text)
    assert len(records) == len(expected_chunks)

    # chunk_index must be exactly 0..n-1, contiguous and in order.
    assert [record.metadata.chunk_index for record in records] == list(
        range(len(expected_chunks))
    )

    for index, record in enumerate(records):
        assert record.metadata.source_file == filename
        assert record.metadata.source_text == expected_chunks[index]
        assert record.metadata.chunk_index == index


# A single VectorRecord whose key is set by the caller (see the strategy below)
# so that keys are guaranteed unique across a generated list. The embedding is a
# fixed placeholder because batching/completeness does not depend on its value.
def _make_record(key: str) -> VectorRecord:
    return VectorRecord(
        key=key,
        embedding=[0.0] * 1024,
        metadata=ChunkMetadata(
            source_file="doc.txt", chunk_index=0, category="hr", source_text=key
        ),
    )


# Feature: sample-s3-vectors-rag-pipeline, Property 8: Vector writes are batched and complete
@settings(max_examples=100, deadline=None)
@given(count=st.integers(min_value=0, max_value=1300))
def test_property_8_batching_completeness(count: int) -> None:
    """`store_vectors` batches writes at <= BATCH_SIZE and covers every record once.

    For any list of records, each ``put_vectors`` call carries at most
    :data:`BATCH_SIZE` (500) vectors, and the concatenation of the keys across
    all calls equals the input record keys exactly once, in order. An empty
    input produces no calls.

    The generated count reaches past several multiples of ``BATCH_SIZE`` so the
    multi-batch path is actually exercised rather than always fitting in one call.

    Validates: Requirements 5.5
    """
    # Use the index as the key so every record is unique by construction.
    records = [_make_record(str(index)) for index in range(count)]
    input_keys = [record.key for record in records]

    config = load_config()
    mock_client = MagicMock()

    store_vectors(records, config, mock_client)

    calls = mock_client.put_vectors.call_args_list

    # Empty input must issue no calls.
    if count == 0:
        assert calls == []
        return

    # Every batch must respect the service maximum.
    for call in calls:
        assert len(call.kwargs["vectors"]) <= BATCH_SIZE

    # The union of all batches must cover every record exactly once, in order.
    written_keys = [
        vector["key"] for call in calls for vector in call.kwargs["vectors"]
    ]
    assert written_keys == input_keys
