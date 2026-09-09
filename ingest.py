# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Document ingestion entry point for the Amazon S3 Vectors RAG pipeline sample.

The ingestor reads ``.md`` / ``.txt`` documents from ``data/sample/``, splits
each into overlapping token-bounded chunks, embeds every chunk with Titan Text
Embeddings v2, and writes the resulting vectors plus metadata into an S3 Vector
index using batched ``put_vectors`` calls.

This module owns the ingestion-side data models (:class:`ChunkMetadata` and
:class:`VectorRecord`) and the document-discovery step. Later stages of the
pipeline (chunk/embed assembly, batched storage, and the ``main`` orchestration)
build on the models defined here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from chunker import chunk_text
from config import CHUNK_OVERLAP, CHUNK_SIZE, Config, load_config
from embeddings import embed_text
from errors import (
    BucketNotFoundError,
    CredentialsError,
    IndexNotFoundError,
    RagError,
)

# Default directory scanned for ingestible documents. Override with --source.
DEFAULT_SOURCE_DIR = "data/sample/"

# Botocore error codes that indicate AWS credentials are missing, invalid, or
# expired (Requirement 8.3). Distinct from a Bedrock model-access denial, which
# is handled separately as ModelAccessError.
CREDENTIAL_ERROR_CODES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "AuthFailure",
        "InvalidClientTokenId",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "ExpiredTokenException",
    }
)

# File extensions the ingestor treats as ingestible documents (Requirement 5.1).
SUPPORTED_EXTENSIONS = (".md", ".txt")

# Filename to category, so retrieval can be filtered to a subset of the corpus.
# category is written as ordinary (filterable) metadata; only non-filterable keys
# are declared at index creation, so nothing here needs a provision.py change.
FILENAME_CATEGORY = {
    "employee-handbook.md": "hr",
    "new-hire-onboarding.md": "hr",
    "health-safety-inspection.md": "operations",
    "equipment-maintenance-schedule.md": "operations",
    "menu-sourcing-guide.md": "operations",
    "supplier-contracts-summary.md": "finance",
    "q3-business-review.md": "finance",
    "marketing-plan-q4.md": "marketing",
    "customer-feedback-q3.md": "marketing",
    "it-systems-guide.md": "it",
}

# Category used for any document not in the mapping above.
DEFAULT_CATEGORY = "general"


def categorize(source_file: str) -> str:
    """Return the metadata category for a document filename.

    Falls back to :data:`DEFAULT_CATEGORY` for anything not in
    :data:`FILENAME_CATEGORY`, so ingesting an unmapped document still succeeds.

    Args:
        source_file: The document filename, for example ``employee-handbook.md``.

    Returns:
        A short lowercase category string used as filterable metadata.
    """
    return FILENAME_CATEGORY.get(source_file, DEFAULT_CATEGORY)

# Maximum number of vectors S3 Vectors accepts in a single put_vectors call.
# The service limit is 500, and the S3 Vectors best practices guidance is to
# write in large batches up to that maximum rather than many small calls.
BATCH_SIZE = 500


@dataclass
class ChunkMetadata:
    """Per-vector metadata stored alongside an embedding.

    These three attributes are written as the ``metadata`` map of each S3
    Vectors record and are what makes retrieval results attributable back to
    their source (Requirement 5.4).

    Attributes:
        source_file: Filename of the document the chunk came from.
        chunk_index: Zero-based position of the chunk within its document.
        category: Short group the document belongs to (for example ``hr`` or
            ``operations``). Filterable metadata, so retrieval can be narrowed to
            a subset of the corpus.
        source_text: The chunk's text. Stored as a non-filterable metadata key
            so the full text is retrievable without being indexed for filtering.
    """

    source_file: str
    chunk_index: int
    category: str
    source_text: str


@dataclass
class VectorRecord:
    """A single vector ready to be written to S3 Vectors.

    Bundles the vector key, its embedding, and the metadata describing the chunk
    it was produced from. Serialized to the S3 Vectors ``put_vectors`` shape at
    storage time.

    Attributes:
        key: Unique vector key, conventionally ``f"{source_file}::{chunk_index}"``.
        embedding: The 1024-element float32 embedding of the chunk text.
        metadata: The :class:`ChunkMetadata` describing this chunk's origin.
    """

    key: str
    embedding: List[float]
    metadata: ChunkMetadata


def discover_documents(directory: str = DEFAULT_SOURCE_DIR) -> List[Path]:
    """Return all ``.md`` and ``.txt`` files in ``directory``.

    Scans the given directory (non-recursively) and selects only regular files
    whose extension is one of the supported document types (Requirement 5.1).
    Extension matching is case-insensitive so that, for example, ``README.MD``
    is discovered alongside ``notes.md``. Results are sorted by path for
    deterministic ordering across runs.

    Args:
        directory: Path to the directory to scan. Defaults to ``"data/sample/"``.

    Returns:
        A sorted list of :class:`~pathlib.Path` objects, one per discovered
        document. Returns an empty list if the directory does not exist or
        contains no supported documents.
    """
    base = Path(directory)
    if not base.is_dir():
        return []

    documents = [
        entry
        for entry in base.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(documents)


def build_vectors(
    path: Path,
    config: Config,
    client: Optional[Any] = None,
) -> List[VectorRecord]:
    """Chunk a document and embed each chunk into :class:`VectorRecord`s.

    Reads the text of the document at ``path``, splits it into overlapping
    token-bounded chunks with the shared chunker (Requirement 5.2), embeds each
    chunk through Titan Text Embeddings v2 (Requirement 5.3), and assembles one
    :class:`VectorRecord` per chunk carrying its embedding and metadata. Each
    record's metadata records the document filename as ``source_file``, the
    chunk's zero-based position as ``chunk_index`` (a contiguous ``0..n-1``
    sequence in chunk order), and the chunk text as ``source_text``
    (Requirement 5.4).

    Args:
        path: Path to the document to ingest. Its ``name`` is used as
            ``source_file`` and as the key prefix.
        config: Resolved configuration supplying the embedding model id and AWS
            region used when embedding each chunk.
        client: Optional pre-constructed ``bedrock-runtime`` client forwarded to
            :func:`~embeddings.embed_text`. When omitted, the embedder
            creates its own client. Injecting a client keeps the function
            testable without live AWS calls.

    Returns:
        A list of :class:`VectorRecord`, one per chunk, in chunk order. Returns
        an empty list when the document contains no chunkable text.
    """
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    records: List[VectorRecord] = []
    for index, chunk in enumerate(chunks):
        embedding = embed_text(chunk, config, client)
        metadata = ChunkMetadata(
            source_file=path.name,
            chunk_index=index,
            category=categorize(path.name),
            source_text=chunk,
        )
        records.append(
            VectorRecord(
                key=f"{path.name}::{index}",
                embedding=embedding,
                metadata=metadata,
            )
        )

    return records


def _serialize_record(record: VectorRecord) -> dict:
    """Serialize a :class:`VectorRecord` to the S3 Vectors ``put_vectors`` shape.

    Produces the per-vector mapping expected by ``s3vectors.put_vectors``: the
    embedding is nested under ``data.float32`` and the chunk's metadata is
    flattened into the ``metadata`` map (Requirement 5.4).

    Args:
        record: The vector record to serialize.

    Returns:
        A dict of the form
        ``{"key": ..., "data": {"float32": [...]}, "metadata": {...}}``.
    """
    return {
        "key": record.key,
        "data": {"float32": record.embedding},
        "metadata": {
            "source_file": record.metadata.source_file,
            "chunk_index": record.metadata.chunk_index,
            "category": record.metadata.category,
            "source_text": record.metadata.source_text,
        },
    }


def _is_index_not_found(err: ClientError) -> bool:
    """Return whether a botocore ``ClientError`` indicates a missing index.

    S3 Vectors surfaces a missing index in a few forms depending on the API
    surface, so this checks both the error ``Code`` (for common not-found codes)
    and the human-readable message for "not found" / "does not exist" phrasing.

    Args:
        err: The botocore ``ClientError`` raised by ``put_vectors``.

    Returns:
        ``True`` if the error looks like a missing-index condition.
    """
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    code = str(error.get("Code", ""))
    message = str(error.get("Message", "")).lower()
    not_found_codes = {"NotFoundException", "ResourceNotFoundException"}
    return code in not_found_codes or "not found" in message or "does not exist" in message


def store_vectors(
    records: List[VectorRecord],
    config: Config,
    s3v_client: Any,
) -> None:
    """Write vectors to S3 Vectors in batches of at most 500 per ``put_vectors``.

    Serializes each :class:`VectorRecord` to the S3 Vectors put shape and issues
    ``put_vectors`` calls in batches of at most :data:`BATCH_SIZE` (500) vectors
    each (Requirement 5.5), so the union of all batches covers every record
    exactly once. An empty ``records`` list results in no calls.

    Args:
        records: The vector records to write. May be empty.
        config: Resolved configuration supplying the target vector bucket and
            index names.
        s3v_client: An ``s3vectors`` client used to issue ``put_vectors`` calls.

    Raises:
        IndexNotFoundError: If the target vector index does not exist. The
            original botocore error is preserved as the exception cause
            (Requirements 5.7, 8.2).
    """
    for start in range(0, len(records), BATCH_SIZE):
        batch = [
            _serialize_record(record)
            for record in records[start : start + BATCH_SIZE]
        ]
        try:
            s3v_client.put_vectors(
                vectorBucketName=config.vector_bucket_name,
                indexName=config.index_name,
                vectors=batch,
            )
        except ClientError as err:
            if _is_index_not_found(err):
                raise IndexNotFoundError(
                    config.index_name, config.vector_bucket_name
                ) from err
            raise


def _is_credentials_error(err: ClientError) -> bool:
    """Return whether a botocore ``ClientError`` indicates a credentials problem.

    Checks the error ``Code`` against :data:`CREDENTIAL_ERROR_CODES`, the set of
    codes AWS uses when credentials are missing, invalid, or expired
    (Requirement 8.3).

    Args:
        err: The botocore ``ClientError`` to classify.

    Returns:
        ``True`` if the error looks like a credential/auth failure.
    """
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    return str(error.get("Code", "")) in CREDENTIAL_ERROR_CODES


def _is_bucket_not_found(err: ClientError) -> bool:
    """Return whether a botocore ``ClientError`` indicates a missing vector bucket.

    S3 Vectors surfaces a missing bucket as a not-found code whose message
    references the bucket, so this checks both the ``Code`` and that the
    human-readable message mentions the bucket (Requirement 8.2).

    Args:
        err: The botocore ``ClientError`` to classify.

    Returns:
        ``True`` if the error looks like a missing-bucket condition.
    """
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    code = str(error.get("Code", ""))
    message = str(error.get("Message", "")).lower()
    not_found_codes = {"NotFoundException", "ResourceNotFoundException"}
    is_not_found = code in not_found_codes or "not found" in message or (
        "does not exist" in message
    )
    return is_not_found and "bucket" in message


def main(argv: Optional[List[str]] = None) -> None:
    """Run the end-to-end ingestion pipeline and report the result.

    Loads configuration, discovers ``.md`` / ``.txt`` documents under the source
    directory (``data/sample/`` by default, or ``--source``), chunks and embeds
    each document, and writes the resulting vectors to S3 Vectors in batches.
    Wires the pipeline together as discover -> chunk -> embed -> store.

    When no documents are found, reports that no documents were found and stores
    zero vectors (Requirement 5.8). Otherwise, after storage completes, prints a
    summary in the exact form ``"Ingested X chunks from Y files"`` where ``X`` is
    the total number of chunks written and ``Y`` is the number of source files
    (Requirement 5.6).

    Common AWS setup failures are converted into concise messages rather than raw
    stack traces: missing/invalid credentials (Requirement 8.3) and a missing
    vector bucket (Requirement 8.2) are reported and re-raised as a
    :class:`~errors.RagError`, which the ``__main__`` guard maps to a
    non-zero exit. Any other :class:`~errors.RagError` (for example a missing
    index) is reported the same way.

    The Bedrock Runtime and S3 Vectors clients are created here (not at import
    time) so the module stays importable without triggering AWS calls.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``). Supports
            ``--source`` to point ingestion at a different document directory.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into an Amazon S3 Vectors index."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory of .md/.txt documents to ingest (default: {DEFAULT_SOURCE_DIR}).",
    )
    args = parser.parse_args(argv)

    config = load_config()

    documents = discover_documents(args.source)
    if not documents:
        print(f"No documents found in {args.source}; nothing to ingest.")
        return

    try:
        brt = boto3.client("bedrock-runtime", region_name=config.aws_region)
        s3v_client = boto3.client("s3vectors", region_name=config.aws_region)

        all_records: List[VectorRecord] = []
        for document in documents:
            all_records += build_vectors(document, config, brt)

        store_vectors(all_records, config, s3v_client)

        print(f"Ingested {len(all_records)} chunks from {len(documents)} files")
    except NoCredentialsError as err:
        error = CredentialsError()
        print(error)
        raise error from err
    except ClientError as err:
        if _is_credentials_error(err):
            error = CredentialsError()
            print(error)
            raise error from err
        if _is_bucket_not_found(err):
            error = BucketNotFoundError(config.vector_bucket_name)
            print(error)
            raise error from err
        raise
    except RagError as err:
        # Includes IndexNotFoundError/BucketNotFoundError raised deeper in the
        # pipeline: report the concise message and let the guard exit non-zero.
        print(err)
        raise


if __name__ == "__main__":
    try:
        main()
    except RagError:
        sys.exit(1)
