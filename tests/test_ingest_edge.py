# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Edge-case tests for ingestion error / empty states (Task 7.8).

These tests cover two boundary conditions of the ingestion pipeline:

* Empty corpus (Requirement 5.8): when no documents are discovered, ``main``
  reports that no documents were found and stores zero vectors -- it must not
  construct AWS clients or call ``store_vectors``.
* Missing index (Requirements 5.7, 8.2): when ``put_vectors`` reports the target
  index as missing, ``store_vectors`` translates the botocore ``ClientError``
  into an :class:`~errors.IndexNotFoundError` that names the index.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from config import Config
from errors import IndexNotFoundError
from ingest import ChunkMetadata, VectorRecord, store_vectors


def _make_config(
    index_name: str = "documents",
    vector_bucket_name: str = "test-bucket",
) -> Config:
    """Build a Config with concrete values for assertions."""
    return Config(
        vector_bucket_name=vector_bucket_name,
        index_name=index_name,
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.anthropic.claude-sonnet-4-6",
        aws_region="us-east-1",
        top_k=5,
    )


def _make_records(count: int = 2) -> List[VectorRecord]:
    """Build a small list of VectorRecords for storage tests."""
    return [
        VectorRecord(
            key=f"doc.md::{i}",
            embedding=[0.0] * 4,
            metadata=ChunkMetadata(
                source_file="doc.md",
                chunk_index=i,
                category="operations",
                source_text=f"chunk {i}",
            ),
        )
        for i in range(count)
    ]


def test_empty_corpus_reports_no_documents_and_stores_zero_vectors(capsys) -> None:
    # Requirement 5.8: an empty corpus reports no documents found and stores zero
    # vectors -- store_vectors must not be called and no AWS clients created.
    with patch("ingest.discover_documents", return_value=[]), patch(
        "ingest.boto3.client"
    ) as mock_client, patch("ingest.store_vectors") as mock_store:
        from ingest import main

        # Pass an explicit empty argv so argparse does not read pytest's own args.
        main([])

    captured = capsys.readouterr()
    assert "No documents found" in captured.out
    # Zero vectors stored: storage never invoked.
    mock_store.assert_not_called()
    # No AWS clients constructed on the empty path.
    mock_client.assert_not_called()


def test_missing_index_raises_index_not_found_naming_index() -> None:
    # Requirements 5.7, 8.2: a put_vectors not-found is translated into an
    # IndexNotFoundError whose message names the missing index.
    index_name = "my-missing-index"
    config = _make_config(index_name=index_name)
    records = _make_records(2)

    mock_client = MagicMock()
    error_response = {
        "Error": {
            "Code": "NotFoundException",
            "Message": "index does not exist",
        }
    }
    mock_client.put_vectors.side_effect = ClientError(error_response, "PutVectors")

    with pytest.raises(IndexNotFoundError) as exc_info:
        store_vectors(records, config, mock_client)

    err = exc_info.value
    assert err.index_name == index_name
    assert index_name in str(err)
    # Original botocore error preserved as the cause.
    assert isinstance(err.__cause__, ClientError)
