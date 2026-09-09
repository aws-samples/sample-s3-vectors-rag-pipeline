# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for teardown (``cleanup.py``).

Teardown must delete the index before the bucket (a bucket cannot be removed
while it still holds an index), and it must be idempotent: running it when the
resources are already gone should report that and exit cleanly rather than fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from botocore.exceptions import ClientError

from cleanup import delete_resources
from config import Config


def _make_config() -> Config:
    """Return a Config with concrete values for teardown tests."""
    return Config(
        vector_bucket_name="sample-rag-vectors",
        index_name="rag-documents",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.anthropic.claude-sonnet-4-6",
        aws_region="us-east-1",
        top_k=5,
    )


def _not_found_error(operation: str) -> ClientError:
    """Build a ClientError representing a missing resource."""
    return ClientError(
        error_response={
            "Error": {"Code": "NotFoundException", "Message": "does not exist"}
        },
        operation_name=operation,
    )


def test_deletes_index_then_bucket(capsys) -> None:
    """Both resources are deleted, and the index goes before the bucket."""
    client = MagicMock()
    config = _make_config()

    delete_resources(config, client=client)

    client.delete_index.assert_called_once_with(
        vectorBucketName=config.vector_bucket_name,
        indexName=config.index_name,
    )
    client.delete_vector_bucket.assert_called_once_with(
        vectorBucketName=config.vector_bucket_name
    )

    # Ordering matters: a bucket cannot be deleted while it still has an index.
    assert client.mock_calls.index(
        call.delete_index(
            vectorBucketName=config.vector_bucket_name,
            indexName=config.index_name,
        )
    ) < client.mock_calls.index(
        call.delete_vector_bucket(vectorBucketName=config.vector_bucket_name)
    )

    out = capsys.readouterr().out
    assert config.index_name in out
    assert config.vector_bucket_name in out


def test_missing_resources_are_reported_and_skipped(capsys) -> None:
    """Teardown is idempotent: already-deleted resources are skipped."""
    client = MagicMock()
    client.delete_index.side_effect = _not_found_error("DeleteIndex")
    client.delete_vector_bucket.side_effect = _not_found_error("DeleteVectorBucket")
    config = _make_config()

    # Must not raise even though both resources are already gone.
    delete_resources(config, client=client)

    out = capsys.readouterr().out.lower()
    assert "not found" in out


def test_unexpected_error_propagates() -> None:
    """A non-not-found error is not swallowed."""
    client = MagicMock()
    client.delete_index.side_effect = ClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        operation_name="DeleteIndex",
    )

    with pytest.raises(ClientError):
        delete_resources(_make_config(), client=client)

    client.delete_vector_bucket.assert_not_called()
