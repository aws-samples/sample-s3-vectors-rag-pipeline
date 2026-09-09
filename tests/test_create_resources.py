# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Edge-case tests for resource provisioning (``provision.py``).

These tests exercise the "already exists" fail-soft behavior required by
Requirement 4.5: if the S3 Vector bucket already exists when provisioning runs,
the provisioner reports the conflict and continues without failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from provision import create_resources
from config import Config


def _make_config() -> Config:
    """Return a Config with concrete values for provisioning tests."""
    return Config(
        vector_bucket_name="sample-rag-vectors",
        index_name="rag-documents",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.anthropic.claude-sonnet-4-6",
        aws_region="us-east-1",
        top_k=5,
    )


def _conflict_error() -> ClientError:
    """Build a ClientError representing a bucket-already-exists conflict."""
    return ClientError(
        error_response={
            "Error": {
                "Code": "ConflictException",
                "Message": "The vector bucket already exists.",
            }
        },
        operation_name="CreateVectorBucket",
    )


def test_bucket_already_exists_reports_and_continues(capsys):
    """A bucket conflict prints an "already exists" notice and continues.

    Requirement 4.5: IF the S3_Vector_Bucket already exists when provisioning
    runs, THEN the Resource_Provisioner SHALL report that the bucket already
    exists and continue without failure.
    """
    client = MagicMock()
    # create_vector_bucket raises an already-exists conflict...
    client.create_vector_bucket.side_effect = _conflict_error()
    # ...while create_index succeeds so we can confirm provisioning continued.
    client.create_index.return_value = {}

    config = _make_config()

    # Must NOT raise despite the bucket conflict.
    create_resources(config, client=client)

    captured = capsys.readouterr()
    output = captured.out.lower()

    # A notice about the bucket already existing is printed.
    assert "already exist" in output
    assert config.vector_bucket_name in captured.out

    # Provisioning continued: the index was still created.
    client.create_index.assert_called_once()
    _, index_kwargs = client.create_index.call_args
    assert index_kwargs["vectorBucketName"] == config.vector_bucket_name
    assert index_kwargs["indexName"] == config.index_name


def test_bucket_already_exists_via_message_only(capsys):
    """An "already exists" message (without a conflict code) is also tolerated."""
    client = MagicMock()
    client.create_vector_bucket.side_effect = ClientError(
        error_response={
            "Error": {
                "Code": "SomeOtherCode",
                "Message": "Bucket already exists in your account.",
            }
        },
        operation_name="CreateVectorBucket",
    )
    client.create_index.return_value = {}

    config = _make_config()

    create_resources(config, client=client)

    captured = capsys.readouterr()
    assert "already exist" in captured.out.lower()
    client.create_index.assert_called_once()


def test_non_conflict_bucket_error_propagates():
    """A non-conflict error during bucket creation is not swallowed."""
    client = MagicMock()
    client.create_vector_bucket.side_effect = ClientError(
        error_response={
            "Error": {"Code": "AccessDeniedException", "Message": "nope"}
        },
        operation_name="CreateVectorBucket",
    )

    config = _make_config()

    with pytest.raises(ClientError):
        create_resources(config, client=client)

    # Provisioning stopped before creating the index.
    client.create_index.assert_not_called()
