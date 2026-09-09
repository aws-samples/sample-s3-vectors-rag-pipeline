# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Edge-case tests for entry-point credential and bucket-missing mappings (Task 10.2).

These tests exercise the ``main`` orchestration of both entry points and assert
that common AWS setup failures are converted into the sample's concise, typed
errors (rather than raw botocore stack traces):

* Missing/invalid credentials (Requirement 8.3): botocore ``NoCredentialsError``
  and credential-code ``ClientError``s (for example ``UnrecognizedClientException``)
  are mapped to :class:`~errors.CredentialsError` and a concise message about
  resolving AWS credentials is printed.
* Missing vector bucket (Requirement 8.2): an ``s3vectors`` not-found
  ``ClientError`` whose message references the bucket is mapped to
  :class:`~errors.BucketNotFoundError`, naming the configured bucket.

They are example-based edge tests (not property-based).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

import ingest
import query
from errors import BucketNotFoundError, CredentialsError


# ---------------------------------------------------------------------------
# Credentials mapping (Requirement 8.3)
# ---------------------------------------------------------------------------


def test_ingest_main_maps_no_credentials_error(capsys) -> None:
    # Requirement 8.3: NoCredentialsError raised while creating AWS clients is
    # translated into a CredentialsError and a concise credentials message.
    with patch(
        "ingest.discover_documents", return_value=[Path("x.md")]
    ), patch("ingest.boto3.client", side_effect=NoCredentialsError()):
        with pytest.raises(CredentialsError) as exc_info:
            # Explicit empty argv so argparse does not read pytest's own args.
            ingest.main([])

    out = capsys.readouterr().out.lower()
    assert "credentials could not be resolved" in out
    # Original botocore error preserved as the cause.
    assert isinstance(exc_info.value.__cause__, NoCredentialsError)


def test_query_main_maps_no_credentials_error(capsys) -> None:
    # Requirement 8.3: NoCredentialsError on the query entry point maps to a
    # CredentialsError with the concise credentials message.
    with patch("query.boto3.client", side_effect=NoCredentialsError()):
        with pytest.raises(CredentialsError) as exc_info:
            query.main(["prog", "what is this?"])

    out = capsys.readouterr().out.lower()
    assert "credentials could not be resolved" in out
    assert isinstance(exc_info.value.__cause__, NoCredentialsError)


def test_query_main_maps_credential_code_client_error(capsys) -> None:
    # Requirement 8.3: a credential-code ClientError (e.g. UnrecognizedClientException)
    # is classified as a credentials failure and mapped to CredentialsError.
    error_response = {
        "Error": {
            "Code": "UnrecognizedClientException",
            "Message": "The security token included in the request is invalid.",
        }
    }
    with patch("query.boto3.client", return_value=MagicMock()), patch(
        "query.embed_text",
        side_effect=ClientError(error_response, "InvokeModel"),
    ):
        with pytest.raises(CredentialsError) as exc_info:
            query.main(["prog", "a question"])

    out = capsys.readouterr().out.lower()
    assert "credentials could not be resolved" in out
    assert isinstance(exc_info.value.__cause__, ClientError)


# ---------------------------------------------------------------------------
# Bucket-missing mapping (Requirement 8.2)
# ---------------------------------------------------------------------------


def test_query_main_maps_bucket_not_found(capsys) -> None:
    # Requirement 8.2: an s3vectors not-found ClientError whose message references
    # the bucket maps to BucketNotFoundError naming the configured bucket.
    config = query.load_config()

    error_response = {
        "Error": {
            "Code": "NotFoundException",
            "Message": "The specified vector bucket does not exist.",
        }
    }

    s3v_client = MagicMock()
    s3v_client.query_vectors.side_effect = ClientError(error_response, "QueryVectors")

    with patch("query.boto3.client", return_value=s3v_client), patch(
        "query.embed_text", return_value=[0.0] * 1024
    ):
        with pytest.raises(BucketNotFoundError) as exc_info:
            query.main(["prog", "a question"])

    err = exc_info.value
    assert err.bucket_name == config.vector_bucket_name
    out = capsys.readouterr().out
    assert config.vector_bucket_name in out
    assert "does not exist" in out.lower()
    # Original botocore error preserved as the cause.
    assert isinstance(err.__cause__, ClientError)
