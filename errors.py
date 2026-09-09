# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom exceptions for the Amazon S3 Vectors RAG pipeline sample.

The pipeline converts common AWS and setup failures into concise, actionable
messages rather than surfacing raw stack traces (Requirement 8). Each exception
below carries a user-facing message and is designed to be raised with
``raise ... from err`` so the original botocore error is preserved as the
exception cause and remains available for ``--debug`` style inspection.

Exceptions:
    RagError: Base class for all sample-specific errors.
    ModelAccessError: A Bedrock model is not access-enabled (Requirements 3.4, 8.1).
    BucketNotFoundError: The S3 Vector bucket does not exist (Requirement 8.2).
    IndexNotFoundError: The vector index does not exist (Requirement 5.7).
    CredentialsError: AWS credentials could not be resolved (Requirement 8.3).
"""

from __future__ import annotations


class RagError(Exception):
    """Base class for all Amazon S3 Vectors RAG pipeline sample errors.

    Every subclass carries a human-readable, user-facing message describing the
    problem and, where relevant, how to resolve it. Catching :class:`RagError`
    at an entry point is sufficient to handle every sample-specific failure.
    """


class ModelAccessError(RagError):
    """Raised when a Bedrock model is not access-enabled for the account.

    This maps a Bedrock ``AccessDeniedException`` raised during embedding or
    generation into a clear message that names the affected model and points to
    Bedrock model-access setup (Requirements 3.4, 8.1).

    Attributes:
        model_id: The Bedrock model identifier that could not be accessed.
    """

    def __init__(self, model_id: str, message: str | None = None) -> None:
        """Initialize the error with the affected model id.

        Args:
            model_id: The Bedrock model identifier that was denied.
            message: Optional custom user-facing message. When omitted, a
                default message naming the model is used.
        """
        self.model_id = model_id
        if message is None:
            message = (
                f"Access to Bedrock model '{model_id}' is not enabled. "
                "Enable model access in the Amazon Bedrock console "
                "(Model access) for this account and region, then retry."
            )
        super().__init__(message)


class BucketNotFoundError(RagError):
    """Raised when the configured S3 Vector bucket does not exist.

    Surfaced when an ``s3vectors`` operation reports the bucket as missing during
    ingestion or querying (Requirement 8.2).

    Attributes:
        bucket_name: The vector bucket name that could not be found.
    """

    def __init__(self, bucket_name: str, message: str | None = None) -> None:
        """Initialize the error with the missing bucket name.

        Args:
            bucket_name: The S3 Vector bucket name that does not exist.
            message: Optional custom user-facing message. When omitted, a
                default message naming the bucket is used.
        """
        self.bucket_name = bucket_name
        if message is None:
            message = (
                f"S3 Vector bucket '{bucket_name}' does not exist. "
                "Provision it first by running provision.py."
            )
        super().__init__(message)


class IndexNotFoundError(RagError):
    """Raised when the configured vector index does not exist.

    Surfaced when a ``put_vectors`` or ``query_vectors`` operation reports the
    index as missing (Requirement 5.7).

    Attributes:
        index_name: The vector index name that could not be found.
        bucket_name: The vector bucket expected to contain the index, if known.
    """

    def __init__(
        self,
        index_name: str,
        bucket_name: str | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the error with the missing index (and optional bucket) name.

        Args:
            index_name: The vector index name that does not exist.
            bucket_name: Optional name of the bucket expected to contain it.
            message: Optional custom user-facing message. When omitted, a
                default message naming the index is used.
        """
        self.index_name = index_name
        self.bucket_name = bucket_name
        if message is None:
            location = (
                f"'{index_name}' in bucket '{bucket_name}'"
                if bucket_name is not None
                else f"'{index_name}'"
            )
            message = (
                f"Vector index {location} does not exist. "
                "Provision it first by running provision.py."
            )
        super().__init__(message)


class CredentialsError(RagError):
    """Raised when AWS credentials are missing or cannot be resolved.

    Maps botocore ``NoCredentialsError`` (or an equivalent credential
    ``ClientError``) into a clear message (Requirement 8.3).
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize the error with an optional custom message.

        Args:
            message: Optional custom user-facing message. When omitted, a
                default message explaining how to configure credentials is used.
        """
        if message is None:
            message = (
                "AWS credentials could not be resolved. Configure credentials "
                "via environment variables, a shared credentials file, or an "
                "IAM role (for example, run 'aws configure'), then retry."
            )
        super().__init__(message)
