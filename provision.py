# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Provision the S3 Vectors bucket and vector index used by this sample.

Amazon S3 Vectors has a dedicated ``s3vectors`` client, separate from the standard
``s3`` client. This script creates the vector bucket and a vector index configured
for Amazon Titan Text Embeddings v2, which means 1024 dimensions, the cosine
distance metric, and the float32 data type.

Run it from the repository root before ingesting any documents.

    python provision.py

Both steps tolerate an "already exists" conflict, so the script is safe to re-run.
"""

from __future__ import annotations

from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from config import Config, load_config

# Vector index configuration matching Amazon Titan Text Embeddings v2 output.
# Public so the Streamlit UI can report the index shape it is querying without
# hardcoding a second copy of these values.
INDEX_DIMENSION = 1024
INDEX_DATA_TYPE = "float32"
INDEX_DISTANCE_METRIC = "cosine"
# The chunk text is stored as metadata but not indexed for filtering.
NON_FILTERABLE_METADATA_KEYS = ["source_text"]


def _is_already_exists_error(error: ClientError) -> bool:
    """Return True when a ``ClientError`` indicates an "already exists" conflict.

    S3 Vectors reports a resource that already exists via a conflict error code
    or an "already exists" message. This treats either signal as an expected,
    non-fatal state so provisioning can continue (Requirement 4.5).

    Args:
        error: The botocore ``ClientError`` raised by an ``s3vectors`` call.

    Returns:
        True if the error represents an already-exists conflict, else False.
    """
    error_info = error.response.get("Error", {}) if error.response else {}
    code = str(error_info.get("Code", ""))
    message = str(error_info.get("Message", ""))
    if code in ("ConflictException", "BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
        return True
    return "already exist" in message.lower()


def create_resources(config: Config, client: Optional[Any] = None) -> None:
    """Create the S3 Vector bucket and index if they do not already exist.

    Creates the vector bucket named ``config.vector_bucket_name`` and, within it,
    a vector index named ``config.index_name`` configured with 1024 dimensions,
    the cosine distance metric, the float32 data type, and ``source_text`` as a
    non-filterable metadata key (Requirements 4.1, 4.2, 4.3). If either resource
    already exists, a friendly notice is printed and provisioning continues
    without failure (Requirement 4.5).

    Args:
        config: Resolved configuration supplying the bucket name, index name, and
            AWS region.
        client: Optional pre-built ``s3vectors`` client (useful for testing). When
            omitted, a client is created for ``config.aws_region``.
    """
    if client is None:
        client = boto3.client("s3vectors", region_name=config.aws_region)

    # 1) Create the vector bucket, tolerating an already-exists conflict.
    try:
        client.create_vector_bucket(vectorBucketName=config.vector_bucket_name)
        print(f"Created vector bucket '{config.vector_bucket_name}'.")
    except ClientError as err:
        if _is_already_exists_error(err):
            print(
                f"Vector bucket '{config.vector_bucket_name}' already exists; "
                "continuing."
            )
        else:
            raise

    # 2) Create the vector index, tolerating an already-exists conflict.
    try:
        client.create_index(
            vectorBucketName=config.vector_bucket_name,
            indexName=config.index_name,
            dataType=INDEX_DATA_TYPE,
            dimension=INDEX_DIMENSION,
            distanceMetric=INDEX_DISTANCE_METRIC,
            metadataConfiguration={
                "nonFilterableMetadataKeys": NON_FILTERABLE_METADATA_KEYS
            },
        )
        print(
            f"Created vector index '{config.index_name}' "
            f"in bucket '{config.vector_bucket_name}'."
        )
    except ClientError as err:
        if _is_already_exists_error(err):
            print(
                f"Vector index '{config.index_name}' already exists; continuing."
            )
        else:
            raise


def main() -> None:
    """Entry point: load configuration and provision the bucket and index."""
    config = load_config()
    create_resources(config)


if __name__ == "__main__":
    main()
