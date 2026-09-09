# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Delete the S3 Vectors index and bucket created by this sample.

This is the teardown counterpart to ``provision.py``. It removes the vector index
first and then the vector bucket, because a bucket cannot be deleted while it
still contains an index. Deleting the index also deletes the vectors stored in it.

Run from the repository root:

    python cleanup.py

The script asks for confirmation before deleting anything. Pass ``--yes`` to skip
the prompt (useful in scripted cleanup):

    python cleanup.py --yes

Both steps fail soft when a resource is already gone, so the script is safe to
re-run.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from config import Config, load_config


def _is_not_found_error(error: ClientError) -> bool:
    """Return True when a ``ClientError`` means the resource does not exist.

    Treating a missing resource as an expected state lets teardown be idempotent:
    running it twice does not fail the second time.

    Args:
        error: The botocore ``ClientError`` raised by an ``s3vectors`` call.

    Returns:
        True if the error represents a not-found condition, else False.
    """
    error_info = error.response.get("Error", {}) if error.response else {}
    code = str(error_info.get("Code", ""))
    message = str(error_info.get("Message", "")).lower()
    if code in ("NotFoundException", "ResourceNotFoundException", "NoSuchBucket"):
        return True
    return "not found" in message or "does not exist" in message


def delete_resources(config: Config, client: Optional[Any] = None) -> None:
    """Delete the sample's vector index and then its vector bucket.

    The index is deleted first because a vector bucket cannot be removed while it
    still holds an index. Deleting the index also removes every vector stored in
    it. If either resource is already gone, a notice is printed and teardown
    continues.

    Args:
        config: Resolved configuration supplying the bucket name, index name, and
            AWS region.
        client: Optional pre-built ``s3vectors`` client (useful for testing). When
            omitted, a client is created for ``config.aws_region``.
    """
    if client is None:
        client = boto3.client("s3vectors", region_name=config.aws_region)

    # 1) Delete the index (and with it, the stored vectors).
    try:
        client.delete_index(
            vectorBucketName=config.vector_bucket_name,
            indexName=config.index_name,
        )
        print(f"Deleted vector index '{config.index_name}'.")
    except ClientError as err:
        if _is_not_found_error(err):
            print(f"Vector index '{config.index_name}' not found; skipping.")
        else:
            raise

    # 2) Delete the now-empty bucket.
    try:
        client.delete_vector_bucket(vectorBucketName=config.vector_bucket_name)
        print(f"Deleted vector bucket '{config.vector_bucket_name}'.")
    except ClientError as err:
        if _is_not_found_error(err):
            print(
                f"Vector bucket '{config.vector_bucket_name}' not found; skipping."
            )
        else:
            raise


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: confirm intent, then delete the index and bucket.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``). Supports
            ``--yes`` to skip the interactive confirmation.
    """
    parser = argparse.ArgumentParser(
        description="Delete the S3 Vectors index and bucket used by this sample."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args(argv)

    config = load_config()

    print("This will permanently delete:")
    print(f"  index  : {config.index_name}")
    print(f"  bucket : {config.vector_bucket_name}")
    print(f"  region : {config.aws_region}")
    print("All vectors stored in the index will be removed.")

    if not args.yes:
        answer = input("Type 'delete' to continue: ").strip().lower()
        if answer != "delete":
            print("Aborted. Nothing was deleted.")
            return

    delete_resources(config)


if __name__ == "__main__":
    main()
