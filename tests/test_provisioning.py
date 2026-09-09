# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Live provisioning integration test for ``provision.py``.

Unlike the unit tests in ``tests/test_create_resources.py`` (which use mocked
clients), this test provisions a *real* S3 Vectors bucket and index against live
AWS and reads the index configuration back to confirm it matches the settings
Titan Text Embeddings v2 requires (Requirements 4.1, 4.2, 4.3):

    * dimension       == 1024
    * distanceMetric  == "cosine"
    * dataType        == "float32"
    * ``source_text`` is a non-filterable metadata key

Because it touches live AWS (creating resources and incurring calls), it is
gated behind the ``RUN_INTEGRATION_TESTS`` environment flag. Without the flag the
test is skipped, so the default test run stays fast and offline. To run it:

    RUN_INTEGRATION_TESTS=1 pytest tests/test_provisioning.py -q

All AWS client construction and calls live inside the test function so that mere
collection of this module never touches the network.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

# Skip the entire test unless the integration flag is explicitly set. This keeps
# ``pytest`` collection offline and prevents accidental live-AWS calls.
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="integration test; set RUN_INTEGRATION_TESTS=1 to run against live AWS",
)


def _extract_index(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return the index-configuration mapping from a get_index response.

    The ``s3vectors`` ``get_index`` response nests the configuration under an
    ``index`` key, but we read it defensively: if that key is absent we fall back
    to the top-level mapping so the assertions still find the values.

    Args:
        response: The raw dict returned by ``s3vectors`` ``get_index``.

    Returns:
        The mapping that holds ``dimension``, ``distanceMetric``, ``dataType``,
        and ``metadataConfiguration``.
    """
    index = response.get("index")
    if isinstance(index, dict):
        return index
    return response


def _non_filterable_keys(index: Dict[str, Any]) -> List[str]:
    """Return the configured non-filterable metadata keys, defensively.

    Reads ``metadataConfiguration.nonFilterableMetadataKeys`` and tolerates a
    missing ``metadataConfiguration`` block by returning an empty list.

    Args:
        index: The index-configuration mapping from ``get_index``.

    Returns:
        The list of non-filterable metadata key names (possibly empty).
    """
    metadata_config = index.get("metadataConfiguration") or {}
    return list(metadata_config.get("nonFilterableMetadataKeys") or [])


def test_provisioning_creates_index_with_titan_configuration() -> None:
    """Provision live resources and verify the index matches Titan v2 settings.

    Requirements 4.1, 4.2, 4.3: the provisioner creates the vector bucket and a
    1024-dimension, cosine, float32 index that stores ``source_text`` as a
    non-filterable metadata key. This creates the real resources (fail-soft on
    already-exists), then reads the index back to assert those four properties.
    """
    import boto3

    from provision import create_resources
    from config import load_config

    config = load_config()
    client = boto3.client("s3vectors", region_name=config.aws_region)

    # Provision the bucket + index (idempotent / fail-soft on already-exists).
    create_resources(config, client=client)

    # Read the index configuration back from S3 Vectors.
    response = client.get_index(
        vectorBucketName=config.vector_bucket_name,
        indexName=config.index_name,
    )
    index = _extract_index(response)

    # Req 4.2: Titan Text Embeddings v2 uses 1024-dim, cosine, float32 vectors.
    assert index.get("dimension") == 1024
    assert index.get("distanceMetric") == "cosine"
    assert index.get("dataType") == "float32"

    # Req 4.3: the large chunk text is stored but not indexed for filtering.
    assert "source_text" in _non_filterable_keys(index)
