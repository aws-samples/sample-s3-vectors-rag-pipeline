# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for metadata filtering on the query path (`query.py`).

Cover the three filter behaviors added for S3 Vectors metadata filtering:

- ``parse_filter_args`` turns ``KEY=VALUE`` CLI arguments into a mapping and
  rejects malformed input.
- ``build_filter`` produces the S3 Vectors filter expression, using ``$eq`` for
  one condition and ``$and`` for several, and ``None`` when there is nothing to
  filter on.
- ``retrieve`` passes ``filter`` to ``query_vectors`` only when a filter is
  given, and issues the original unfiltered request shape otherwise.

These are example-based unit tests with a fake S3 Vectors client, no live AWS.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from config import Config
from query import build_filter, parse_filter_args, retrieve


def _make_config() -> Config:
    return Config(
        vector_bucket_name="test-bucket",
        index_name="rag-documents",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region="us-east-1",
        top_k=5,
    )


class FakeS3Vectors:
    """Records the kwargs of the last query_vectors call and returns one vector."""

    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}

    def query_vectors(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = kwargs
        return {
            "vectors": [
                {
                    "key": "employee-handbook.md::0",
                    "distance": 0.12,
                    "metadata": {
                        "source_file": "employee-handbook.md",
                        "chunk_index": 0,
                        "category": "hr",
                        "source_text": "Part-time PTO accrues at 1 hour per 30 worked.",
                    },
                }
            ]
        }


# --- parse_filter_args -------------------------------------------------------


def test_parse_filter_args_none_returns_empty() -> None:
    assert parse_filter_args(None) == {}


def test_parse_filter_args_single_pair() -> None:
    assert parse_filter_args(["category=hr"]) == {"category": "hr"}


def test_parse_filter_args_multiple_pairs() -> None:
    parsed = parse_filter_args(["category=hr", "source_file=employee-handbook.md"])
    assert parsed == {"category": "hr", "source_file": "employee-handbook.md"}


def test_parse_filter_args_value_may_contain_equals() -> None:
    # Only the first '=' separates key from value.
    assert parse_filter_args(["key=a=b"]) == {"key": "a=b"}


def test_parse_filter_args_trims_whitespace() -> None:
    assert parse_filter_args([" category = hr "]) == {"category": "hr"}


@pytest.mark.parametrize("bad", ["category", "=hr", "category=", "   "])
def test_parse_filter_args_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_filter_args([bad])


# --- build_filter ------------------------------------------------------------


def test_build_filter_none_and_empty() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_build_filter_single_uses_eq() -> None:
    assert build_filter({"category": "hr"}) == {"category": {"$eq": "hr"}}


def test_build_filter_multiple_uses_and() -> None:
    expr = build_filter({"category": "hr", "source_file": "employee-handbook.md"})
    assert expr == {
        "$and": [
            {"category": {"$eq": "hr"}},
            {"source_file": {"$eq": "employee-handbook.md"}},
        ]
    }


# --- retrieve ----------------------------------------------------------------


def test_retrieve_passes_filter_when_provided() -> None:
    s3v = FakeS3Vectors()
    metadata_filter = build_filter({"category": "hr"})

    chunks = retrieve([0.0] * 1024, _make_config(), s3v, metadata_filter)

    assert "filter" in s3v.last_kwargs
    assert s3v.last_kwargs["filter"] == {"category": {"$eq": "hr"}}
    assert chunks[0].category == "hr"
    assert chunks[0].source_file == "employee-handbook.md"


def test_retrieve_omits_filter_when_absent() -> None:
    s3v = FakeS3Vectors()

    retrieve([0.0] * 1024, _make_config(), s3v)

    # An unfiltered call must not carry a filter key at all, so the request shape
    # is identical to the pre-filtering behavior.
    assert "filter" not in s3v.last_kwargs
    assert s3v.last_kwargs["topK"] == 5
    assert s3v.last_kwargs["returnMetadata"] is True


def test_retrieve_default_argument_is_unfiltered() -> None:
    s3v = FakeS3Vectors()

    # Calling without the metadata_filter argument at all behaves as unfiltered.
    retrieve([0.0] * 1024, _make_config(), s3v)

    assert "filter" not in s3v.last_kwargs
