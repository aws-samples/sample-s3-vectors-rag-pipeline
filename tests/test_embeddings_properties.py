# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the embedding wrapper.

These tests exercise the request-shaping logic in ``embeddings.embed_text``
across many generated inputs using Hypothesis. A mocked ``bedrock-runtime`` client
is injected so the tests run fast and offline while still exercising the real logic
that selects the model id and builds the request body.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from embeddings import embed_text

# A fixed 1024-element float embedding the mock returns for every call. The value
# itself is irrelevant to Property 9 (which is about the request, not the vector).
_FAKE_EMBEDDING = [0.0] * 1024


def _make_mock_client() -> MagicMock:
    """Build a mock bedrock-runtime client whose invoke_model returns a Titan-shaped body."""
    client = MagicMock()

    def _invoke_model(**kwargs: Any) -> dict:
        body = MagicMock()
        body.read.return_value = json.dumps({"embedding": _FAKE_EMBEDDING})
        return {"body": body}

    client.invoke_model.side_effect = _invoke_model
    return client


def _make_config(model_id: str) -> Config:
    """Build a Config with an arbitrary embedding model id (other fields fixed)."""
    return Config(
        vector_bucket_name="bucket",
        index_name="index",
        embedding_model_id=model_id,
        generation_model_id="gen-model",
        aws_region="us-east-1",
        top_k=5,
    )


# Text that is safe to embed: no null bytes, bounded size.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=200,
)

# Non-empty model identifiers drawn from arbitrary printable text.
_model_id = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=60,
)


# Feature: sample-s3-vectors-rag-pipeline, Property 9: Embedding is consistent and uses the configured model
@settings(max_examples=100)
@given(text=_text, model_id=_model_id)
def test_embedding_uses_configured_model_and_is_consistent(
    text: str, model_id: str
) -> None:
    """For any text and model id, embed_text invokes Bedrock with the configured
    model id, and the ingest and query paths issue an identical request."""
    config = _make_config(model_id)

    # Simulate the ingestion path and the query path embedding the same text.
    client = _make_mock_client()
    embed_text(text, config, client=client)  # ingestion path
    embed_text(text, config, client=client)  # query path

    assert client.invoke_model.call_count == 2

    ingest_call = client.invoke_model.call_args_list[0]
    query_call = client.invoke_model.call_args_list[1]

    # The model id used is exactly the one from the Config (Requirement 3.2).
    assert ingest_call.kwargs["modelId"] == config.embedding_model_id
    assert query_call.kwargs["modelId"] == config.embedding_model_id

    # The two requests are identical in every keyword argument, so ingestion and
    # querying embed the same text with the same model id and body (Requirement 3.3).
    assert ingest_call.kwargs == query_call.kwargs

    # The request body carries the text under the Titan "inputText" key.
    assert json.loads(ingest_call.kwargs["body"]) == {"inputText": text}
