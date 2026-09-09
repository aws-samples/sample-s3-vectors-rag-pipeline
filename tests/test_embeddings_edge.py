# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Edge-case tests for the embedding wrapper's error handling.

These tests cover the model-access failure path of ``embed_text`` (Task 4.4):
when Bedrock denies access to the embedding model, the wrapper must translate the
botocore ``AccessDeniedException`` into a :class:`~errors.ModelAccessError`
that names the affected model (Requirements 3.4, 8.1).
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from config import Config
from errors import ModelAccessError
from embeddings import embed_text


def _make_config(embedding_model_id: str = "amazon.titan-embed-text-v2:0") -> Config:
    """Build a Config with a known embedding model id for assertions."""
    return Config(
        vector_bucket_name="test-bucket",
        index_name="rag-documents",
        embedding_model_id=embedding_model_id,
        generation_model_id="us.anthropic.claude-sonnet-4-6",
        aws_region="us-east-1",
        top_k=5,
    )


class _AccessDeniedClient:
    """Mock bedrock-runtime client whose invoke_model raises AccessDeniedException."""

    def invoke_model(self, **kwargs: Any):
        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "You don't have access to the model with the specified id.",
            }
        }
        raise ClientError(error_response, "InvokeModel")


def test_access_denied_raises_model_access_error_naming_model() -> None:
    # Requirements 3.4, 8.1: a Bedrock access-denied on embedding is translated
    # into a ModelAccessError that identifies the affected model.
    model_id = "amazon.titan-embed-text-v2:0"
    config = _make_config(embedding_model_id=model_id)

    with pytest.raises(ModelAccessError) as exc_info:
        embed_text("some text to embed", config, client=_AccessDeniedClient())

    err = exc_info.value
    # The error must name the affected model, both structurally and in its message.
    assert err.model_id == model_id
    assert model_id in str(err)


def test_access_denied_names_custom_configured_model() -> None:
    # The named model must track the configured embedding model id, not a constant.
    model_id = "amazon.some-other-embed-model-v9:0"
    config = _make_config(embedding_model_id=model_id)

    with pytest.raises(ModelAccessError) as exc_info:
        embed_text("hello world", config, client=_AccessDeniedClient())

    assert exc_info.value.model_id == model_id
    assert model_id in str(exc_info.value)


def test_access_denied_preserves_original_cause() -> None:
    # The wrapper chains from the botocore error (raise ... from err) so the
    # underlying cause remains available for debugging.
    config = _make_config()

    with pytest.raises(ModelAccessError) as exc_info:
        embed_text("text", config, client=_AccessDeniedClient())

    assert isinstance(exc_info.value.__cause__, ClientError)
