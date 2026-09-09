# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Edge/unit tests for query wiring and error handling (`query.py`).

These tests cover three narrow paths of the query flow (Task 8.12):

- The interactive prompt path of ``get_question`` when no question argument is
  supplied on the command line (Requirement 6.2).
- ``retrieve`` reporting that no relevant context was found when the vector
  query returns zero results (Requirement 6.9).
- ``generate_answer`` translating a Bedrock ``AccessDeniedException`` into a
  :class:`~errors.ModelAccessError` that names the configured generation
  model (Requirement 8.1).

They are example-based edge tests (not property-based).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from config import Config
from errors import ModelAccessError
from query import generate_answer, get_question, retrieve


def _make_config(
    generation_model_id: str = "us.anthropic.claude-sonnet-4-6",
    top_k: int = 5,
) -> Config:
    """Build a Config with concrete values for assertions."""
    return Config(
        vector_bucket_name="test-bucket",
        index_name="rag-documents",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id=generation_model_id,
        aws_region="us-east-1",
        top_k=top_k,
    )


def test_get_question_uses_interactive_prompt_when_no_arg() -> None:
    # Requirement 6.2: with no question argument on the command line, the user is
    # prompted interactively and the entered line is returned verbatim.
    with patch("builtins.input", return_value="typed question") as mock_input:
        result = get_question(["prog"])

    assert result == "typed question"
    mock_input.assert_called_once()


def test_retrieve_reports_no_context_on_zero_results(capsys) -> None:
    # Requirement 6.9: when the vector query returns no vectors, retrieve reports
    # that no relevant context was found and returns an empty list.
    config = _make_config()

    s3v_client = MagicMock()
    s3v_client.query_vectors.return_value = {"vectors": []}

    result = retrieve([0.0] * 1024, config, s3v_client)

    assert result == []
    out = capsys.readouterr().out
    assert "no relevant context" in out.lower()


def test_generate_answer_translates_access_denied_to_model_access_error() -> None:
    # Requirement 8.1: a Bedrock AccessDeniedException on generation is translated
    # into a ModelAccessError naming the configured generation model.
    model_id = "us.anthropic.claude-sonnet-4-6"
    config = _make_config(generation_model_id=model_id)

    error_response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "You don't have access to the model with the specified id.",
        }
    }

    brt_client = MagicMock()
    brt_client.converse.side_effect = ClientError(error_response, "Converse")

    with pytest.raises(ModelAccessError) as exc_info:
        generate_answer("prompt", config, brt_client)

    err = exc_info.value
    assert err.model_id == model_id
    assert model_id in str(err)
