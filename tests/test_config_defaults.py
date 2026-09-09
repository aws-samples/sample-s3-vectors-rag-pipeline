# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the documented configuration defaults.

These tests pin down the exact default value of every configuration field when
no environment variable is set. Each assertion maps directly to an acceptance
criterion in Requirement 1 (see references below), so a change to any documented
default will surface here.
"""

from __future__ import annotations

import pytest

from config import load_config

# The six configuration environment variables that can override a default.
_CONFIG_ENV_VARS = (
    "VECTOR_BUCKET_NAME",
    "INDEX_NAME",
    "EMBEDDING_MODEL_ID",
    "GENERATION_MODEL_ID",
    "AWS_REGION",
    "TOP_K",
)


@pytest.fixture
def config_no_env(monkeypatch: pytest.MonkeyPatch):
    """Load config with all six configuration env vars guaranteed unset."""
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return load_config()


def test_default_vector_bucket_name(config_no_env) -> None:
    # Requirement 1.1
    assert config_no_env.vector_bucket_name == "sample-rag-vectors"


def test_default_index_name(config_no_env) -> None:
    # Requirement 1.2
    assert config_no_env.index_name == "rag-documents"


def test_default_embedding_model_id(config_no_env) -> None:
    # Requirement 1.3
    assert config_no_env.embedding_model_id == "amazon.titan-embed-text-v2:0"


def test_default_generation_model_id(config_no_env) -> None:
    # Requirement 1.4
    assert (
        config_no_env.generation_model_id
        == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )


def test_default_aws_region(config_no_env) -> None:
    # Requirement 1.5
    assert config_no_env.aws_region == "us-east-1"


def test_default_top_k(config_no_env) -> None:
    # Requirement 1.6
    assert config_no_env.top_k == 5
    # TOP_K must be an int, not the string form of the default.
    assert isinstance(config_no_env.top_k, int)
