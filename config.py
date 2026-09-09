# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Central configuration for the Amazon S3 Vectors RAG pipeline sample.

SAMPLE ONLY: this project is intended for educational purposes and is not
intended for use in production environments.

This module is the single source of truth for configuration. Every other module
reads its settings from here rather than reading environment variables directly,
which is what makes swapping the generation model a config-only change.

Each setting has a documented default that can be overridden by an environment
variable of the same name (see the ``DEFAULTS`` map and the ``Config`` fields).
Values may also be supplied through a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Union

from dotenv import load_dotenv

# Documented defaults for every configuration value. Environment variables of the
# same name override these at load time (see ``load_config``).
DEFAULTS: Dict[str, Union[str, int]] = {
    "VECTOR_BUCKET_NAME": "sample-rag-vectors",
    "INDEX_NAME": "rag-documents",
    # Ingestion and query must use the same embedding model. Vectors are only
    # comparable when produced by the same model, so changing this after
    # documents are indexed requires clearing the index and re-ingesting.
    # Adding more documents with the same model is always fine.
    "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
    "GENERATION_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "AWS_REGION": "us-east-1",
    "TOP_K": 5,
}

# Commented alternative GENERATION_MODEL_ID values.
# To use one of these, set the GENERATION_MODEL_ID environment variable (or edit
# the default above). All are Bedrock models supported by the Converse API, so no
# query-logic changes are required to swap them in. Model availability varies by
# account and region, so confirm the id with list_inference_profiles first.
#   Grok 4.6        -> "us.xai.grok-4.6"
#   DeepSeek        -> "us.deepseek.r1-v1:0"
#   Amazon Nova Pro -> "us.amazon.nova-pro-v1:0"

# S3 Vectors metadata limits worth knowing before you add fields: up to 40 KB of
# total metadata per vector, but only 2 KB of that can be filterable, and an index
# allows at most 10 non-filterable metadata keys.

# Chunking parameters used during ingestion. Chunks are sized in approximate
# whitespace tokens, with an overlap so context is not lost at chunk boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of resolved configuration values.

    Attributes:
        vector_bucket_name: Name of the S3 Vector bucket that holds indexes.
        index_name: Name of the vector index within the bucket.
        embedding_model_id: Bedrock model id used to embed text (Titan v2).
        generation_model_id: Bedrock model id used to generate answers.
        aws_region: AWS region used for all AWS clients.
        top_k: Number of most-similar chunks retrieved per query.
    """

    vector_bucket_name: str
    index_name: str
    embedding_model_id: str
    generation_model_id: str
    aws_region: str
    top_k: int


def load_config() -> Config:
    """Load configuration, letting environment variables override defaults.

    Each value resolves via ``os.getenv(NAME, DEFAULT)`` using the keys in
    ``DEFAULTS``. ``TOP_K`` is cast to ``int`` because environment variables are
    always strings.

    Returns:
        A frozen :class:`Config` populated with the resolved values.
    """
    # Load a local .env file if present. Existing environment variables take
    # precedence (override=False), so real env vars still win over the file.
    load_dotenv(override=False)

    return Config(
        vector_bucket_name=os.getenv(
            "VECTOR_BUCKET_NAME", str(DEFAULTS["VECTOR_BUCKET_NAME"])
        ),
        index_name=os.getenv("INDEX_NAME", str(DEFAULTS["INDEX_NAME"])),
        embedding_model_id=os.getenv(
            "EMBEDDING_MODEL_ID", str(DEFAULTS["EMBEDDING_MODEL_ID"])
        ),
        generation_model_id=os.getenv(
            "GENERATION_MODEL_ID", str(DEFAULTS["GENERATION_MODEL_ID"])
        ),
        aws_region=os.getenv("AWS_REGION", str(DEFAULTS["AWS_REGION"])),
        top_k=int(os.getenv("TOP_K", str(DEFAULTS["TOP_K"]))),
    )
