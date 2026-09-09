# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""End-to-end integration test for the S3 AI Vectors RAG pipeline.

This module exercises the full pipeline against **live AWS resources** — S3
Vectors for storage/retrieval and Amazon Bedrock for embedding and generation.
It ingests a small fixture document, runs a query, and asserts that a non-empty
answer is produced together with at least one source attribution
(Requirements 10.1, 10.2).

Because it makes real AWS calls (and therefore requires provisioned resources,
credentials, and Bedrock model access), the test is **gated behind the
``RUN_INTEGRATION_TESTS`` environment variable**. Without that flag the test is
skipped, so the default offline ``pytest`` run stays fast and dependency-free.
All AWS interactions live inside the test function, so this module is safe to
import and collect without touching AWS.

To run it against live AWS (from the repository root)::

    RUN_INTEGRATION_TESTS=1 pytest tests/test_pipeline.py -q

On Windows PowerShell::

    $env:RUN_INTEGRATION_TESTS = "1"; pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

# A short fixture document whose content is specific enough that a grounded
# answer must draw on it (rather than the model's prior knowledge).
_FIXTURE_FILENAME = "integration_fixture.md"
_FIXTURE_TEXT = (
    "# Zephyr Widget Maintenance\n\n"
    "The Zephyr widget must be lubricated every 42 days using grade-7 "
    "silicone oil. The recommended operating temperature for the Zephyr "
    "widget is between 15 and 30 degrees Celsius. If the indicator light "
    "turns violet, the widget should be power-cycled twice before use.\n"
)
_FIXTURE_QUESTION = "How often must the Zephyr widget be lubricated?"


@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="integration test; set RUN_INTEGRATION_TESTS=1 to run against live AWS",
)
def test_end_to_end_pipeline_returns_answer_and_sources() -> None:
    """Ingest a fixture, query it, and assert an answer plus a source attribution.

    Exercises chunking, embedding, storage, retrieval, and answer generation end
    to end against provisioned S3 Vectors resources with Bedrock model access
    enabled (Requirement 10.1). Asserts that the query yields a non-empty answer
    and that at least one ``source_file`` attribution is returned from the
    retrieved chunks (Requirement 10.2).
    """
    import boto3

    from provision import create_resources
    from config import load_config
    from ingest import build_vectors, store_vectors
    from query import build_prompt, generate_answer, retrieve
    from embeddings import embed_text

    config = load_config()

    # Real AWS clients: s3vectors for storage/retrieval, bedrock-runtime for
    # embedding and generation.
    s3v_client = boto3.client("s3vectors", region_name=config.aws_region)
    brt_client = boto3.client("bedrock-runtime", region_name=config.aws_region)

    # Ensure the vector bucket and index exist (idempotent; fails soft if the
    # resources are already provisioned).
    create_resources(config, s3v_client)

    # Write the fixture into data/sample/ so it flows through the normal
    # ingestion path, then clean it up afterwards.
    fixture_path = Path("data/sample") / _FIXTURE_FILENAME
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(_FIXTURE_TEXT, encoding="utf-8")

    try:
        # --- Ingest: chunk -> embed -> store ---
        records = build_vectors(fixture_path, config, brt_client)
        assert records, "fixture should produce at least one vector record"
        store_vectors(records, config, s3v_client)

        # --- Query: embed question -> retrieve -> build prompt -> generate ---
        question_vector = embed_text(_FIXTURE_QUESTION, config, brt_client)
        chunks = retrieve(question_vector, config, s3v_client)

        # Retrieval must return context, and every chunk must carry a source.
        assert chunks, "query should retrieve at least one chunk"
        source_files: List[str] = [
            chunk.source_file for chunk in chunks if chunk.source_file
        ]
        assert source_files, "at least one source attribution must be returned"

        contexts = [chunk.source_text for chunk in chunks]
        prompt = build_prompt(_FIXTURE_QUESTION, contexts)
        answer = generate_answer(prompt, config, brt_client)

        # The generated answer must be a non-empty string (Requirement 10.2).
        assert isinstance(answer, str)
        assert answer.strip(), "generation should return a non-empty answer"
    finally:
        # Best-effort cleanup of the fixture file so repeated runs stay clean.
        if fixture_path.exists():
            fixture_path.unlink()
