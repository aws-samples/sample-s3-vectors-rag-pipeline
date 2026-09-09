# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the query engine (`query.py`).

Each test corresponds to a numbered correctness property from the
`sample-s3-vectors-rag-pipeline` design document and runs a minimum of 100 examples.

Later tasks append additional property tests to this module, so imports are
collected at the top and each test uses a distinct, property-numbered name.
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from query import (
    RetrievedChunk,
    answer_question,
    build_prompt,
    generate_answer,
    get_question,
    retrieve,
)


# Feature: sample-s3-vectors-rag-pipeline, Property 10: The CLI argument is used verbatim as the question
@settings(max_examples=100)
@given(question=st.text(min_size=1))
def test_property_10_cli_question_passthrough(question: str) -> None:
    """The CLI argument is used verbatim as the question.

    For any non-empty command-line argument, ``get_question`` returns that
    argument unchanged when it is supplied as ``argv[1]``.

    Validates: Requirements 6.1
    """
    assert get_question(["prog", question]) == question


# Feature: sample-s3-vectors-rag-pipeline, Property 11: Retrieval requests top_k results with metadata
@settings(max_examples=100)
@given(top_k=st.integers(min_value=1, max_value=100))
def test_property_11_retrieval_request_params(top_k: int) -> None:
    """Retrieval requests top_k results with metadata.

    For any configured ``TOP_K`` value, ``retrieve`` calls ``query_vectors``
    with ``topK`` equal to that value and with metadata return enabled.

    Validates: Requirements 6.4
    """
    config = Config(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding_model_id="test-embed-model",
        generation_model_id="test-gen-model",
        aws_region="us-east-1",
        top_k=top_k,
    )

    s3v_client = MagicMock()
    s3v_client.query_vectors.return_value = {"vectors": []}

    retrieve([0.1] * 1024, config, s3v_client)

    s3v_client.query_vectors.assert_called_once()
    _, kwargs = s3v_client.query_vectors.call_args
    assert kwargs["topK"] == config.top_k
    assert kwargs["returnMetadata"] is True


# Feature: sample-s3-vectors-rag-pipeline, Property 12: Retrieval collects the text of every returned chunk
@settings(max_examples=100)
@given(
    vectors=st.lists(
        st.fixed_dictionaries(
            {
                "key": st.text(),
                "distance": st.floats(
                    min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
                ),
                "metadata": st.fixed_dictionaries(
                    {
                        "source_file": st.text(),
                        "chunk_index": st.integers(min_value=0, max_value=10_000),
                        "source_text": st.text(),
                    }
                ),
            }
        ),
        min_size=1,
        max_size=25,
    )
)
def test_property_12_collect_chunk_text(vectors: list) -> None:
    """Retrieval collects the text of every returned chunk.

    For any set of vectors returned by ``query_vectors``, ``retrieve`` collects
    the ``source_text`` value of every returned vector, preserving order.

    Validates: Requirements 6.5
    """
    config = Config(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding_model_id="test-embed-model",
        generation_model_id="test-gen-model",
        aws_region="us-east-1",
        top_k=len(vectors),
    )

    s3v_client = MagicMock()
    s3v_client.query_vectors.return_value = {"vectors": vectors}

    chunks = retrieve([0.1] * 1024, config, s3v_client)

    collected = [chunk.source_text for chunk in chunks]
    expected = [vector["metadata"]["source_text"] for vector in vectors]
    assert collected == expected


# Feature: sample-s3-vectors-rag-pipeline, Property 13: The prompt is grounded in the retrieved context
@settings(max_examples=100)
@given(
    question=st.text(),
    contexts=st.lists(st.text(), max_size=25),
)
def test_property_13_prompt_grounding(question: str, contexts: list[str]) -> None:
    """The prompt is grounded in the retrieved context.

    For any question and set of retrieved context texts, the built prompt
    contains the question, contains every retrieved context text, and includes
    an instruction to answer using only the provided context and to state when
    the context does not contain the answer.

    Validates: Requirements 6.6
    """
    prompt = build_prompt(question, contexts)

    # The prompt contains the question verbatim.
    assert question in prompt

    # The prompt contains every retrieved context text.
    for context in contexts:
        assert context in prompt

    # The prompt includes a grounding instruction: answer using ONLY the
    # provided context, and state when the answer is not present in it.
    lowered = prompt.lower()
    assert "only" in lowered
    assert "not present" in lowered
    assert "context" in lowered


# Feature: sample-s3-vectors-rag-pipeline, Property 14: Generation is model-swappable via configuration only
@settings(max_examples=100)
@given(
    prompt=st.text(),
    generation_model_id=st.text(min_size=1),
)
def test_property_14_model_swappability(prompt: str, generation_model_id: str) -> None:
    """Generation is model-swappable via configuration only.

    For any prompt and any configured ``generation_model_id``, ``generate_answer``
    calls the Bedrock Converse API with ``modelId`` equal to the configured model
    id and sends the identical, model-agnostic Converse request, with no branching
    on the model id. Swapping the model is therefore a configuration-only change.

    Validates: Requirements 6.7, 7.1, 7.2
    """
    config = Config(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding_model_id="test-embed-model",
        generation_model_id=generation_model_id,
        aws_region="us-east-1",
        top_k=5,
    )

    brt_client = MagicMock()
    brt_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "answer"}]}}
    }

    generate_answer(prompt, config, brt_client)

    brt_client.converse.assert_called_once()
    _, kwargs = brt_client.converse.call_args

    # The configured model id is used verbatim as the request's modelId.
    assert kwargs["modelId"] == config.generation_model_id

    # The Converse request is the same model-agnostic shape regardless of the
    # configured model id (no branching on model id).
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": prompt}]}]
    assert kwargs["inferenceConfig"] == {"maxTokens": 1024}


# Feature: sample-s3-vectors-rag-pipeline, Property 15: Answers are attributed to their source files
@settings(max_examples=100)
@given(
    question=st.text(min_size=1),
    source_files=st.lists(
        # Draw from a small pool of names so duplicates occur naturally, letting
        # the test exercise the deduplication of source_file values.
        st.sampled_from(["a.txt", "b.txt", "c.txt", "notes.md", "doc.pdf", "a.txt"]),
        min_size=1,
        max_size=25,
    ),
)
def test_property_15_source_attribution(
    question: str, source_files: list[str]
) -> None:
    """Answers are attributed to their source files.

    For any set of retrieved chunks, ``answer_question`` prints the generated
    answer together with the deduplicated set of ``source_file`` values drawn
    from those chunks.

    This targets ``answer_question`` rather than ``main`` so the property is
    isolated from command-line parsing. ``main`` is a thin argparse wrapper
    around this function, and a question that begins with a hyphen is parsed as
    a CLI flag (standard argparse behavior), which is unrelated to attribution.

    Validates: Requirements 6.8
    """
    answer = "ANSWER-SENTINEL"

    # Build a non-empty list of retrieved chunks whose source_file values may
    # include duplicates; the query flow must still attribute the answer to the
    # distinct set of files.
    chunks = [
        RetrievedChunk(
            key=f"{source_file}::{index}",
            distance=0.1,
            source_file=source_file,
            chunk_index=index,
            category="hr",
            source_text=f"context text {index}",
        )
        for index, source_file in enumerate(source_files)
    ]

    # Hypothesis reruns this function many times per test, and pytest's capsys
    # fixture does not reset between examples, so capture stdout explicitly with
    # a fresh buffer for each example instead.
    config = Config(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding_model_id="test-embed-model",
        generation_model_id="test-gen-model",
        aws_region="us-east-1",
        top_k=len(chunks),
    )

    buffer = io.StringIO()
    with patch("query.embed_text", return_value=[0.0] * 1024), patch(
        "query.retrieve", return_value=chunks
    ), patch("query.generate_answer", return_value=answer), contextlib.redirect_stdout(
        buffer
    ):
        answer_question(question, config, MagicMock(), MagicMock())

    captured = buffer.getvalue()

    # The generated answer is printed.
    assert answer in captured

    # Every distinct source_file value backing the answer is printed.
    for source_file in set(source_files):
        assert source_file in captured
