# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Streamlit UI for the Amazon S3 Vectors RAG pipeline sample.

A single-page internal knowledge assistant over the same query pipeline used by
``query.py``. It embeds a natural-language question, retrieves the most similar
chunks from Amazon S3 Vectors, and asks a foundation model on Amazon Bedrock to
answer using only that retrieved context. All embedding, retrieval, prompt, and
generation logic is imported from the core modules; this file only handles
presentation, timing, and error display.

Run it locally with:

    streamlit run app.py

then open http://localhost:8501.

Sample only: this is intended for educational purposes and is not intended for
use in production environments.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, NoCredentialsError

from config import Config, load_config
from embeddings import embed_text
from errors import RagError
from provision import INDEX_DIMENSION, INDEX_DISTANCE_METRIC
from query import (
    RetrievedChunk,
    build_filter,
    build_prompt,
    generate_answer,
    retrieve,
)

# Categories a query can be filtered to, matching the FILENAME_CATEGORY mapping
# in ingest.py. "All documents" leaves retrieval unfiltered.
CATEGORY_CHOICES: List[str] = [
    "All documents",
    "hr",
    "operations",
    "finance",
    "marketing",
    "it",
]

# Generation models the sidebar can swap between at runtime. Each label maps to a
# Bedrock model id; selecting one only changes GENERATION_MODEL_ID, because every
# option is called through the same Bedrock Converse API (see query.py).
MODEL_CHOICES: Dict[str, str] = {
    "Claude Sonnet 4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Grok 4.6": "us.xai.grok-4.6",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0",
    "DeepSeek R1": "us.deepseek.r1-v1:0",
}

# Questions answerable only from the bundled documents for Wally's Coffee Co., a
# fictional company. Because the company is invented, the model cannot answer from
# training data, so a correct answer demonstrates that retrieval worked. These
# match the example queries in the README.
SUGGESTED_QUESTIONS: List[str] = [
    "What is the PTO policy for part-time employees?",
    "Where do we source our Ethiopian beans and what is the roast profile?",
    "How did online subscriptions perform in Q3?",
    "How do I fix the POS system when it freezes?",
]

# A question no bundled document answers. Included so the grounding instruction in
# build_prompt() can be demonstrated in one click: the model should say the answer
# is not in the provided context rather than inventing a policy.
OUT_OF_SCOPE_QUESTION = "What is our parental leave policy?"

# Number of characters of chunk text shown as a preview in Referenced Documents.
# Long enough to see the sentence that earned the match, not the whole chunk.
PREVIEW_CHARS = 240


def set_question(question: str) -> None:
    """Callback that loads a suggested question into the input box.

    Runs before the input widget is re-instantiated on the next rerun, so
    assigning to the widget's ``st.session_state`` key updates the field.

    Args:
        question: The suggested question text to place in the input.
    """
    st.session_state["question"] = question


def get_clients(region: str) -> Dict[str, object]:
    """Return cached boto3 clients for the given region, creating them once.

    The ``bedrock-runtime`` and ``s3vectors`` clients are stored in
    ``st.session_state`` so they are reused across reruns instead of being
    rebuilt on every interaction. If the region changes, the clients are rebuilt.

    Args:
        region: AWS region for both clients.

    Returns:
        A mapping with ``"bedrock-runtime"`` and ``"s3vectors"`` clients.
    """
    if st.session_state.get("clients_region") != region:
        st.session_state["clients"] = {
            "bedrock-runtime": boto3.client("bedrock-runtime", region_name=region),
            "s3vectors": boto3.client("s3vectors", region_name=region),
        }
        st.session_state["clients_region"] = region
    return st.session_state["clients"]


def similarity_percent(distance: float) -> float:
    """Convert a cosine distance into a 0-100 similarity percentage.

    The index uses the cosine distance metric, where ``distance = 1 - cosine
    similarity``. This maps that back to a similarity share and clamps it to the
    ``0-100`` range for display.

    Args:
        distance: The cosine distance reported by S3 Vectors.

    Returns:
        A similarity percentage in ``[0, 100]``.
    """
    return max(0.0, min(1.0, 1.0 - distance)) * 100.0


def chunk_preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Return a single-line preview of chunk text, truncated to ``limit`` chars.

    Newlines are collapsed so each referenced document stays on one visual line.

    Args:
        text: The full chunk text.
        limit: Maximum number of characters to show.

    Returns:
        The collapsed, truncated preview.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def model_options(configured_model_id: str) -> Dict[str, str]:
    """Return the selectable models, including the configured one if it is custom.

    ``MODEL_CHOICES`` is a curated list, but ``GENERATION_MODEL_ID`` can be set to
    any Converse-capable Bedrock model. When the configured id is not one of the
    curated options it is added, so a ``.env`` override stays selectable instead of
    being silently replaced by the first curated entry.

    Args:
        configured_model_id: The generation model id resolved from configuration.

    Returns:
        A label-to-model-id mapping that always contains ``configured_model_id``.
    """
    if configured_model_id in MODEL_CHOICES.values():
        return MODEL_CHOICES
    return {f"{configured_model_id} (from config)": configured_model_id, **MODEL_CHOICES}


def render_sidebar(base_config: Config) -> tuple[str, int]:
    """Render the sidebar controls and return the chosen model id and top_k.

    Both controls are seeded from ``base_config`` rather than from hardcoded
    literals, so ``GENERATION_MODEL_ID`` and ``TOP_K`` set in the environment or a
    ``.env`` file are honored here exactly as they are by ``query.py``.

    Args:
        base_config: Configuration as resolved from defaults, environment
            variables, and ``.env``, before any sidebar override is applied.

    Returns:
        A tuple of ``(generation_model_id, top_k)`` reflecting the current
        sidebar selections.
    """
    st.sidebar.header("Developer Controls")
    st.sidebar.caption(
        "Pipeline settings for testing. An end user would not normally see these. "
        "Defaults come from config.py and any .env override."
    )

    choices = model_options(base_config.generation_model_id)
    labels = list(choices.keys())
    configured_label = next(
        label for label, model_id in choices.items()
        if model_id == base_config.generation_model_id
    )

    model_label = st.sidebar.selectbox(
        "Generation model",
        labels,
        index=labels.index(configured_label),
        help="Swaps GENERATION_MODEL_ID at runtime. All options are called "
        "through the same Bedrock Converse API, so nothing else changes.",
    )
    top_k = st.sidebar.slider(
        "Document chunks to retrieve",
        min_value=1,
        max_value=10,
        value=min(max(base_config.top_k, 1), 10),
        help="Sets topK on the query_vectors call, which is an upper bound rather "
        "than a guarantee. Approximate nearest-neighbor search can return fewer. "
        "More chunks give the model broader context but may include weaker "
        "matches.",
    )

    render_index_panel(base_config)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Sample application for educational purposes. Not for production use."
    )

    return choices[model_label], top_k


def render_index_panel(config: Config) -> None:
    """Show which S3 Vectors resource and index shape the app is querying.

    The vector bucket, index, and index shape are the S3 Vectors side of this
    sample, so a builder testing the app should be able to confirm what it is
    pointed at without reading the code. Dimension and distance metric are
    imported from ``provision.py`` so this panel cannot drift from the index that
    was actually created.

    Args:
        config: The resolved configuration naming the bucket, index, and region.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Vector Index")
    st.sidebar.caption(
        f"Bucket: `{config.vector_bucket_name}`  \n"
        f"Index: `{config.index_name}`  \n"
        f"Region: `{config.aws_region}`  \n"
        f"Shape: {INDEX_DIMENSION} dimensions, {INDEX_DISTANCE_METRIC} distance"
    )


def render_results(
    chunks: List[RetrievedChunk],
    answer: str,
    prompt: str,
    timings: Dict[str, float],
    config: Config,
) -> None:
    """Render the answer, referenced documents, and pipeline diagnostics.

    Args:
        chunks: Retrieved chunks, ordered closest match first.
        answer: The generated answer text.
        prompt: The grounded prompt that produced ``answer``, shown in the
            sidebar so the retrieval-to-prompt step is inspectable.
        timings: Milliseconds per stage, keyed ``embed``, ``search``,
            ``generate``, and ``total``.
        config: The resolved configuration, used to report which models ran.
    """
    # Center column: what an end user cares about, the answer and its sources.
    with st.container(border=True):
        st.markdown(answer)

    with st.expander(f"Referenced Documents ({len(chunks)})", expanded=False):
        for chunk in chunks:
            st.markdown(
                f"**{chunk.source_file}** &nbsp;·&nbsp; "
                f"chunk {chunk.chunk_index} &nbsp;·&nbsp; "
                f"{similarity_percent(chunk.distance):.0f}% match"
            )
            st.caption(chunk_preview(chunk.source_text))
            st.divider()

    st.caption(
        "Latency by stage, raw distances, and the prompt for this query are in "
        "the sidebar under **Last Query**."
    )

    render_diagnostics(chunks, prompt, timings, config)


def render_diagnostics(
    chunks: List[RetrievedChunk],
    prompt: str,
    timings: Dict[str, float],
    config: Config,
) -> None:
    """Show per-stage latency, raw distances, and the prompt in the sidebar.

    Latency is broken out by stage rather than reported as one number, because the
    split is the interesting part. Once connections are warm, the vector search is
    a small share of the total while answer generation dominates. That is the
    tradeoff S3 Vectors is built around, and it is easier to believe when measured
    than when described. Expect the first query after startup to be slower, since
    it pays client and TLS setup on every hop.

    Args:
        chunks: The retrieved chunks, closest match first.
        prompt: The grounded prompt sent to the generation model.
        timings: Milliseconds per stage, keyed ``embed``, ``search``,
            ``generate``, and ``total``.
        config: The resolved configuration, used to report which models ran.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Last Query")

    st.sidebar.metric("Vector search", f"{timings['search']:.0f} ms")
    st.sidebar.caption(
        f"Embed question: {timings['embed']:.0f} ms  \n"
        f"Generate answer: {timings['generate']:.0f} ms  \n"
        f"**Total: {timings['total']:.0f} ms**"
    )
    requested = config.top_k
    if len(chunks) < requested:
        st.sidebar.caption(
            f"Chunks retrieved: {len(chunks)} of {requested} requested. "
            "topK is an upper bound and approximate search can return fewer."
        )
    else:
        st.sidebar.caption(f"Chunks retrieved: {len(chunks)}")

    with st.sidebar.expander("Raw distances"):
        st.caption(
            f"`query_vectors` returns {INDEX_DISTANCE_METRIC} distance, where "
            "lower is closer. The match percentage shown with each document is "
            "`1 - distance`."
        )
        for chunk in chunks:
            st.text(f"{chunk.distance:.4f}  {chunk.key}")

    with st.sidebar.expander("Prompt sent to model"):
        st.caption(
            "Assembled by build_prompt() from the retrieved chunks. This is the "
            "whole of what the model sees."
        )
        st.code(prompt, language="text")

    st.sidebar.caption(
        f"Embedding: `{config.embedding_model_id}`  \n"
        f"Generation: `{config.generation_model_id}`"
    )


def run_query(
    question: str,
    config: Config,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> None:
    """Embed, retrieve, generate, and render results for a single question.

    Times the S3 Vectors query separately from embedding and generation so all
    three latencies can be displayed. AWS and setup failures are surfaced as
    friendly ``st.error`` messages rather than raw stack traces.

    Args:
        question: The natural-language question to answer.
        config: The resolved configuration (already reflecting the sidebar
            model and chunk-count selections).
        metadata_filter: An optional S3 Vectors filter expression restricting
            retrieval to a category. When ``None``, retrieval is unfiltered.
    """
    try:
        clients = get_clients(config.aws_region)
        brt = clients["bedrock-runtime"]
        s3v = clients["s3vectors"]

        with st.spinner("Searching documents..."):
            start = time.perf_counter()

            # Embed the question with the same model used at ingestion time.
            question_vector = embed_text(question, config, brt)
            embed_ms = (time.perf_counter() - start) * 1000.0

            # Time the S3 Vectors query on its own.
            search_start = time.perf_counter()
            chunks: List[RetrievedChunk] = retrieve(
                question_vector, config, s3v, metadata_filter
            )
            search_ms = (time.perf_counter() - search_start) * 1000.0

            if not chunks:
                st.warning(
                    "Nothing in the indexed documents matched that question. If "
                    "you have not ingested yet, run "
                    "`python ingest.py --source data/sample/`."
                )
                return

            # Rank closest-first for display (lower cosine distance == closer).
            chunks = sorted(chunks, key=lambda c: c.distance)

            contexts = [chunk.source_text for chunk in chunks]
            prompt = build_prompt(question, contexts)

            generate_start = time.perf_counter()
            answer = generate_answer(prompt, config, brt)
            generate_ms = (time.perf_counter() - generate_start) * 1000.0

            timings = {
                "embed": embed_ms,
                "search": search_ms,
                "generate": generate_ms,
                "total": (time.perf_counter() - start) * 1000.0,
            }

        render_results(chunks, answer, prompt, timings, config)

    except RagError as err:
        # Friendly, actionable messages (model access, missing bucket/index,
        # credentials) already carried by the custom exception.
        st.error(str(err))
    except NoCredentialsError:
        st.error(
            "AWS credentials could not be resolved. Configure them via the AWS "
            "CLI (`aws configure`), environment variables, or an IAM role, then "
            "retry."
        )
    except BotoCoreError as err:
        st.error(f"A network or AWS client error occurred: {err}")


def render_empty_state() -> None:
    """Show suggested questions before the first query is submitted."""
    st.caption("Try asking:")
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        st.button(
            suggestion,
            key=f"suggest_{index}",
            on_click=set_question,
            args=(suggestion,),
            use_container_width=True,
        )

    st.caption("Or ask something the documents do not cover:")
    st.button(
        OUT_OF_SCOPE_QUESTION,
        key="suggest_out_of_scope",
        on_click=set_question,
        args=(OUT_OF_SCOPE_QUESTION,),
        use_container_width=True,
        help="Nothing in the sample documents answers this. The model should say "
        "so instead of inventing an answer.",
    )


def main() -> None:
    """Render the page: header, sidebar, question input, and results."""
    # The center column is the end-user view, so it is titled as the assistant an
    # employee would use. The sample's AWS framing lives in the caption, the
    # sidebar, and the footer rather than in the heading.
    # The sidebar starts expanded because it holds the parts of this sample worth
    # studying. Streamlit collapses it by default on narrow windows, which hides
    # the diagnostics from someone opening the demo for the first time.
    st.set_page_config(
        page_title="Wally's Coffee Co. Knowledge Assistant",
        page_icon="☕",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("Wally's Coffee Co. Knowledge Assistant")
    st.caption(
        "Sample assistant built on Amazon S3 Vectors and Amazon Bedrock. "
        "Wally's Coffee Co. is a fictional company."
    )

    st.info(
        "Search across company handbooks, policies, operational guides, and "
        "business reports using natural language. Each document is split into "
        "chunks and embedded, and the vectors are stored in Amazon S3 Vectors. "
        "Queries are matched by meaning, not keywords."
    )

    # Point at the sidebar explicitly. Everything that shows how the pipeline
    # works lives there, and it is easy to miss otherwise.
    st.caption(
        "Looking at how this works? Open **Developer Controls** in the left "
        "sidebar to swap the generation model, change how many chunks are "
        "retrieved, and see per-query latency, raw vector distances, and the "
        "exact prompt sent to the model."
    )

    # Load configuration first so the sidebar controls can be seeded from it,
    # which is what makes .env overrides behave the same here as in query.py.
    base_config = load_config()
    generation_model_id, top_k = render_sidebar(base_config)

    # Reflect the sidebar selections in an immutable Config for this run.
    config = replace(
        base_config,
        generation_model_id=generation_model_id,
        top_k=top_k,
    )

    st.text_input(
        "Ask a question",
        key="question",
        placeholder="Ask about policies, procedures, sourcing, operations...",
    )

    # Optional metadata filter. S3 Vectors evaluates this alongside the
    # similarity search, so retrieval is restricted to the chosen category.
    category = st.selectbox(
        "Limit to a category (optional)",
        CATEGORY_CHOICES,
        index=0,
        help="Uses S3 Vectors metadata filtering to search only documents in "
        "this category. Leave on All documents to search everything.",
    )
    metadata_filter = (
        build_filter({"category": category}) if category != CATEGORY_CHOICES[0] else None
    )

    question = st.session_state.get("question", "")
    if question:
        run_query(question, config, metadata_filter)
    else:
        render_empty_state()

    st.markdown("---")
    st.caption("Powered by Amazon S3 Vectors + Amazon Bedrock")


if __name__ == "__main__":
    main()
