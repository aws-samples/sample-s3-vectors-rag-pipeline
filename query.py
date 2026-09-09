# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Query entry point for the Amazon S3 Vectors RAG pipeline sample.

This module owns the query flow: it resolves a natural-language question (from a
command-line argument or an interactive prompt), embeds it, retrieves the most
similar chunks from S3 Vectors, builds a grounded prompt, calls the configured
generation model through the Bedrock Converse API, and prints the answer with
its source attributions.

The generation model is read exclusively from :class:`~config.Config`, so a
model can be swapped without changing any query logic (Requirement 7).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import Config, load_config
from errors import (
    BucketNotFoundError,
    CredentialsError,
    ModelAccessError,
    RagError,
)
from embeddings import embed_text

# Botocore error codes that indicate AWS credentials are missing, invalid, or
# expired (Requirement 8.3). Distinct from a Bedrock model-access denial, which
# is handled separately as ModelAccessError.
CREDENTIAL_ERROR_CODES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "AuthFailure",
        "InvalidClientTokenId",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "ExpiredTokenException",
    }
)


@dataclass
class RetrievedChunk:
    """A single chunk returned by an S3 Vectors ``query_vectors`` call.

    Each instance is parsed from one entry of the ``query_vectors`` response
    ``vectors[]`` list and carries the retrieved chunk's identity, similarity
    distance, and stored metadata.

    Attributes:
        key: The vector key (for example ``"<source_file>::<chunk_index>"``).
        distance: The similarity distance returned for the chunk (lower is
            closer under the cosine metric used by the index).
        source_file: Filename the chunk was ingested from.
        chunk_index: Zero-based position of the chunk within its source document.
        category: The document's category (filterable metadata).
        source_text: The chunk text stored as non-filterable metadata.
    """

    key: str
    distance: float
    source_file: str
    chunk_index: int
    category: str
    source_text: str


def get_question(argv: List[str]) -> str:
    """Resolve the question to answer from CLI arguments or interactive input.

    Treats ``argv`` as ``sys.argv``-style, where index 0 is the program name. If
    a question argument is present (``len(argv) > 1``), the first such argument
    (``argv[1]``) is returned verbatim, with no modification (Requirement 6.1).
    Otherwise the user is prompted for the question through interactive input
    (Requirement 6.2).

    Args:
        argv: The argument vector, typically ``sys.argv``. Index 0 is the
            program name; index 1, when present, is the question.

    Returns:
        The question string: ``argv[1]`` verbatim when provided, otherwise the
        line entered at the interactive prompt.
    """
    if len(argv) > 1:
        return argv[1]
    return input("Enter your question: ")


def parse_filter_args(raw: Optional[List[str]]) -> Dict[str, str]:
    """Parse ``--filter KEY=VALUE`` arguments into a mapping.

    Accepts the repeated ``--filter`` values collected by argparse. Each must be
    ``key=value``; the value may itself contain ``=`` (only the first ``=`` is
    the separator). Returns an empty mapping when nothing was passed.

    Args:
        raw: The list of raw ``KEY=VALUE`` strings, or ``None``.

    Returns:
        A mapping of metadata key to value.

    Raises:
        ValueError: If any argument is not in ``key=value`` form.
    """
    pairs: Dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(
                f"invalid --filter '{item}', expected KEY=VALUE "
                "(for example category=hr)"
            )
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(
                f"invalid --filter '{item}', both key and value are required"
            )
        pairs[key] = value
    return pairs


def build_filter(pairs: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Build an S3 Vectors metadata filter expression from key=value pairs.

    S3 Vectors filter expressions use MongoDB-style operators. A single pair
    becomes ``{"key": {"$eq": "value"}}``; multiple pairs are combined with
    ``$and``. Passing an empty or ``None`` mapping returns ``None``, which the
    caller treats as an unfiltered query. The ``$eq`` operator is used explicitly
    rather than the bare-value shorthand so the intent is clear in the request.

    Args:
        pairs: A mapping of filterable metadata key to the value to match on, or
            ``None``.

    Returns:
        A filter expression suitable for the ``filter`` argument of
        ``query_vectors``, or ``None`` when no pairs are given.
    """
    if not pairs:
        return None
    conditions = [{key: {"$eq": value}} for key, value in pairs.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    question_vector: List[float],
    config: Config,
    s3v_client,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[RetrievedChunk]:
    """Retrieve the most similar chunks for a question from S3 Vectors.

    Performs an approximate nearest-neighbor search against the configured vector
    index, requesting ``config.top_k`` results with metadata and distance
    enabled (Requirement 6.4). Each returned vector is parsed into a
    :class:`RetrievedChunk`, collecting the ``source_text`` stored as metadata
    (Requirement 6.5). If the query returns no vectors, a message reporting that
    no relevant context was found is printed and an empty list is returned
    (Requirement 6.9).

    When ``metadata_filter`` is supplied, it is passed to ``query_vectors`` as
    the ``filter`` argument, so S3 Vectors evaluates similarity and the filter
    together and returns only matching vectors. When omitted, the query is
    unfiltered exactly as before. Note that a filtered query can return fewer
    than ``top_k`` results when few vectors match.

    Args:
        question_vector: The 1024-element embedding of the question to search
            for, as produced by the embedder.
        config: The resolved configuration supplying the vector bucket name,
            index name, and ``top_k`` value.
        s3v_client: An S3 Vectors (``s3vectors``) client used to issue the
            ``query_vectors`` call.
        metadata_filter: An optional S3 Vectors filter expression (see
            :func:`build_filter`). When ``None``, the query is unfiltered.

    Returns:
        A list of :class:`RetrievedChunk` objects, one per returned vector, in
        the order provided by S3 Vectors. Returns an empty list when the query
        matches no vectors.
    """
    params: Dict[str, Any] = {
        "vectorBucketName": config.vector_bucket_name,
        "indexName": config.index_name,
        "topK": config.top_k,
        "queryVector": {"float32": question_vector},
        "returnMetadata": True,
        "returnDistance": True,
    }
    # Only include filter when set, so an unfiltered call is byte-for-byte the
    # original request shape. Filtering also requires s3vectors:GetVectors.
    if metadata_filter:
        params["filter"] = metadata_filter

    response = s3v_client.query_vectors(**params)

    vectors = response.get("vectors", [])
    if not vectors:
        print("No relevant context was found for your question.")
        return []

    chunks: List[RetrievedChunk] = []
    for vector in vectors:
        metadata = vector.get("metadata", {})
        chunks.append(
            RetrievedChunk(
                key=vector.get("key", ""),
                distance=vector.get("distance", 0.0),
                source_file=metadata.get("source_file", ""),
                chunk_index=metadata.get("chunk_index", 0),
                category=metadata.get("category", ""),
                source_text=metadata.get("source_text", ""),
            )
        )
    return chunks


def build_prompt(question: str, contexts: List[str]) -> str:
    """Build a grounded prompt instructing the model to answer only from context.

    Assembles a single prompt string that carries the retrieved context and the
    user's question, together with an explicit instruction that the model must
    answer using *only* the provided context and must state clearly when the
    context does not contain the answer (Requirement 6.6). Grounding the model
    this way keeps answers attributable to the ingested documents and avoids
    unsupported claims.

    The returned prompt always contains the ``question`` text and every string in
    ``contexts``. Each context is numbered for readability; an empty
    ``contexts`` list still produces a well-formed prompt (with an empty context
    section), leaving it to the instruction to make the model state that no
    answer is available.

    Args:
        question: The natural-language question to answer.
        contexts: The ``source_text`` values of the retrieved chunks, in
            retrieval order. May be empty.

    Returns:
        A prompt string suitable for use as the user message content sent to the
        generation model, containing the instruction, every context passage, and
        the question.
    """
    instruction = (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "provided below. Do not use any prior knowledge or make assumptions "
        "beyond the context. If the context does not contain the information "
        "needed to answer the question, explicitly state that the answer is not "
        "present in the provided context."
    )

    if contexts:
        context_block = "\n\n".join(
            f"[Context {index}]\n{context}"
            for index, context in enumerate(contexts, start=1)
        )
    else:
        context_block = "(no context available)"

    return (
        f"{instruction}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(
    prompt: str, config: Config, brt_client: Optional[Any] = None
) -> str:
    """Generate an answer for a grounded prompt via the Bedrock Converse API.

    Invokes ``config.generation_model_id`` through Bedrock Runtime's ``converse``
    operation and returns the generated answer text (Requirements 6.7, 7.1, 7.2).
    The Converse API accepts one unified request/response shape across every
    Bedrock text model (Claude, Nova, Grok, DeepSeek, Llama, and others), so the
    configured model can be swapped without changing any query logic and without
    branching on the model id (Requirement 7, model swappability).

    If ``brt_client`` is not supplied, a ``bedrock-runtime`` client is created for
    ``config.aws_region``. A Bedrock ``AccessDeniedException`` is translated into a
    :class:`~errors.ModelAccessError` that names the affected model, so callers
    receive a clear, actionable message instead of a raw botocore stack trace
    (Requirements 8.1).

    Args:
        prompt: The grounded prompt (typically from :func:`build_prompt`) to send
            as the user message content.
        config: The resolved configuration supplying ``generation_model_id`` and
            ``aws_region``.
        brt_client: An optional Bedrock Runtime (``bedrock-runtime``) client. When
            ``None``, a client is created for ``config.aws_region``.

    Returns:
        The generated answer text parsed from the model response.

    Raises:
        ModelAccessError: If Bedrock denies access to the generation model
            (``AccessDeniedException``), naming the affected model
            (Requirements 8.1).
    """
    if brt_client is None:
        brt_client = boto3.client("bedrock-runtime", region_name=config.aws_region)

    try:
        response = brt_client.converse(
            modelId=config.generation_model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024},
        )
    except ClientError as err:
        error_code = err.response.get("Error", {}).get("Code")
        if error_code == "AccessDeniedException":
            raise ModelAccessError(config.generation_model_id) from err
        raise

    # The Converse response content is a list of blocks. Reasoning models (e.g.
    # Grok, DeepSeek) may emit a reasoningContent block before the text block, so
    # collect every text block rather than assuming it is first.
    content_blocks = response["output"]["message"]["content"]
    texts = [block["text"] for block in content_blocks if "text" in block]
    return "\n".join(texts).strip()


def _is_credentials_error(err: ClientError) -> bool:
    """Return whether a botocore ``ClientError`` indicates a credentials problem.

    Checks the error ``Code`` against :data:`CREDENTIAL_ERROR_CODES`, the set of
    codes AWS uses when credentials are missing, invalid, or expired
    (Requirement 8.3).

    Args:
        err: The botocore ``ClientError`` to classify.

    Returns:
        ``True`` if the error looks like a credential/auth failure.
    """
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    return str(error.get("Code", "")) in CREDENTIAL_ERROR_CODES


def _is_bucket_not_found(err: ClientError) -> bool:
    """Return whether a botocore ``ClientError`` indicates a missing vector bucket.

    S3 Vectors surfaces a missing bucket as a not-found code whose message
    references the bucket, so this checks both the ``Code`` and that the
    human-readable message mentions the bucket (Requirement 8.2).

    Args:
        err: The botocore ``ClientError`` to classify.

    Returns:
        ``True`` if the error looks like a missing-bucket condition.
    """
    error = err.response.get("Error", {}) if hasattr(err, "response") else {}
    code = str(error.get("Code", ""))
    message = str(error.get("Message", "")).lower()
    not_found_codes = {"NotFoundException", "ResourceNotFoundException"}
    is_not_found = code in not_found_codes or "not found" in message or (
        "does not exist" in message
    )
    return is_not_found and "bucket" in message


def answer_question(
    question: str,
    config: Config,
    brt_client,
    s3v_client,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> None:
    """Answer one question from the indexed documents and print the result.

    Embeds the question with the same embedder used at ingestion time so the
    query vector lives in the same space as the stored chunks (Requirement 6.3),
    retrieves the closest chunks (Requirement 6.4), builds a grounded prompt
    (Requirement 6.6), generates the answer (Requirement 6.7), and prints it with
    the deduplicated set of source files (Requirement 6.8). If retrieval returns
    nothing, :func:`retrieve` has already reported that and this returns early
    (Requirement 6.9).

    Args:
        question: The natural-language question to answer.
        config: Resolved configuration.
        brt_client: A ``bedrock-runtime`` client used for embedding and generation.
        s3v_client: An ``s3vectors`` client used for retrieval.
        metadata_filter: An optional S3 Vectors filter expression restricting
            retrieval to matching documents. When ``None``, retrieval is
            unfiltered.
    """
    question_vector = embed_text(question, config, brt_client)

    chunks = retrieve(question_vector, config, s3v_client, metadata_filter)
    if not chunks:
        return

    contexts = [chunk.source_text for chunk in chunks]
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt, config, brt_client)

    print(answer)

    sources = sorted({chunk.source_file for chunk in chunks})
    print("\nSources:")
    for source in sources:
        print(f"  - {source}")


def main(argv: Optional[List[str]] = None) -> None:
    """Run the end-to-end query flow and print the grounded answer with sources.

    Orchestrates the full query pipeline: resolve the question (Requirement 6.1,
    6.2), embed it (Requirement 6.3), retrieve the most similar chunks
    (Requirement 6.4), build a grounded prompt (Requirement 6.6), generate an
    answer through the configured model (Requirement 6.7), and print the answer
    together with the deduplicated set of ``source_file`` values from the
    retrieved chunks (Requirement 6.8).

    Two AWS clients are created up front from ``config.aws_region``: a
    ``bedrock-runtime`` client shared by the embedding and generation calls, and
    an ``s3vectors`` client used for retrieval. If retrieval returns no chunks,
    the function returns early because :func:`retrieve` has already reported that
    no relevant context was found (Requirement 6.9).

    Supports a positional question, ``--question``, and ``--interactive`` for a
    multi-question session.

    Args:
        argv: An optional ``sys.argv``-style argument vector. When ``None``,
            ``sys.argv`` is used. Index 0 is the program name.

    Returns:
        None. Output (the answer and its sources) is written to stdout.
    """
    if argv is None:
        argv = sys.argv

    # Model output can contain characters a legacy Windows console code page
    # (cp1252) cannot encode, which would raise UnicodeEncodeError on print.
    # Degrade those characters instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Ask a question answered from documents in Amazon S3 Vectors."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help=(
            "The question to answer. If your question begins with a hyphen, "
            'separate it with -- first, for example: query.py -- "-40C means what?"'
        ),
    )
    parser.add_argument(
        "--question",
        dest="question_flag",
        default=None,
        help="The question to answer (alternative to the positional form).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask questions repeatedly until you submit an empty line.",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Restrict retrieval to matching documents, for example "
            "--filter category=hr or --filter source_file=employee-handbook.md. "
            "Repeat to require several conditions."
        ),
    )
    args = parser.parse_args(argv[1:])

    metadata_filter = build_filter(parse_filter_args(args.filters))

    config = load_config()

    try:
        brt = boto3.client("bedrock-runtime", region_name=config.aws_region)
        s3v = boto3.client("s3vectors", region_name=config.aws_region)

        if args.interactive:
            print("Interactive mode. Submit an empty line to exit.")
            while True:
                question = input("\nQuestion: ").strip()
                if not question:
                    print("Exiting.")
                    break
                answer_question(question, config, brt, s3v, metadata_filter)
            return

        question = args.question_flag or args.question
        if question is None:
            question = input("Enter your question: ")

        answer_question(question, config, brt, s3v, metadata_filter)
    except NoCredentialsError as err:
        error = CredentialsError()
        print(error)
        raise error from err
    except ClientError as err:
        if _is_credentials_error(err):
            error = CredentialsError()
            print(error)
            raise error from err
        if _is_bucket_not_found(err):
            error = BucketNotFoundError(config.vector_bucket_name)
            print(error)
            raise error from err
        raise
    except RagError as err:
        # Includes ModelAccessError/BucketNotFoundError raised deeper in the
        # flow: report the concise message and let the guard exit non-zero.
        print(err)
        raise


if __name__ == "__main__":
    try:
        main()
    except RagError:
        sys.exit(1)
