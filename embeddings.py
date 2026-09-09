# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Embedding wrapper around Amazon Titan Text Embeddings v2.

This module is the single point through which both ingestion and querying turn
text into vectors. Routing every embedding call through one function guarantees
that ingestion and query use an identical model id and request shape
(Requirement 3.3), which is what makes retrieval meaningful.

The wrapper also translates a Bedrock ``AccessDeniedException`` into a
:class:`~errors.ModelAccessError` that names the affected model, so callers
get a clear, actionable message instead of a raw botocore stack trace
(Requirements 3.4, 8.1).
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import boto3
from botocore.exceptions import ClientError

from config import Config
from errors import ModelAccessError


def embed_text(
    text: str,
    config: Config,
    client: Optional[Any] = None,
) -> List[float]:
    """Return a 1024-element float32 embedding for ``text``.

    Builds the Titan request body ``{"inputText": text}``, invokes the embedding
    model configured on ``config`` through Bedrock Runtime, and parses the
    ``embedding`` array from the response. Because both the ingestion and query
    paths call this one function, they always embed with an identical model id
    and request shape (Requirements 3.1, 3.2, 3.3).

    Args:
        text: The text to embed.
        config: Resolved configuration supplying ``embedding_model_id`` and
            ``aws_region``.
        client: Optional pre-constructed ``bedrock-runtime`` client. When
            omitted, a client is created for ``config.aws_region``. Injecting a
            client keeps the function testable without live AWS calls.

    Returns:
        The embedding as a ``list`` of 1024 ``float`` values.

    Raises:
        ModelAccessError: If Bedrock denies access to the embedding model
            (``AccessDeniedException``), naming the affected model
            (Requirements 3.4, 8.1).
    """
    if client is None:
        client = boto3.client("bedrock-runtime", region_name=config.aws_region)

    body = json.dumps({"inputText": text})

    try:
        response = client.invoke_model(
            modelId=config.embedding_model_id,
            body=body,
        )
    except ClientError as err:
        error_code = err.response.get("Error", {}).get("Code")
        if error_code == "AccessDeniedException":
            raise ModelAccessError(config.embedding_model_id) from err
        raise

    payload = json.loads(response["body"].read())
    return payload["embedding"]
