# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the configuration loader.

These tests exercise the pure environment-variable-override logic in
``config.load_config`` across many generated inputs using Hypothesis.
"""

from __future__ import annotations

import os
from typing import Dict

from hypothesis import given, settings
from hypothesis import strategies as st

from config import DEFAULTS, load_config

# Maps each environment-variable key to the corresponding Config attribute.
KEY_TO_ATTR: Dict[str, str] = {
    "VECTOR_BUCKET_NAME": "vector_bucket_name",
    "INDEX_NAME": "index_name",
    "EMBEDDING_MODEL_ID": "embedding_model_id",
    "GENERATION_MODEL_ID": "generation_model_id",
    "AWS_REGION": "aws_region",
    "TOP_K": "top_k",
}

# String config keys receive arbitrary text; TOP_K receives an integer because
# load_config casts it to int.
STRING_KEYS = [k for k in KEY_TO_ATTR if k != "TOP_K"]

# Text that is safe to place in an environment variable: no null bytes.
_env_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=40,
)


@st.composite
def config_overrides(draw: st.DrawFn) -> Dict[str, object]:
    """Generate an arbitrary subset of config keys mapped to arbitrary values.

    String keys map to arbitrary text; ``TOP_K`` maps to an arbitrary integer.
    """
    chosen = draw(st.lists(st.sampled_from(list(KEY_TO_ATTR)), unique=True))
    overrides: Dict[str, object] = {}
    for key in chosen:
        if key == "TOP_K":
            overrides[key] = draw(st.integers(min_value=0, max_value=1000))
        else:
            overrides[key] = draw(_env_text)
    return overrides


# Feature: sample-s3-vectors-rag-pipeline, Property 1: Environment variables override configuration defaults
@settings(max_examples=100)
@given(overrides=config_overrides())
def test_env_vars_override_defaults(overrides: Dict[str, object]) -> None:
    """Env vars override defaults; unset keys fall back to documented defaults."""
    # Save and clear all managed config env vars so each example starts clean.
    saved = {key: os.environ.get(key) for key in KEY_TO_ATTR}
    try:
        for key in KEY_TO_ATTR:
            os.environ.pop(key, None)
        # Inject the generated subset as environment variables.
        for key, value in overrides.items():
            os.environ[key] = str(value)

        config = load_config()

        for key, attr in KEY_TO_ATTR.items():
            actual = getattr(config, attr)
            if key in overrides:
                if key == "TOP_K":
                    # TOP_K is parsed as an integer.
                    assert actual == int(overrides[key])
                else:
                    assert actual == overrides[key]
            else:
                # Unset keys fall back to their documented defaults.
                expected_default = DEFAULTS[key]
                if key == "TOP_K":
                    assert actual == int(expected_default)
                else:
                    assert actual == str(expected_default)
    finally:
        # Restore the original environment.
        for key, original in saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
