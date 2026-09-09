# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Smoke / convention tests for the sample-s3-vectors-rag-pipeline project.

These tests assert the project's structural and documentation conventions:
required files exist, dependencies are declared, the README carries the required
title and sections, the ``.env.example`` lists all configuration keys, the config
module documents alternative generation models, and the aws-samples governance
files are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repository root, located relative to this test file (tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Required files exist (flat layout + aws-samples governance files)
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "LICENSE",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "provision.py",
    "ingest.py",
    "query.py",
    "app.py",
    "config.py",
    "chunker.py",
    "embeddings.py",
    "cleanup.py",
    "tests/test_pipeline.py",
]


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_exists(relative_path: str) -> None:
    """Every file mandated by Requirement 9.1 must be present in the repo."""
    target = REPO_ROOT / relative_path
    assert target.is_file(), f"Required file missing: {relative_path}"


# ---------------------------------------------------------------------------
# Requirement 9.2 — requirements.txt declares boto3 and python-dotenv
# ---------------------------------------------------------------------------


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("dependency", ["boto3", "python-dotenv"])
def test_requirements_declares_dependency(dependency: str) -> None:
    """requirements.txt must declare boto3 and python-dotenv (Req 9.2)."""
    contents = _read("requirements.txt")
    declared = {
        line.strip().split("==")[0].split(">=")[0].split("<")[0].strip().lower()
        for line in contents.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert dependency.lower() in declared, (
        f"requirements.txt must declare '{dependency}'. Declared: {sorted(declared)}"
    )


# ---------------------------------------------------------------------------
# Requirements 9.3 / 9.4 — README title and required sections
# ---------------------------------------------------------------------------

README_TITLE = "Amazon S3 Vectors RAG Pipeline"
README_SECTIONS = [
    "Why S3 Vectors",
    "Architecture",
    "Project Structure",
    "Prerequisites",
    "Getting Started",
    "Configuration Options",
    "Cleanup",
    "Security",
    "License",
]


def test_readme_has_exact_title() -> None:
    """README.md must use the exact required title (Req 9.3)."""
    readme = _read("README.md")
    assert README_TITLE in readme, f"README.md must contain the title '{README_TITLE}'"


@pytest.mark.parametrize("section", README_SECTIONS)
def test_readme_contains_section(section: str) -> None:
    """README.md must contain each required section heading (Req 9.4)."""
    readme = _read("README.md").lower()
    assert section.lower() in readme, f"README.md is missing a '{section}' section"


# ---------------------------------------------------------------------------
# Requirement 9.5 — .env.example lists all six configuration keys
# ---------------------------------------------------------------------------

ENV_KEYS = [
    "VECTOR_BUCKET_NAME",
    "INDEX_NAME",
    "EMBEDDING_MODEL_ID",
    "GENERATION_MODEL_ID",
    "AWS_REGION",
    "TOP_K",
]


@pytest.mark.parametrize("key", ENV_KEYS)
def test_env_example_lists_key(key: str) -> None:
    """.env.example must provide a template entry for each config key (Req 9.5)."""
    env_example = _read(".env.example")
    assert key in env_example, f".env.example must list the '{key}' key"


# ---------------------------------------------------------------------------
# Requirement 1.8 — config.py documents commented alternative model ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["Grok", "DeepSeek", "Nova"])
def test_config_documents_alternative_models(model_name: str) -> None:
    """config.py must include commented alternative model ids (Req 1.8)."""
    config_src = _read("config.py")
    # Only consider commented lines so we validate documented alternatives.
    commented = "\n".join(
        line for line in config_src.splitlines() if line.lstrip().startswith("#")
    )
    assert model_name.lower() in commented.lower(), (
        f"config.py must mention '{model_name}' among commented alternative models"
    )


# ---------------------------------------------------------------------------
# aws-samples publishing conventions
# ---------------------------------------------------------------------------


def test_license_is_mit_zero() -> None:
    """LICENSE must be MIT-0 with the Amazon copyright line."""
    license_text = _read("LICENSE")
    assert "Copyright Amazon.com, Inc. or its affiliates" in license_text
    # MIT-0 drops the attribution clause that standard MIT requires.
    assert "The above copyright notice" not in license_text


@pytest.mark.parametrize(
    "relative_path",
    ["provision.py", "ingest.py", "query.py", "app.py", "config.py", "chunker.py",
     "embeddings.py", "cleanup.py", "errors.py"],
)
def test_python_file_has_copyright_header(relative_path: str) -> None:
    """Every Python module must carry the aws-samples copyright header."""
    lines = _read(relative_path).splitlines()[:2]
    assert lines[0] == (
        "# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."
    )
    assert lines[1] == "# SPDX-License-Identifier: MIT-0"


def test_no_series_or_personal_branding() -> None:
    """The published sample must not carry series branding or attribution."""
    readme = _read("README.md")
    for banned in ("S3 + AI", "Part 1", "Built by", "blog post"):
        assert banned not in readme, f"README should not mention '{banned}'"
