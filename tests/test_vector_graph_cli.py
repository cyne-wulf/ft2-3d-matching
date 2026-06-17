from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ft2_3d_matching.data import load_prepared_companies, prepare_sample, synthesize_founder_profiles
from ft2_3d_matching.embeddings import DeterministicEmbedder
from ft2_3d_matching.graph import run_pipeline
from ft2_3d_matching.textify import build_enriched_records
from ft2_3d_matching.vector_store import index_records, local_client, query_records


def test_qdrant_local_index_and_query(tmp_path: Path):
    prepare_sample(limit=3)
    companies = load_prepared_companies()
    records = build_enriched_records(companies, synthesize_founder_profiles(companies))
    embedder = DeterministicEmbedder()
    vectors = embedder.embed([record.company_text for record in records])
    client = local_client(tmp_path / "qdrant")

    status = index_records(records, vectors, client=client, collection="test_records")
    results = query_records(
        embedder.embed(["enterprise workflow automation"])[0],
        client=client,
        collection="test_records",
    )

    assert status["indexed_count"] == 3
    assert results


def test_langgraph_pipeline_renders_workspace():
    final_state = run_pipeline(top_k=3, embedding_backend="deterministic")

    assert final_state["scored_leads"]
    assert Path(final_state["review_workspace_path"]).exists()
    assert "render_review_workspace" in final_state["logs"][-1]


def test_cli_smoke_rank_candidates(tmp_path: Path):
    env = {**os.environ, "FT2_VECTOR_STORE_PATH": str(tmp_path / "cli-qdrant")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ft2_3d_matching.cli",
            "rank-candidates",
            "--backend",
            "deterministic",
            "--top-k",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 2
    assert {"Review", "Watch", "Pass"} >= {item["recommendation"] for item in payload}


def test_cli_loads_airtable_credentials_from_env_file(tmp_path: Path):
    tmp_path.joinpath(".env").write_text(
        "AIRTABLE_PERSONAL_ACCESS_TOKEN=pat_from_env_file\n"
        "AIRTABLE_BASE_ID=app_from_env_file\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"AIRTABLE_PERSONAL_ACCESS_TOKEN", "AIRTABLE_" + "API_KEY", "AIRTABLE_BASE_ID"}
    }

    result = subprocess.run(
        [sys.executable, "-m", "ft2_3d_matching.cli", "verify-airtable"],
        check=True,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )

    assert json.loads(result.stdout) == {
        "configured": True,
        "required_env": ["AIRTABLE_PERSONAL_ACCESS_TOKEN", "AIRTABLE_BASE_ID"],
    }


def test_cli_shell_airtable_credentials_override_env_file(tmp_path: Path):
    tmp_path.joinpath(".env").write_text(
        "AIRTABLE_PERSONAL_ACCESS_TOKEN=pat_from_env_file\n"
        "AIRTABLE_BASE_ID=app_from_env_file\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "AIRTABLE_PERSONAL_ACCESS_TOKEN": "pat_from_shell",
        "AIRTABLE_BASE_ID": "app_from_shell",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; "
                "from ft2_3d_matching.cli import _verify_airtable; "
                "_verify_airtable(); "
                "assert os.environ['AIRTABLE_PERSONAL_ACCESS_TOKEN'] == 'pat_from_shell'; "
                "assert os.environ['AIRTABLE_BASE_ID'] == 'app_from_shell'"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )

    assert json.loads(result.stdout)["configured"] is True
