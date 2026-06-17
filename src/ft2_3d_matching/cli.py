from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

from .airtable import (
    AIRTABLE_ENV,
    airtable_configured,
    read_labels_from_airtable,
    setup_airtable_base,
    setup_instructions,
    sync_to_airtable,
)
from .data import (
    download_crunchbase_snapshot,
    load_prepared_companies,
    prepare_sample as prepare_sample_data,
    synthesize_founder_profiles,
    verify_data_files,
)
from .embeddings import cosine_similarity, get_embedder
from .evaluation import evaluate_with_pi, evidence_prompt
from .graph import run_pipeline
from .metrics import compute_precision_recall
from .models import Candidate, Label
from .review import render_review_workspace as render_workspace
from .scoring import rank_candidates as rank_candidate_list
from .scoring import score_founder_fit
from .settings import DEFAULT_THESIS, FASTEMBED_MODEL
from .textify import build_enriched_records, company_to_text
from .vector_store import index_records as qdrant_index_records
from .vector_store import local_client, query_records

load_dotenv(Path.cwd() / ".env", override=False)


def _print(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _prepared_records():
    companies = load_prepared_companies()
    founders = synthesize_founder_profiles(companies)
    return build_enriched_records(companies, founders)


def _ranked(
    thesis: str = DEFAULT_THESIS,
    top_k: int = 5,
    backend: str = "fastembed",
) -> list[Candidate]:
    records = _prepared_records()
    embedder = get_embedder(backend)
    vectors = embedder.embed([record.company_text for record in records])
    qdrant_index_records(records, vectors, local_client())
    thesis_vector = embedder.embed([thesis])[0]
    candidates = query_records(thesis_vector, limit=top_k, client=local_client())
    candidates = score_founder_fit(candidates, embedder)
    return rank_candidate_list(candidates)


def _pi_check() -> None:
    pi_config = Path(".pi/settings.json").exists()
    auth_present = bool(os.getenv("PI_AUTH_PRESENT") or os.getenv("PI_HOME") or Path.home().joinpath(".pi").exists())
    _print(
        {
            "pi_config_present": pi_config,
            "pi_auth_presence_hint": auth_present,
            "oauth_token_contents_read": False,
            "message": "Pi should be installed and logged in before the Pi/OpenAI OAuth chapter.",
        }
    )


def _download_data(dry_run: bool = typer.Option(False, help="Print target files without downloading.")) -> None:
    paths = download_crunchbase_snapshot(dry_run=dry_run)
    _print({"dry_run": dry_run, "targets": [str(path) for path in paths]})


def _verify_data() -> None:
    _print(verify_data_files())


def _prepare_sample(limit: int = 6) -> None:
    path = prepare_sample_data(limit=limit)
    _print({"sample_path": str(path), "rows": limit})


def _inspect_record(company_id: str = "c001") -> None:
    companies = load_prepared_companies()
    match = next((company for company in companies if company.company_id == company_id), companies[0])
    _print(match.model_dump())


def _textify_record(company_id: str = "c001") -> None:
    companies = load_prepared_companies()
    match = next((company for company in companies if company.company_id == company_id), companies[0])
    _print({"company_id": match.company_id, "text": company_to_text(match)})


def _vectorize_sample(backend: str = "fastembed") -> None:
    records = _prepared_records()
    embedder = get_embedder(backend)
    vectors = embedder.embed([records[0].company_text])
    _print(
        {
            "backend": backend,
            "model": FASTEMBED_MODEL if backend == "fastembed" else "deterministic-hash",
            "dimension": embedder.dimension,
            "first_values": [round(value, 6) for value in vectors[0][:8]],
        }
    )


def _start_vector_store() -> None:
    client = local_client()
    _print({"backend": "qdrant-local", "collections": [item.name for item in client.get_collections().collections]})


def _index_records(backend: str = "fastembed") -> None:
    records = _prepared_records()
    embedder = get_embedder(backend)
    vectors = embedder.embed([record.company_text for record in records])
    status = qdrant_index_records(records, vectors, local_client())
    status["backend"] = backend
    _print(status)


def _query_vector_store(
    query: str = DEFAULT_THESIS,
    top_k: int = 3,
    backend: str = "fastembed",
) -> None:
    records = _prepared_records()
    embedder = get_embedder(backend)
    vectors = embedder.embed([record.company_text for record in records])
    qdrant_index_records(records, vectors, local_client())
    results = query_records(embedder.embed([query])[0], limit=top_k, client=local_client())
    _print([candidate.model_dump() for candidate in results])


def _compare_vectors(backend: str = "fastembed") -> None:
    records = _prepared_records()
    embedder = get_embedder(backend)
    vectors = embedder.embed([DEFAULT_THESIS, records[0].company_text])
    manual = cosine_similarity(vectors[0], vectors[1])
    qdrant_index_records(records, embedder.embed([record.company_text for record in records]), local_client())
    nearest = query_records(vectors[0], limit=1, client=local_client())[0]
    _print(
        {
            "manual_cosine_for_first_record": round(manual, 6),
            "qdrant_top_company": nearest.company_name,
            "qdrant_score": round(nearest.company_thesis_similarity, 6),
        }
    )


def _score_market_stage(backend: str = "fastembed") -> None:
    candidates = _ranked(backend=backend)
    _print(
        [
            {
                "company_id": candidate.company_id,
                "company_name": candidate.company_name,
                "market_stage_score": candidate.market_stage_score,
            }
            for candidate in candidates
        ]
    )


def _rank_candidates(top_k: int = 5, backend: str = "fastembed") -> None:
    _print([candidate.model_dump() for candidate in _ranked(top_k=top_k, backend=backend)])


def _evaluate_leads(top_k: int = 5, backend: str = "fastembed") -> None:
    candidates = evaluate_with_pi(_ranked(top_k=top_k, backend=backend))
    _print(
        [
            {
                **candidate.model_dump(),
                "pi_evidence_prompt": evidence_prompt(candidate),
            }
            for candidate in candidates
        ]
    )


def _setup_airtable() -> None:
    result = setup_airtable_base()
    result["instructions"] = setup_instructions()
    _print(result)


def _verify_airtable() -> None:
    _print({"configured": airtable_configured(), "required_env": AIRTABLE_ENV})


def _sync_airtable(top_k: int = 5, backend: str = "fastembed") -> None:
    candidates = _ranked(top_k=top_k, backend=backend)
    _print(sync_to_airtable("local-cli-run", DEFAULT_THESIS, candidates, top_k, backend))


def _read_airtable_labels() -> None:
    _print(read_labels_from_airtable())


def _compute_metrics(top_k: int = 5, backend: str = "fastembed") -> None:
    candidates = _ranked(top_k=top_k, backend=backend)
    label_result = read_labels_from_airtable()
    labels = [Label(**item) for item in label_result.get("labels", [])]
    _print(compute_precision_recall(candidates, labels).model_dump())


def _render_review_workspace(top_k: int = 5, backend: str = "fastembed") -> None:
    final_state = run_pipeline(DEFAULT_THESIS, top_k, backend)
    _print(
        {
            "run_id": final_state["run_id"],
            "review_workspace_path": final_state["review_workspace_path"],
            "logs": final_state["logs"],
        }
    )


def pi_check() -> None:
    typer.run(_pi_check)


def download_data() -> None:
    typer.run(_download_data)


def verify_data() -> None:
    typer.run(_verify_data)


def prepare_sample() -> None:
    typer.run(_prepare_sample)


def inspect_record() -> None:
    typer.run(_inspect_record)


def textify_record() -> None:
    typer.run(_textify_record)


def vectorize_sample() -> None:
    typer.run(_vectorize_sample)


def start_vector_store() -> None:
    typer.run(_start_vector_store)


def index_records() -> None:
    typer.run(_index_records)


def query_vector_store() -> None:
    typer.run(_query_vector_store)


def compare_vectors() -> None:
    typer.run(_compare_vectors)


def score_market_stage() -> None:
    typer.run(_score_market_stage)


def rank_candidates() -> None:
    typer.run(_rank_candidates)


def evaluate_leads() -> None:
    typer.run(_evaluate_leads)


def setup_airtable() -> None:
    typer.run(_setup_airtable)


def verify_airtable() -> None:
    typer.run(_verify_airtable)


def sync_airtable() -> None:
    typer.run(_sync_airtable)


def read_airtable_labels() -> None:
    typer.run(_read_airtable_labels)


def compute_metrics() -> None:
    typer.run(_compute_metrics)


def render_review_workspace() -> None:
    typer.run(_render_review_workspace)


app = typer.Typer(help="3D deal matching workshop checkpoints.")

app.command("pi-check")(_pi_check)
app.command("download-data")(_download_data)
app.command("verify-data")(_verify_data)
app.command("prepare-sample")(_prepare_sample)
app.command("inspect-record")(_inspect_record)
app.command("textify-record")(_textify_record)
app.command("vectorize-sample")(_vectorize_sample)
app.command("start-vector-store")(_start_vector_store)
app.command("index-records")(_index_records)
app.command("query-vector-store")(_query_vector_store)
app.command("compare-vectors")(_compare_vectors)
app.command("score-market-stage")(_score_market_stage)
app.command("rank-candidates")(_rank_candidates)
app.command("evaluate-leads")(_evaluate_leads)
app.command("setup-airtable")(_setup_airtable)
app.command("verify-airtable")(_verify_airtable)
app.command("sync-airtable")(_sync_airtable)
app.command("read-airtable-labels")(_read_airtable_labels)
app.command("compute-metrics")(_compute_metrics)
app.command("render-review-workspace")(_render_review_workspace)


if __name__ == "__main__":
    app()
