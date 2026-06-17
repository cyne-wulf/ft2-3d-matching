from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from .metrics import compute_precision_recall
from .models import Candidate, Label, Metrics

AIRTABLE_TOKEN_ENV = "AIRTABLE_PERSONAL_ACCESS_TOKEN"
AIRTABLE_BASE_ENV = "AIRTABLE_BASE_ID"
AIRTABLE_ENV = [AIRTABLE_TOKEN_ENV, AIRTABLE_BASE_ENV]
TABLES = {"runs": "Runs", "leads": "Leads", "labels": "Labels"}
AIRTABLE_TABLE_SCHEMAS = {
    TABLES["runs"]: [
        {"name": "Run ID", "type": "singleLineText"},
        {"name": "Timestamp", "type": "singleLineText"},
        {"name": "Thesis", "type": "multilineText"},
        {"name": "Top K", "type": "number", "options": {"precision": 0}},
        {"name": "Embedding Backend", "type": "singleLineText"},
        {"name": "Vector Store Backend", "type": "singleLineText"},
        {"name": "Company Count", "type": "number", "options": {"precision": 0}},
        {"name": "Precision", "type": "number", "options": {"precision": 6}},
        {"name": "Recall", "type": "number", "options": {"precision": 6}},
    ],
    TABLES["leads"]: [
        {"name": "Company ID", "type": "singleLineText"},
        {"name": "Run ID", "type": "singleLineText"},
        {"name": "Company Name", "type": "singleLineText"},
        {"name": "Category", "type": "singleLineText"},
        {"name": "Status", "type": "singleLineText"},
        {"name": "Country", "type": "singleLineText"},
        {"name": "Founded Year", "type": "number", "options": {"precision": 0}},
        {"name": "Funding Total USD", "type": "number", "options": {"precision": 2}},
        {"name": "Funding Rounds", "type": "number", "options": {"precision": 0}},
        {"name": "Company Thesis Similarity", "type": "number", "options": {"precision": 6}},
        {"name": "Market Stage Score", "type": "number", "options": {"precision": 6}},
        {"name": "Founder Team Similarity", "type": "number", "options": {"precision": 6}},
        {"name": "Overall Score", "type": "number", "options": {"precision": 6}},
        {"name": "Recommendation", "type": "singleLineText"},
        {
            "name": "Is Relevant",
            "type": "checkbox",
            "options": {"icon": "check", "color": "greenBright"},
        },
        {"name": "Label Reason", "type": "multilineText"},
        {"name": "Evaluation Outcome", "type": "singleLineText"},
        {"name": "Run Precision", "type": "number", "options": {"precision": 6}},
        {"name": "Run Recall", "type": "number", "options": {"precision": 6}},
        {"name": "Rationale", "type": "multilineText"},
        {"name": "Evidence", "type": "multilineText"},
        {"name": "Risks", "type": "multilineText"},
        {"name": "Missing Information", "type": "multilineText"},
    ],
    TABLES["labels"]: [
        {"name": "Company ID", "type": "singleLineText"},
        {"name": "Company Name", "type": "singleLineText"},
        {
            "name": "Is Relevant",
            "type": "checkbox",
            "options": {"icon": "check", "color": "greenBright"},
        },
        {"name": "Label Reason", "type": "multilineText"},
        {"name": "Reviewer", "type": "singleLineText"},
        {"name": "Reviewed At", "type": "singleLineText"},
    ],
}
DEMO_REVIEWED_AT = "2026-06-17T00:00:00Z"
DEMO_LABELS = [
    Label(
        company_id="c001",
        company_name="OpsPilot",
        is_relevant=True,
        label_reason="Strong workflow automation fit.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
    Label(
        company_id="c002",
        company_name="ComplyGrid",
        is_relevant=True,
        label_reason="Compliance workflow automation match.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
    Label(
        company_id="c003",
        company_name="DataForge",
        is_relevant=False,
        label_reason="Marketing analytics is outside the target thesis.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
    Label(
        company_id="c004",
        company_name="RoboFoundry",
        is_relevant=False,
        label_reason="Hardware robotics is outside the software workflow thesis.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
    Label(
        company_id="c005",
        company_name="InfraWeave",
        is_relevant=True,
        label_reason="AI infrastructure orchestration match.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
    Label(
        company_id="c006",
        company_name="Wellnest",
        is_relevant=False,
        label_reason="Consumer wellness is outside the target market.",
        reviewer="Demo seed",
        reviewed_at=DEMO_REVIEWED_AT,
    ),
]


def airtable_configured() -> bool:
    return all(os.getenv(name) for name in AIRTABLE_ENV)


def setup_instructions() -> str:
    return (
        "Create a default Airtable base. Then create a personal access token with "
        "data.records:read, data.records:write, schema.bases:read, and "
        "schema.bases:write scopes for that base, and set "
        "AIRTABLE_PERSONAL_ACCESS_TOKEN and AIRTABLE_BASE_ID in your shell. "
        "setup-airtable creates the Runs, Leads, and Labels tables. The workshop "
        "does not read or print token contents."
    )


def ensure_airtable_schema() -> dict[str, Any]:
    if not airtable_configured():
        return {"configured": False, "message": setup_instructions()}

    base = _airtable_base()
    existing_tables = {table.name: table for table in base.tables(force=True)}
    created_tables: list[str] = []
    created_fields: list[str] = []

    for table_name, fields in AIRTABLE_TABLE_SCHEMAS.items():
        table = existing_tables.get(table_name)
        if table is None:
            base.create_table(table_name, fields)
            created_tables.append(table_name)
            continue

        existing_fields = {field.name for field in table.schema(force=True).fields}
        for field in fields:
            field_name = field["name"]
            if field_name in existing_fields:
                continue
            if _create_field_if_missing(table, field):
                created_fields.append(f"{table_name}.{field_name}")

    return {
        "configured": True,
        "created_tables": created_tables,
        "created_fields": created_fields,
        "tables": list(AIRTABLE_TABLE_SCHEMAS),
    }


def setup_airtable_base() -> dict[str, Any]:
    schema_result = ensure_airtable_schema()
    if not schema_result.get("configured"):
        return schema_result

    seed_result = seed_demo_labels()
    return {
        **schema_result,
        "seeded_labels": seed_result["seeded_labels"],
        "updated_labels": seed_result["updated_labels"],
        "labels": seed_result["labels"],
    }


def run_payload(
    run_id: str,
    thesis: str,
    top_k: int,
    embedding_backend: str,
    vector_backend: str,
    company_count: int,
    metrics: Metrics | None = None,
) -> dict[str, Any]:
    values = metrics or Metrics()
    return {
        "Run ID": run_id,
        "Timestamp": datetime.now(UTC).isoformat(),
        "Thesis": thesis,
        "Top K": top_k,
        "Embedding Backend": embedding_backend,
        "Vector Store Backend": vector_backend,
        "Company Count": company_count,
        "Precision": values.precision,
        "Recall": values.recall,
    }


def lead_payload(
    run_id: str,
    candidate: Candidate,
    metrics: Metrics | None = None,
    label: Label | None = None,
) -> dict[str, Any]:
    values = metrics or Metrics()
    payload: dict[str, Any] = {
        "Run ID": run_id,
        "Company ID": candidate.company_id,
        "Company Name": candidate.company_name,
        "Category": candidate.category_code,
        "Status": candidate.status,
        "Country": candidate.country_code,
        "Founded Year": candidate.founded_year,
        "Funding Total USD": candidate.funding_total_usd,
        "Funding Rounds": candidate.funding_rounds,
        "Company Thesis Similarity": candidate.company_thesis_similarity,
        "Market Stage Score": candidate.market_stage_score,
        "Founder Team Similarity": candidate.founder_team_similarity,
        "Overall Score": candidate.overall_score,
        "Recommendation": candidate.recommendation,
        "Run Precision": values.precision,
        "Run Recall": values.recall,
        "Rationale": candidate.rationale,
        "Evidence": "\n".join(candidate.evidence),
        "Risks": "\n".join(candidate.risks),
        "Missing Information": "\n".join(candidate.missing_information),
    }
    if label:
        payload.update(
            {
                "Is Relevant": label.is_relevant,
                "Label Reason": label.label_reason,
                "Evaluation Outcome": evaluation_outcome(candidate, label),
            }
        )
    else:
        payload["Evaluation Outcome"] = "Unlabeled"
    return payload


def label_from_airtable(record: dict[str, Any]) -> Label | None:
    fields = record.get("fields", record)
    company_id = fields.get("Company ID")
    if not company_id:
        return None
    raw_relevant = fields.get("Is Relevant", False)
    is_relevant = (
        raw_relevant
        if isinstance(raw_relevant, bool)
        else str(raw_relevant).lower() == "true"
    )
    return Label(
        company_id=str(company_id),
        company_name=str(fields.get("Company Name", "")),
        is_relevant=is_relevant,
        label_reason=str(fields.get("Label Reason", "")),
        reviewer=str(fields.get("Reviewer", "")),
        reviewed_at=str(fields.get("Reviewed At", "")),
    )


def label_payload(label: Label) -> dict[str, Any]:
    return {
        "Company ID": label.company_id,
        "Company Name": label.company_name,
        "Is Relevant": label.is_relevant,
        "Label Reason": label.label_reason,
        "Reviewer": label.reviewer,
        "Reviewed At": label.reviewed_at,
    }


def seed_demo_labels() -> dict[str, Any]:
    if not airtable_configured():
        return {"configured": False, "message": setup_instructions(), "labels": []}

    ensure_airtable_schema()
    labels_table = _airtable_base().table(TABLES["labels"])
    existing_records = labels_table.all()
    existing_by_company_id = {
        str(record.get("fields", {}).get("Company ID")): record
        for record in existing_records
        if record.get("fields", {}).get("Company ID")
    }
    seeded = 0
    updated = 0

    for label in DEMO_LABELS:
        payload = label_payload(label)
        existing = existing_by_company_id.get(label.company_id)
        if not existing:
            labels_table.create(payload)
            seeded += 1
            continue

        fields = existing.get("fields", {})
        updates = {
            name: value
            for name, value in payload.items()
            if name not in fields or _blank(fields.get(name))
        }
        if updates:
            labels_table.update(existing["id"], updates)
            updated += 1

    labels = read_labels_from_airtable(seed_missing=False)["labels"]
    return {
        "configured": True,
        "seeded_labels": seeded,
        "updated_labels": updated,
        "labels": len(labels),
    }


def evaluation_outcome(candidate: Candidate, label: Label) -> str:
    predicted_positive = candidate.recommendation == "Review"
    actual_positive = label.is_relevant
    if predicted_positive and actual_positive:
        return "True Positive"
    if predicted_positive and not actual_positive:
        return "False Positive"
    if not predicted_positive and actual_positive:
        return "False Negative"
    return "True Negative"


def sync_to_airtable(
    run_id: str,
    thesis: str,
    candidates: list[Candidate],
    top_k: int,
    embedding_backend: str = "fastembed",
    vector_backend: str = "qdrant-local",
) -> dict[str, Any]:
    if not airtable_configured():
        return {"configured": False, "message": setup_instructions()}

    setup_airtable_base()
    base = _airtable_base()
    labels = [
        Label(**item) for item in read_labels_from_airtable(seed_missing=False)["labels"]
    ]
    labels_by_id = {label.company_id: label for label in labels}
    synced_candidates = candidates[:top_k]
    metrics = compute_precision_recall(synced_candidates, labels)
    base.table(TABLES["runs"]).create(
        run_payload(
            run_id,
            thesis,
            top_k,
            embedding_backend,
            vector_backend,
            len(synced_candidates),
            metrics,
        )
    )
    for candidate in synced_candidates:
        base.table(TABLES["leads"]).create(
            lead_payload(
                run_id,
                candidate,
                metrics,
                labels_by_id.get(candidate.company_id),
            )
        )
    return {
        "configured": True,
        "runs": 1,
        "leads": len(synced_candidates),
        "labels": len(labels),
        "metrics": metrics.model_dump(),
    }


def read_labels_from_airtable(seed_missing: bool = True) -> dict[str, Any]:
    if not airtable_configured():
        return {"configured": False, "message": setup_instructions(), "labels": []}

    ensure_airtable_schema()
    if seed_missing:
        seed_demo_labels()
    records = _airtable_base().table(TABLES["labels"]).all()
    labels = [label for record in records if (label := label_from_airtable(record))]
    return {"configured": True, "labels": [label.model_dump() for label in labels]}


def _airtable_base():
    from pyairtable import Api

    return Api(os.environ[AIRTABLE_TOKEN_ENV]).base(os.environ[AIRTABLE_BASE_ENV])


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def _create_field_if_missing(table: Any, field: dict[str, Any]) -> bool:
    try:
        table.create_field(
            field["name"],
            field["type"],
            options=field.get("options"),
        )
        return True
    except Exception as exc:
        if "DUPLICATE_OR_EMPTY_FIELD_NAME" in str(exc):
            return False
        raise
