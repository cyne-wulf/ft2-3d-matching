from __future__ import annotations

import sys
from types import SimpleNamespace

from ft2_3d_matching.airtable import (
    AIRTABLE_ENV,
    AIRTABLE_TABLE_SCHEMAS,
    DEMO_LABELS,
    airtable_configured,
    ensure_airtable_schema,
    read_labels_from_airtable,
    label_from_airtable,
    lead_payload,
    setup_airtable_base,
    setup_instructions,
    sync_to_airtable,
)
from ft2_3d_matching.metrics import compute_precision_recall
from ft2_3d_matching.models import Candidate, Label
from ft2_3d_matching.scoring import market_stage_score, rank_candidates


def candidate(company_id: str = "c001", similarity: float = 0.8) -> Candidate:
    return Candidate(
        company_id=company_id,
        company_name="OpsPilot",
        category_code="software",
        status="operating",
        country_code="USA",
        founded_year=2021,
        funding_total_usd=4_200_000,
        funding_rounds=2,
        company_text="enterprise workflow automation for internal operations",
        founder_text="founder with enterprise software background",
        company_thesis_similarity=similarity,
        founder_team_similarity=0.7,
    )


def test_market_stage_scoring_rewards_target_profile():
    score = market_stage_score(candidate())

    assert score > 0.9


def test_hybrid_ranking_sets_recommendation_and_evidence():
    ranked = rank_candidates([candidate()])

    assert ranked[0].recommendation == "Review"
    assert ranked[0].overall_score > 0.7
    assert ranked[0].evidence


def test_airtable_payload_mapping_and_label_read():
    lead = rank_candidates([candidate()])[0]
    metrics = compute_precision_recall([lead], [Label(company_id="c001", is_relevant=True)])
    label = Label(company_id="c001", is_relevant=True, label_reason="Strong fit")
    payload = lead_payload("run-1", lead, metrics, label)
    parsed = label_from_airtable({"fields": {"Company ID": "c001", "Is Relevant": True}})

    assert payload["Company ID"] == "c001"
    assert payload["Recommendation"] == "Review"
    assert payload["Run Precision"] == 1.0
    assert payload["Run Recall"] == 1.0
    assert payload["Evaluation Outcome"] == "True Positive"
    assert parsed is not None
    assert parsed.is_relevant is True


def test_airtable_configuration_uses_personal_access_tokens(monkeypatch):
    legacy_env = "AIRTABLE_" + "API_KEY"
    monkeypatch.delenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv(legacy_env, raising=False)
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")

    assert AIRTABLE_ENV == ["AIRTABLE_PERSONAL_ACCESS_TOKEN", "AIRTABLE_BASE_ID"]
    assert not airtable_configured()

    monkeypatch.setenv(legacy_env, "legacy-api-key")
    assert not airtable_configured()

    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    assert airtable_configured()


def test_airtable_setup_instructions_describe_pat_scope():
    instructions = setup_instructions()

    assert "personal access token" in instructions
    assert "data.records:read" in instructions
    assert "data.records:write" in instructions
    assert "schema.bases:read" in instructions
    assert "schema.bases:write" in instructions
    assert "setup-airtable creates the Runs, Leads, and Labels tables" in instructions
    assert "AIRTABLE_PERSONAL_ACCESS_TOKEN" in instructions
    assert "AIRTABLE_" + "API_KEY" not in instructions


def test_airtable_schema_bootstrap_creates_required_tables(monkeypatch):
    base = FakeBase([])
    install_fake_airtable(monkeypatch, base)
    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")

    result = ensure_airtable_schema()

    assert result["configured"] is True
    assert result["created_tables"] == ["Runs", "Leads", "Labels"]
    assert result["created_fields"] == []
    assert set(base.tables_by_name) >= {"Runs", "Leads", "Labels"}
    assert [field["name"] for field in base.tables_by_name["Runs"].created_with] == [
        field["name"] for field in AIRTABLE_TABLE_SCHEMAS["Runs"]
    ]


def test_airtable_schema_bootstrap_without_credentials_returns_setup(monkeypatch):
    monkeypatch.delenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("AIRTABLE_BASE_ID", raising=False)
    monkeypatch.setitem(sys.modules, "pyairtable", None)

    result = ensure_airtable_schema()

    assert result["configured"] is False
    assert "setup-airtable creates the Runs, Leads, and Labels tables" in result["message"]


def test_airtable_schema_bootstrap_repairs_existing_tables(monkeypatch):
    base = FakeBase([FakeTable("Runs", ["Run ID"])])
    install_fake_airtable(monkeypatch, base)
    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")

    result = ensure_airtable_schema()

    assert result["created_tables"] == ["Leads", "Labels"]
    assert "Runs.Timestamp" in result["created_fields"]
    assert "Runs.Recall" in result["created_fields"]
    assert "Run ID" in base.tables_by_name["Runs"].field_names
    assert "Timestamp" in base.tables_by_name["Runs"].field_names


def test_airtable_setup_seeds_demo_labels_idempotently(monkeypatch):
    base = FakeBase([])
    install_fake_airtable(monkeypatch, base)
    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")

    first = setup_airtable_base()
    second = setup_airtable_base()

    assert first["seeded_labels"] == len(DEMO_LABELS)
    assert first["updated_labels"] == 0
    assert second["seeded_labels"] == 0
    assert second["updated_labels"] == 0
    assert len(base.tables_by_name["Labels"].records) == len(DEMO_LABELS)
    assert len(read_labels_from_airtable()["labels"]) == len(DEMO_LABELS)


def test_airtable_setup_preserves_human_label_edits(monkeypatch):
    labels = FakeTable("Labels", ["Company ID", "Is Relevant", "Reviewer"])
    labels.create({"Company ID": "c001", "Is Relevant": False, "Reviewer": "Human"})
    base = FakeBase([labels])
    install_fake_airtable(monkeypatch, base)
    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")

    result = setup_airtable_base()
    fields = base.tables_by_name["Labels"].records[0]["fields"]

    assert result["seeded_labels"] == len(DEMO_LABELS) - 1
    assert fields["Is Relevant"] is False
    assert fields["Reviewer"] == "Human"
    assert fields["Company Name"] == "OpsPilot"
    assert fields["Label Reason"] == "Strong workflow automation fit."


def test_sync_airtable_writes_metrics_and_label_context(monkeypatch):
    base = FakeBase([])
    install_fake_airtable(monkeypatch, base)
    monkeypatch.setenv("AIRTABLE_PERSONAL_ACCESS_TOKEN", "token-from-test")
    monkeypatch.setenv("AIRTABLE_BASE_ID", "app123")
    reviewed = rank_candidates([candidate("c001"), candidate("c003", similarity=0.8)])
    reviewed[0].recommendation = "Review"
    reviewed[1].recommendation = "Review"

    result = sync_to_airtable("run-1", "workflow thesis", reviewed, top_k=2)

    assert result["runs"] == 1
    assert result["leads"] == 2
    assert result["labels"] == len(DEMO_LABELS)
    assert result["metrics"]["precision"] == 0.5
    assert result["metrics"]["recall"] == 0.3333
    run_fields = base.tables_by_name["Runs"].records[0]["fields"]
    lead_fields = base.tables_by_name["Leads"].records[0]["fields"]
    assert run_fields["Precision"] == 0.5
    assert run_fields["Recall"] == 0.3333
    assert lead_fields["Run Precision"] == 0.5
    assert lead_fields["Run Recall"] == 0.3333
    assert lead_fields["Is Relevant"] is True
    assert lead_fields["Evaluation Outcome"] == "True Positive"


def test_precision_recall():
    reviewed = rank_candidates([candidate("c001"), candidate("c002", similarity=0.3)])
    reviewed[1].recommendation = "Pass"
    metrics = compute_precision_recall(
        reviewed,
        [
            Label(company_id="c001", is_relevant=True),
            Label(company_id="c003", is_relevant=True),
        ],
    )

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5


class FakeTable:
    def __init__(self, name: str, field_names: list[str] | None = None):
        self.name = name
        self.field_names = field_names or []
        self.created_with: list[dict[str, object]] = []
        self.records: list[dict[str, object]] = []

    def schema(self, *, force: bool = False):
        return SimpleNamespace(
            fields=[SimpleNamespace(name=field_name) for field_name in self.field_names]
        )

    def create_field(self, name: str, field_type: str, options=None):
        self.field_names.append(name)
        self.created_with.append({"name": name, "type": field_type, "options": options})

    def all(self):
        return self.records

    def create(self, fields: dict[str, object]):
        record = {"id": f"rec{len(self.records) + 1}", "fields": fields.copy()}
        self.records.append(record)
        return record

    def update(self, record_id: str, fields: dict[str, object]):
        record = next(record for record in self.records if record["id"] == record_id)
        record["fields"].update(fields)
        return record


class FakeBase:
    def __init__(self, tables: list[FakeTable]):
        self.tables_by_name = {table.name: table for table in tables}

    def tables(self, *, force: bool = False):
        return list(self.tables_by_name.values())

    def create_table(self, name: str, fields: list[dict[str, object]]):
        table = FakeTable(name, [str(field["name"]) for field in fields])
        table.created_with = fields
        self.tables_by_name[name] = table
        return table

    def table(self, name: str):
        return self.tables_by_name[name]


class FakeApi:
    def __init__(self, token: str, base: FakeBase):
        self.token = token
        self._base = base

    def base(self, base_id: str):
        return self._base


def install_fake_airtable(monkeypatch, base: FakeBase) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pyairtable",
        SimpleNamespace(Api=lambda token: FakeApi(token, base)),
    )
