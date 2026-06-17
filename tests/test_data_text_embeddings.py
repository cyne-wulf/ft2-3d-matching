from __future__ import annotations

from pathlib import Path

import pytest

from ft2_3d_matching.data import (
    download_crunchbase_snapshot,
    prepare_sample,
    read_company_csv,
    synthesize_founder_profiles,
)
from ft2_3d_matching.embeddings import DeterministicEmbedder, cosine_similarity
from ft2_3d_matching.settings import FIXTURES
from ft2_3d_matching.textify import build_enriched_records, company_to_text


def test_downloader_dry_run_lists_expected_files():
    targets = download_crunchbase_snapshot(dry_run=True)

    assert [path.name for path in targets] == [
        "crunchbase-companies.csv",
        "crunchbase-investments.csv",
    ]


def test_csv_parsing_and_sample_preparation():
    records = read_company_csv(FIXTURES / "sample_companies.csv")

    assert records[0].company_id == "c001"
    assert records[0].funding_rounds == 2
    assert prepare_sample(limit=2).exists()


def test_csv_parsing_falls_back_for_historical_non_utf8_exports(tmp_path: Path):
    csv_path = tmp_path / "companies.csv"
    csv_path.write_bytes(
        b"company_id,name,category_code,status,country_code,founded_year,funding_total_usd,funding_rounds,description\r"
        b"c-latin,Caf\xe9Co,software,operating,USA,2012,1000000,1,Older export with cp1252 text\r"
    )

    records = read_company_csv(csv_path)

    assert records[0].name == "CaféCo"


def test_founder_synthesis_and_textification():
    companies = read_company_csv(FIXTURES / "sample_companies.csv")[:1]
    founders = synthesize_founder_profiles(companies)
    enriched = build_enriched_records(companies, founders)

    assert "Workflow automation" in company_to_text(companies[0])
    assert enriched[0].company.company_id == enriched[0].founder.company_id
    assert "enterprise software" in enriched[0].founder_text


def test_deterministic_embedder_and_cosine_similarity():
    embedder = DeterministicEmbedder(dimension=16)
    left, right = embedder.embed(["enterprise workflow automation", "enterprise workflow automation"])

    assert len(left) == 16
    assert cosine_similarity(left, right) == pytest.approx(1.0)
