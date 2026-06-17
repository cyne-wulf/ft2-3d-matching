from __future__ import annotations

import csv
import shutil
from collections.abc import Iterator
from pathlib import Path

import requests

from .models import CompanyRecord, FounderProfile
from .settings import (
    CRUNCHBASE_FILES,
    CRUNCHBASE_RAW_BASE,
    DATA_PROCESSED,
    DATA_RAW,
    FIXTURES,
    ensure_dirs,
)


def download_crunchbase_snapshot(dry_run: bool = False) -> list[Path]:
    """Download the small set of Crunchbase CSVs used by the workshop."""
    ensure_dirs()
    targets = [DATA_RAW / filename for filename in CRUNCHBASE_FILES]
    if dry_run:
        return targets

    for filename, target in zip(CRUNCHBASE_FILES, targets, strict=True):
        response = requests.get(f"{CRUNCHBASE_RAW_BASE}/{filename}", timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
    return targets


def verify_data_files(raw_dir: Path = DATA_RAW) -> dict[str, object]:
    files = {filename: raw_dir / filename for filename in CRUNCHBASE_FILES}
    present = {name: path.exists() and path.stat().st_size > 0 for name, path in files.items()}
    return {
        "passed": all(present.values()),
        "files": {name: str(path) for name, path in files.items()},
        "present": present,
    }


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_company_csv(path: Path, limit: int | None = None) -> list[CompanyRecord]:
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return list(_iter_company_csv(path, encoding, limit))
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    return []


def _iter_company_csv(path: Path, encoding: str, limit: int | None = None) -> Iterator[CompanyRecord]:
    records: list[CompanyRecord] = []
    with path.open(newline="", encoding=encoding) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            records.append(_company_from_row(index, row))
    yield from records


def _company_from_row(index: int, row: dict[str, str | None]) -> CompanyRecord:
    name = row.get("name") or row.get("company_name") or row.get("permalink") or f"company-{index}"
    return CompanyRecord(
        company_id=row.get("company_id") or row.get("permalink") or f"company-{index}",
        name=name,
        category_code=row.get("category_code", ""),
        status=row.get("status", ""),
        country_code=row.get("country_code", ""),
        founded_year=_parse_int(row.get("founded_year")),
        funding_total_usd=_parse_float(row.get("funding_total_usd")),
        funding_rounds=_parse_int(row.get("funding_rounds")) or 0,
        description=row.get("description") or row.get("overview") or "",
    )


def prepare_sample(limit: int = 6, source: Path | None = None) -> Path:
    ensure_dirs()
    source_path = source or DATA_RAW / "crunchbase-companies.csv"
    if not source_path.exists():
        source_path = FIXTURES / "sample_companies.csv"

    records = read_company_csv(source_path, limit=limit)
    output = DATA_PROCESSED / "sample_companies.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CompanyRecord.model_fields))
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())
    return output


def load_prepared_companies(path: Path | None = None) -> list[CompanyRecord]:
    prepared = path or DATA_PROCESSED / "sample_companies.csv"
    if not prepared.exists():
        prepare_sample()
    return read_company_csv(prepared)


def synthesize_founder_profiles(companies: list[CompanyRecord]) -> list[FounderProfile]:
    profiles: list[FounderProfile] = []
    for index, company in enumerate(companies):
        enterprise_signal = any(
            term in f"{company.category_code} {company.description}".lower()
            for term in ["enterprise", "workflow", "automation", "compliance", "infrastructure", "software"]
        )
        if enterprise_signal:
            skills = ["enterprise sales", "workflow design", "technical leadership"]
            profile = (
                f"Founder of {company.name}. Previously led enterprise software projects, "
                "built internal operations tooling, and sold technical products to business teams."
            )
        else:
            skills = ["product management", "partnerships", "consumer growth"]
            profile = (
                f"Founder of {company.name}. Background in product launches, partnerships, "
                "and customer acquisition outside deep enterprise infrastructure."
            )
        profiles.append(
            FounderProfile(
                company_id=company.company_id,
                founder_name=f"{company.name.split()[0]} Founder {index + 1}",
                title="CEO and Co-founder",
                profile=profile,
                skills=skills,
            )
        )
    return profiles


def reset_local_outputs() -> None:
    for path in [DATA_PROCESSED, DATA_RAW]:
        path.mkdir(parents=True, exist_ok=True)
    if (DATA_PROCESSED / "sample_companies.csv").exists():
        (DATA_PROCESSED / "sample_companies.csv").unlink()
    vector_store = Path("artifacts/vector_store")
    if vector_store.exists():
        shutil.rmtree(vector_store)
