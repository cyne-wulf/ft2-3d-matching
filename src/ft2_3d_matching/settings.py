from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"
VECTOR_STORE = ARTIFACTS / "vector_store"
REVIEW_ARTIFACTS = ARTIFACTS / "review"
FIXTURES = REPO_ROOT / "fixtures"

CRUNCHBASE_REPO = "https://github.com/datahoarder/crunchbase-october-2013"
CRUNCHBASE_RAW_BASE = (
    "https://raw.githubusercontent.com/datahoarder/crunchbase-october-2013/master"
)
CRUNCHBASE_FILES = [
    "crunchbase-companies.csv",
    "crunchbase-investments.csv",
]

DEFAULT_THESIS = (
    "Seed and Series A companies building enterprise workflow automation, "
    "AI infrastructure, internal operations tooling, compliance automation, "
    "or orchestration software for business users."
)

IDEAL_FOUNDER_PROFILE = (
    "Founders with enterprise software experience, workflow automation domain "
    "knowledge, technical leadership, go-to-market clarity, and evidence of "
    "building tools for business operations teams."
)

COLLECTION_NAME = "ft2_3d_matching_records"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, ARTIFACTS, VECTOR_STORE, REVIEW_ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)
