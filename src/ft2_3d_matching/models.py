from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class CompanyRecord(BaseModel):
    company_id: str
    name: str
    category_code: str = ""
    status: str = ""
    country_code: str = ""
    founded_year: int | None = None
    funding_total_usd: float = 0.0
    funding_rounds: int = 0
    description: str = ""


class FounderProfile(BaseModel):
    company_id: str
    founder_name: str
    title: str
    profile: str
    skills: list[str] = Field(default_factory=list)


class EnrichedRecord(BaseModel):
    company: CompanyRecord
    founder: FounderProfile
    company_text: str
    founder_text: str


class Candidate(BaseModel):
    company_id: str
    company_name: str
    category_code: str
    status: str
    country_code: str
    founded_year: int | None
    funding_total_usd: float
    funding_rounds: int
    company_text: str
    founder_text: str
    company_thesis_similarity: float
    founder_team_similarity: float = 0.0
    market_stage_score: float = 0.0
    overall_score: float = 0.0
    recommendation: str = "Watch"
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class Label(BaseModel):
    company_id: str
    company_name: str = ""
    is_relevant: bool
    label_reason: str = ""
    reviewer: str = ""
    reviewed_at: str = ""


class Metrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class Verification(BaseModel):
    name: str
    passed: bool
    detail: str


class GraphState(TypedDict, total=False):
    run_id: str
    timestamp: str
    thesis: str
    top_k: int
    raw_data: dict[str, list[dict[str, Any]]]
    companies: list[dict[str, Any]]
    founders: list[dict[str, Any]]
    records: list[dict[str, Any]]
    embeddings: dict[str, list[float]]
    vector_store_status: dict[str, Any]
    candidates: list[dict[str, Any]]
    scored_leads: list[dict[str, Any]]
    airtable_records: dict[str, Any]
    labels: list[dict[str, Any]]
    metrics: dict[str, Any]
    logs: list[str]
    verification: list[dict[str, Any]]
    review_workspace_path: str
