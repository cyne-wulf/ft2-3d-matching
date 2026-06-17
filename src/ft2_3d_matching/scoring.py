from __future__ import annotations

from .embeddings import Embedder, cosine_similarity
from .models import Candidate
from .settings import IDEAL_FOUNDER_PROFILE

TARGET_TERMS = {
    "software",
    "enterprise",
    "ai",
    "automation",
    "workflow",
    "compliance",
    "infrastructure",
    "orchestration",
    "operations",
}


def market_stage_score(candidate: Candidate) -> float:
    text = f"{candidate.category_code} {candidate.company_text}".lower()
    category_score = 1.0 if any(term in text for term in TARGET_TERMS) else 0.25
    status_score = 1.0 if candidate.status.lower() in {"operating", "ipo"} else 0.3
    geography_score = 1.0 if candidate.country_code in {"USA", "CAN", "GBR"} else 0.65
    rounds_score = 1.0 if 1 <= candidate.funding_rounds <= 3 else 0.55
    funding_score = 1.0 if 1_000_000 <= candidate.funding_total_usd <= 15_000_000 else 0.6
    founded_score = 1.0 if candidate.founded_year and candidate.founded_year >= 2018 else 0.55
    return round(
        0.25 * category_score
        + 0.15 * status_score
        + 0.15 * geography_score
        + 0.15 * rounds_score
        + 0.15 * funding_score
        + 0.15 * founded_score,
        4,
    )


def score_founder_fit(candidates: list[Candidate], embedder: Embedder) -> list[Candidate]:
    if not candidates:
        return []
    vectors = embedder.embed([IDEAL_FOUNDER_PROFILE] + [candidate.founder_text for candidate in candidates])
    ideal = vectors[0]
    for candidate, vector in zip(candidates, vectors[1:], strict=True):
        candidate.founder_team_similarity = round(cosine_similarity(ideal, vector), 4)
    return candidates


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        candidate.market_stage_score = market_stage_score(candidate)
        candidate.overall_score = round(
            0.45 * candidate.company_thesis_similarity
            + 0.25 * candidate.market_stage_score
            + 0.30 * candidate.founder_team_similarity,
            4,
        )
        if candidate.overall_score >= 0.72:
            candidate.recommendation = "Review"
        elif candidate.overall_score >= 0.52:
            candidate.recommendation = "Watch"
        else:
            candidate.recommendation = "Pass"
        candidate.evidence = [
            f"Semantic thesis similarity: {candidate.company_thesis_similarity:.3f}",
            f"Market/stage score: {candidate.market_stage_score:.3f}",
            f"Founder/team similarity: {candidate.founder_team_similarity:.3f}",
        ]
        candidate.risks = _risks(candidate)
        candidate.missing_information = ["Customer traction", "Current round details"]
        candidate.rationale = (
            f"{candidate.company_name} is a {candidate.recommendation.lower()} because the combined "
            f"3D score is {candidate.overall_score:.3f}."
        )
    return sorted(candidates, key=lambda item: item.overall_score, reverse=True)


def _risks(candidate: Candidate) -> list[str]:
    risks: list[str] = []
    if candidate.market_stage_score < 0.65:
        risks.append("Structured market/stage signal is below the review threshold.")
    if candidate.founder_team_similarity < 0.45:
        risks.append("Synthetic founder profile has weak fit with the ideal profile.")
    if not risks:
        risks.append("Scores are retrieval triage signals and need human diligence.")
    return risks
