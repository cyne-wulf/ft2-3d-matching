from __future__ import annotations

from .models import Candidate


def evidence_prompt(candidate: Candidate) -> str:
    return (
        "Explain this lead using only supplied evidence. Do not infer unavailable facts.\n\n"
        f"Company: {candidate.company_name}\n"
        f"Recommendation: {candidate.recommendation}\n"
        f"Overall score: {candidate.overall_score:.3f}\n"
        f"Company text: {candidate.company_text}\n"
        f"Founder/team text: {candidate.founder_text}\n"
        f"Evidence: {'; '.join(candidate.evidence)}"
    )


def evaluate_with_pi(candidates: list[Candidate]) -> list[Candidate]:
    """Prepare strict evidence rationales for Pi/OpenAI OAuth evaluation.

    The repo never receives OAuth tokens. Pi reads the prompt and can rewrite or
    explain the rationale interactively; this offline path produces the same
    evidence-bound fields for verification and tests.
    """
    for candidate in candidates:
        candidate.rationale = (
            f"{candidate.company_name}: {candidate.recommendation}. "
            f"Evidence is limited to the company record, synthetic founder record, and component scores."
        )
    return candidates
