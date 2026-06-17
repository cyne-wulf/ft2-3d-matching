from __future__ import annotations

from .models import Candidate, Label, Metrics


def compute_precision_recall(candidates: list[Candidate], labels: list[Label]) -> Metrics:
    relevant_by_id = {label.company_id: label.is_relevant for label in labels}
    predicted_positive = {
        candidate.company_id for candidate in candidates if candidate.recommendation == "Review"
    }
    actual_positive = {company_id for company_id, is_relevant in relevant_by_id.items() if is_relevant}

    true_positives = len(predicted_positive & actual_positive)
    false_positives = len(predicted_positive - actual_positive)
    false_negatives = len(actual_positive - predicted_positive)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return Metrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
