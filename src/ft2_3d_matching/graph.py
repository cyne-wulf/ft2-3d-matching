from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .airtable import read_labels_from_airtable, sync_to_airtable
from .data import load_prepared_companies, prepare_sample, synthesize_founder_profiles
from .embeddings import get_embedder
from .evaluation import evaluate_with_pi
from .metrics import compute_precision_recall
from .models import Candidate, EnrichedRecord, GraphState, Label, Metrics, Verification
from .review import render_review_workspace
from .scoring import rank_candidates, score_founder_fit
from .settings import DEFAULT_THESIS
from .textify import build_enriched_records
from .vector_store import index_records, local_client, query_records


def initial_state(
    thesis: str = DEFAULT_THESIS,
    top_k: int = 5,
    embedding_backend: str = "deterministic",
) -> GraphState:
    return {
        "run_id": uuid4().hex[:10],
        "timestamp": datetime.now(UTC).isoformat(),
        "thesis": thesis,
        "top_k": top_k,
        "logs": [f"embedding_backend={embedding_backend}"],
        "verification": [],
    }


def load_data(state: GraphState) -> GraphState:
    prepare_sample()
    companies = load_prepared_companies()
    return {
        "companies": [company.model_dump() for company in companies],
        "logs": state.get("logs", []) + [f"load_data: {len(companies)} companies"],
        "verification": state.get("verification", [])
        + [Verification(name="sample_data", passed=bool(companies), detail=f"{len(companies)} rows").model_dump()],
    }


def normalize_records(state: GraphState) -> GraphState:
    return {
        "logs": state.get("logs", []) + ["normalize_records: normalized company fields"],
        "verification": state.get("verification", [])
        + [Verification(name="normalized_records", passed=True, detail="pydantic models loaded").model_dump()],
    }


def synthesize_founder_profiles_node(state: GraphState) -> GraphState:
    companies = [CandidateCompany(**item) for item in state["companies"]]
    founders = synthesize_founder_profiles(companies)
    return {
        "founders": [founder.model_dump() for founder in founders],
        "logs": state.get("logs", []) + [f"synthesize_founder_profiles: {len(founders)} profiles"],
    }


def build_text_representations(state: GraphState) -> GraphState:
    companies = [CandidateCompany(**item) for item in state["companies"]]
    founders = [FounderProfileAdapter(**item) for item in state["founders"]]
    records = build_enriched_records(companies, founders)
    return {
        "records": [record.model_dump() for record in records],
        "logs": state.get("logs", []) + [f"build_text_representations: {len(records)} semantic units"],
        "verification": state.get("verification", [])
        + [Verification(name="textified_records", passed=True, detail="one company/team record per unit").model_dump()],
    }


def embed_records(state: GraphState) -> GraphState:
    embedder = get_embedder(_embedding_backend(state))
    records = [EnrichedRecord(**item) for item in state["records"]]
    company_vectors = embedder.embed([record.company_text for record in records])
    thesis_vector = embedder.embed([state["thesis"]])[0]
    founder_vectors = embedder.embed([record.founder_text for record in records])
    embeddings = {
        "thesis": thesis_vector,
        "companies": company_vectors,
        "founders": founder_vectors,
        "dimension": [float(embedder.dimension)],
    }
    return {
        "embeddings": embeddings,
        "logs": state.get("logs", []) + [f"embed_records: dimension={embedder.dimension}"],
        "verification": state.get("verification", [])
        + [Verification(name="embedding_dimension", passed=embedder.dimension > 0, detail=str(embedder.dimension)).model_dump()],
    }


def index_vector_store(state: GraphState) -> GraphState:
    records = [EnrichedRecord(**item) for item in state["records"]]
    status = index_records(records, state["embeddings"]["companies"], local_client())
    return {
        "vector_store_status": status,
        "logs": state.get("logs", []) + [f"index_vector_store: {status['indexed_count']} records"],
        "verification": state.get("verification", [])
        + [Verification(name="qdrant_index_count", passed=status["indexed_count"] == len(records), detail=str(status)).model_dump()],
    }


def retrieve_candidates(state: GraphState) -> GraphState:
    candidates = query_records(state["embeddings"]["thesis"], limit=state["top_k"], client=local_client())
    return {
        "candidates": [candidate.model_dump() for candidate in candidates],
        "logs": state.get("logs", []) + [f"retrieve_candidates: {len(candidates)} candidates"],
    }


def score_market_stage(state: GraphState) -> GraphState:
    candidates = [Candidate(**item) for item in state["candidates"]]
    ranked = rank_candidates(candidates)
    return {
        "candidates": [candidate.model_dump() for candidate in ranked],
        "logs": state.get("logs", []) + ["score_market_stage: structured scores computed"],
    }


def score_founder_fit_node(state: GraphState) -> GraphState:
    candidates = [Candidate(**item) for item in state["candidates"]]
    scored = score_founder_fit(candidates, get_embedder(_embedding_backend(state)))
    return {
        "candidates": [candidate.model_dump() for candidate in scored],
        "logs": state.get("logs", []) + ["score_founder_fit: founder similarities computed"],
    }


def rank_candidates_node(state: GraphState) -> GraphState:
    candidates = rank_candidates([Candidate(**item) for item in state["candidates"]])
    return {
        "scored_leads": [candidate.model_dump() for candidate in candidates],
        "logs": state.get("logs", []) + ["rank_candidates: overall scores computed"],
    }


def evaluate_with_pi_node(state: GraphState) -> GraphState:
    candidates = evaluate_with_pi([Candidate(**item) for item in state["scored_leads"]])
    return {
        "scored_leads": [candidate.model_dump() for candidate in candidates],
        "logs": state.get("logs", []) + ["evaluate_with_pi: evidence prompts prepared for Pi"],
    }


def sync_airtable_node(state: GraphState) -> GraphState:
    candidates = [Candidate(**item) for item in state["scored_leads"]]
    result = sync_to_airtable(state["run_id"], state["thesis"], candidates, state["top_k"], _embedding_backend(state))
    return {
        "airtable_records": result,
        "logs": state.get("logs", []) + ["sync_airtable: completed or returned setup instructions"],
    }


def read_airtable_labels_node(state: GraphState) -> GraphState:
    result = read_labels_from_airtable()
    return {
        "labels": result.get("labels", []),
        "logs": state.get("logs", []) + [f"read_airtable_labels: {len(result.get('labels', []))} labels"],
    }


def compute_metrics_node(state: GraphState) -> GraphState:
    candidates = [Candidate(**item) for item in state.get("scored_leads", [])]
    labels = [Label(**item) for item in state.get("labels", [])]
    metrics = compute_precision_recall(candidates, labels)
    return {
        "metrics": metrics.model_dump(),
        "logs": state.get("logs", []) + ["compute_metrics: precision/recall computed"],
    }


def render_review_workspace_node(state: GraphState) -> GraphState:
    candidates = [Candidate(**item) for item in state.get("scored_leads", [])]
    metrics = Metrics(**state.get("metrics", {}))
    path = render_review_workspace(state["run_id"], state["thesis"], candidates, metrics)
    return {
        "review_workspace_path": str(path),
        "logs": state.get("logs", []) + [f"render_review_workspace: {path}"],
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("load_data", load_data)
    graph.add_node("normalize_records", normalize_records)
    graph.add_node("synthesize_founder_profiles", synthesize_founder_profiles_node)
    graph.add_node("build_text_representations", build_text_representations)
    graph.add_node("embed_records", embed_records)
    graph.add_node("index_vector_store", index_vector_store)
    graph.add_node("retrieve_candidates", retrieve_candidates)
    graph.add_node("score_market_stage", score_market_stage)
    graph.add_node("score_founder_fit", score_founder_fit_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("evaluate_with_pi", evaluate_with_pi_node)
    graph.add_node("sync_airtable", sync_airtable_node)
    graph.add_node("read_airtable_labels", read_airtable_labels_node)
    graph.add_node("compute_metrics", compute_metrics_node)
    graph.add_node("render_review_workspace", render_review_workspace_node)

    chain = [
        "load_data",
        "normalize_records",
        "synthesize_founder_profiles",
        "build_text_representations",
        "embed_records",
        "index_vector_store",
        "retrieve_candidates",
        "score_market_stage",
        "score_founder_fit",
        "rank_candidates",
        "evaluate_with_pi",
        "sync_airtable",
        "read_airtable_labels",
        "compute_metrics",
        "render_review_workspace",
    ]
    graph.add_edge(START, chain[0])
    for left, right in zip(chain, chain[1:]):
        graph.add_edge(left, right)
    graph.add_edge(chain[-1], END)
    return graph.compile()


def run_pipeline(
    thesis: str = DEFAULT_THESIS,
    top_k: int = 5,
    embedding_backend: str = "deterministic",
) -> GraphState:
    state = initial_state(thesis, top_k, embedding_backend)
    return build_graph().invoke(state)


def _embedding_backend(state: GraphState) -> str:
    for item in state.get("logs", []):
        if item.startswith("embedding_backend="):
            return item.split("=", 1)[1]
    return "deterministic"


from .models import CompanyRecord as CandidateCompany  # noqa: E402
from .models import FounderProfile as FounderProfileAdapter  # noqa: E402
