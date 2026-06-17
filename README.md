# Pi-First 3D Deal Matching Workshop

This repo gives you a clone-and-run pipeline for turning company records and synthetic founder/team profiles into semantic text, local embeddings, vector search results, scored leads, Airtable review rows, precision/recall metrics, and a Deal Review Workspace.

Pi is the guide. LangGraph runs the auditable workflow. FastEmbed and Qdrant Local Mode provide local retrieval. Airtable is the human review queue.

## What You Will Run

The default thesis is seed and Series A companies building enterprise workflow automation, AI infrastructure, internal operations tooling, compliance automation, or orchestration software for business users.

The "3D" part means each company is matched from three angles instead of treating one embedding score as the whole answer:

- `Company thesis fit` compares the company's semantic description with the workshop investment thesis.
- `Market/stage fit` checks structured signals like category, operating status, geography, funding range, funding rounds, and founding year.
- `Founder/team fit` compares synthetic founder/team profiles with the ideal founder profile for the thesis.

The matching model uses three dimensions:

```text
overall_score =
    0.45 * company_thesis_similarity
  + 0.25 * market_stage_score
  + 0.30 * founder_team_similarity
```

Recommendations are `Review`, `Watch`, and `Pass`. Scores are triage signals for review, not investment truth.

## Setup

Install Python 3.11 or newer, then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Check that the repo can see the Pi workshop configuration:

```bash
pi-check
```

The important safety field is:

```json
"oauth_token_contents_read": false
```

Log in to Pi through the normal Pi flow. This repo checks setup signals, but it never reads, stores, or prints OpenAI OAuth token contents.

Start Pi and say:

```text
Run the 3D matching workshop.
```

## Checkpoints

Run the local commands in this order. These work from a fresh clone after the
Python install step and do not require Airtable credentials:

```bash
pi-check
download-data --dry-run
verify-data
prepare-sample
inspect-record
textify-record
vectorize-sample
start-vector-store
index-records
query-vector-store
compare-vectors
score-market-stage
rank-candidates
evaluate-leads
render-review-workspace
```

Use deterministic embeddings when you want a fast smoke test without downloading the FastEmbed model:

```bash
rank-candidates --backend deterministic --top-k 3
```

The real workshop path uses FastEmbed with `BAAI/bge-small-en-v1.5` and Qdrant Local Mode.

## Data And Privacy

`download-data --dry-run` shows which historical Crunchbase CSVs would be downloaded. `download-data` writes them into ignored local storage under `data/raw/`.

If you do not download the CSVs, `prepare-sample` falls back to the committed tiny fixture in `fixtures/sample_companies.csv`.

Founder/team records are synthetic LinkedIn-style profiles. This workshop does not scrape LinkedIn, call a live LinkedIn API, or claim live Crunchbase access.

Generated outputs stay local:

- prepared sample data: `data/processed/`
- Qdrant Local Mode files: `artifacts/vector_store/`
- Deal Review Workspace HTML: `artifacts/review/`

Those directories are ignored except for placeholder files.

## Airtable

The only manual Airtable work for the demo is creating a default base and a
personal access token, then pasting those two values into `.env`. The agent
creates the workshop tables and seeds demo review labels.

Create a default Airtable base. The workshop will create the `Runs`, `Leads`,
and `Labels` tables for you.

Create an Airtable personal access token with `data.records:read`,
`data.records:write`, `schema.bases:read`, and `schema.bases:write` scopes, and
grant it access to that base. Then copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder values:

```bash
AIRTABLE_PERSONAL_ACCESS_TOKEN=pat_your_personal_access_token_here
AIRTABLE_BASE_ID=app_your_airtable_base_id_here
```

Copying is recommended so `.env.example` remains available as a template. Renaming
also works with `mv .env.example .env`, but it removes the template from your local
checkout.

Then run:

```bash
setup-airtable
verify-airtable
sync-airtable
read-airtable-labels
compute-metrics
```

`setup-airtable` creates or repairs the `Runs`, `Leads`, and `Labels` tables
from a fresh default base and seeds editable demo labels. `sync-airtable` writes
the run, lead rows, label context, and precision/recall metrics, so Airtable has
a complete review view without manual table setup.

If Airtable credentials are missing, Airtable sync returns setup instructions,
labels are treated as empty, and the earlier local checkpoints still work.

## Tests

Run the deterministic tests:

```bash
pytest
```

Run a quick CLI smoke test:

```bash
python -m ft2_3d_matching.cli rank-candidates --backend deterministic --top-k 2
```

## References

- [datahoarder/crunchbase-october-2013](https://github.com/datahoarder/crunchbase-october-2013)
- [Qdrant FastEmbed documentation](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/)
- [Qdrant Local Quickstart](https://qdrant.tech/documentation/quickstart/)
- [LangGraph StateGraph reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [pyAirtable documentation](https://pyairtable.readthedocs.io/en/stable/)
