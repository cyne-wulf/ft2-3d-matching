---
name: three-dimensional-matching
description: Guide the Pi-first workshop for company/thesis fit, market/stage/category fit, and founder/team fit using local retrieval, LangGraph, and Airtable labels.
---

# Three-Dimensional Matching Workshop Skill

Use this skill when a participant asks Pi to run or explain the 3D matching workshop.

## Required stance

- Pi is the guide and reasoning surface.
- LangGraph is the auditable workflow engine.
- FastEmbed plus Qdrant Local Mode is the default local retrieval stack.
- Airtable is the human review queue and is required for final workshop acceptance.
- Synthetic LinkedIn-style founder/team data is used for compliance and reliability.
- Crunchbase CSVs are downloaded locally and are not committed.

## Scoring model

Overall score:

```text
0.45 * company_thesis_similarity
+ 0.25 * market_stage_score
+ 0.30 * founder_team_similarity
```

Recommendations are `Review`, `Watch`, and `Pass`.

## Evidence discipline

Use only supplied records, synthetic founder profiles, component scores, and Airtable labels. Do not imply access to OAuth token contents, live LinkedIn data, or live Crunchbase data.
