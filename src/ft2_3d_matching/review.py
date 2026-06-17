from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Template

from .models import Candidate, Metrics
from .settings import REVIEW_ARTIFACTS, ensure_dirs

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deal Review Workspace</title>
  <style>
    :root { color-scheme: light; --ink: #18202a; --muted: #586274; --line: #d8dee8; --panel: #f7f9fc; --review: #0f7b63; --watch: #8a5b00; --pass: #9b2f37; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #ffffff; }
    header { padding: 28px 36px 18px; border-bottom: 1px solid var(--line); background: #f3f6fa; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    .sub { margin: 0; color: var(--muted); max-width: 980px; line-height: 1.45; }
    main { padding: 24px 36px 40px; }
    .status { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin-bottom: 22px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: var(--panel); min-height: 78px; }
    .metric b { display: block; font-size: 20px; margin-top: 8px; font-variant-numeric: tabular-nums; }
    table { width: 100%; border-collapse: collapse; border: 1px solid var(--line); table-layout: fixed; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 14px; }
    th { background: #eef2f7; color: #2d3848; }
    td.num { font-variant-numeric: tabular-nums; }
    .rec { font-weight: 700; }
    .Review { color: var(--review); }
    .Watch { color: var(--watch); }
    .Pass { color: var(--pass); }
    .evidence { color: var(--muted); line-height: 1.4; }
    @media (max-width: 900px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .status { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      table { display: block; overflow-x: auto; white-space: nowrap; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Deal Review Workspace</h1>
    <p class="sub">{{ thesis }}</p>
  </header>
  <main>
    <section class="status" aria-label="Pipeline status">
      <div class="metric">Run ID<b>{{ run_id }}</b></div>
      <div class="metric">Leads<b>{{ candidates|length }}</b></div>
      <div class="metric">Precision<b>{{ "%.2f"|format(metrics.precision) }}</b></div>
      <div class="metric">Recall<b>{{ "%.2f"|format(metrics.recall) }}</b></div>
      <div class="metric">Generated<b>{{ generated_at }}</b></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Company</th><th>Recommendation</th><th>Overall</th><th>Company Fit</th><th>Market</th><th>Founder</th><th>Evidence</th><th>Risks</th>
        </tr>
      </thead>
      <tbody>
        {% for lead in candidates %}
        <tr>
          <td>{{ lead.company_name }}<br><span class="evidence">{{ lead.category_code }} · {{ lead.country_code }}</span></td>
          <td class="rec {{ lead.recommendation }}">{{ lead.recommendation }}</td>
          <td class="num">{{ "%.3f"|format(lead.overall_score) }}</td>
          <td class="num">{{ "%.3f"|format(lead.company_thesis_similarity) }}</td>
          <td class="num">{{ "%.3f"|format(lead.market_stage_score) }}</td>
          <td class="num">{{ "%.3f"|format(lead.founder_team_similarity) }}</td>
          <td class="evidence">{{ lead.evidence|join("<br>") }}</td>
          <td class="evidence">{{ lead.risks|join("<br>") }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def render_review_workspace(
    run_id: str,
    thesis: str,
    candidates: list[Candidate],
    metrics: Metrics | None = None,
    output: Path | None = None,
) -> Path:
    ensure_dirs()
    target = output or REVIEW_ARTIFACTS / f"deal-review-workspace-{run_id}.html"
    html = Template(TEMPLATE).render(
        run_id=run_id,
        thesis=thesis,
        candidates=candidates,
        metrics=metrics or Metrics(),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    target.write_text(html, encoding="utf-8")
    return target
