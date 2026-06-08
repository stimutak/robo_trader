# Project Status Reports

Periodic status briefings for stakeholders (non-day-to-day collaborators).

## Convention
- One report per file, named `YYYY-MM-DD_project_status_report.md` (+ matching `.pdf`).
- **Report #001 (2026-05-30)** is the comprehensive baseline — every feature, problem, and plan.
- Subsequent reports are **incremental**: what changed since the previous one (new features, resolved/new problems, readiness-score movement, security progress). They can reference #001 for unchanged background.

## Audience & tone
- Reader understands financial systems in theory but is not in the code daily.
- Define software/trading terms inline; keep a short glossary.
- Always distinguish **paper-mode "works"** from **validated-for-live**.

## Rendering a report to PDF
```bash
python3 reports/_render_pdf.py reports/<name>.md
```
Requires the `markdown-pdf` package (`pip install markdown-pdf`). No LaTeX needed.

## Source basis
Report #001 was built from a full read of all root + `docs/` documentation, the
`handoff/` engineering notes, the four security-audit reports, and a survey of all
source modules. For later reports, the fastest refresh is: recent `git log`, the
newest `handoff/` notes, `PRODUCTION_READINESS_PLAN.md`, and the CLAUDE.md
"Common Mistakes" table (which logs every recurring incident).
