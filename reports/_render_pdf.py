#!/usr/bin/env python3
"""Render a status-report markdown file to a styled PDF.

Reusable for the recurring report series. Usage:
    python3 reports/_render_pdf.py reports/<name>.md
Produces reports/<name>.pdf
"""

import sys
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

CSS = """
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
       font-size: 11px; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22px; color: #0b3d2e; border-bottom: 3px solid #0b3d2e;
     padding-bottom: 6px; margin-top: 18px; }
h2 { font-size: 16px; color: #0b3d2e; border-bottom: 1px solid #cdd6d1;
     padding-bottom: 3px; margin-top: 20px; }
h3 { font-size: 13px; color: #145c43; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10px; }
th { background: #0b3d2e; color: #fff; text-align: left; padding: 5px 8px; }
td { border: 1px solid #cdd6d1; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f3f6f4; }
code { background: #eef2f0; padding: 1px 4px; border-radius: 3px;
       font-family: 'SF Mono', Menlo, monospace; font-size: 10px; }
blockquote { border-left: 4px solid #2e8b6f; background: #f3f6f4;
             margin: 10px 0; padding: 8px 14px; color: #2a2a2a; }
a { color: #145c43; }
hr { border: none; border-top: 1px solid #cdd6d1; margin: 16px 0; }
strong { color: #0b1a14; }
ul, ol { margin: 6px 0 6px 0; }
li { margin: 2px 0; }
"""


def main() -> None:
    src = Path(sys.argv[1])
    md = src.read_text()
    out = src.with_suffix(".pdf")

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.add_section(Section(md, toc=True), user_css=CSS)
    pdf.meta["title"] = "RoboTrader — Project Status Report"
    pdf.meta["author"] = "RoboTrader project"
    pdf.meta["subject"] = "Comprehensive project status briefing"
    pdf.save(str(out))
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
