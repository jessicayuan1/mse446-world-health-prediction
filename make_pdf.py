"""
Render the report notebook to a clean, submittable PDF.

Two deliberate choices:
  * `exclude_input=True` -> the PDF shows only narrative text, tables and figures
    (no code cells). This matches the assignment's "20 pages excluding code" rule
    and removes the wide code blocks that were forcing narrow print margins.
  * Chromium (via Playwright) prints the HTML with explicit 1-inch Letter margins,
    so no LaTeX toolchain is required.

Usage:  python make_pdf.py
Output: MSE446_Project_Report.pdf
"""
from __future__ import annotations

import os

from nbconvert import HTMLExporter

NOTEBOOK = "MSE446_Project_Report.ipynb"
TMP_HTML = "_report_print.html"
OUT_PDF = "MSE446_Project_Report.pdf"

# Print-oriented CSS: full-width content, sensible page breaks, wrapped text.
PRINT_CSS = """
<style>
  body { max-width: 100% !important; margin: 0 !important; padding: 0 !important;
         font-size: 10pt; line-height: 1.3; }
  #notebook-container, .container { width: 100% !important;
         max-width: 100% !important; box-shadow: none !important;
         padding: 0 !important; }
  p { margin: 0.35em 0 !important; }
  pre, code { white-space: pre-wrap !important; word-wrap: break-word !important;
              font-size: 8.5pt; }
  /* Figures scaled to the printable width; allowed to flow to pack pages. */
  img { max-width: 82% !important; height: auto !important; display: block;
        margin: 0.3em auto; }
  table { page-break-inside: avoid; font-size: 8.5pt; margin: 0.3em 0; }
  h1 { font-size: 17pt; margin: 0.3em 0; }
  h2 { font-size: 14pt; margin: 0.5em 0 0.3em; page-break-after: avoid; }
  h3, h4 { font-size: 11.5pt; margin: 0.4em 0 0.2em; page-break-after: avoid; }
  .jp-OutputArea-output, .output_subarea { max-width: 100% !important; }
</style>
"""


def build_html() -> str:
    exporter = HTMLExporter()
    exporter.exclude_input = True          # drop code cells (keep their outputs)
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_filename(NOTEBOOK)
    # Inject print CSS just before </head>.
    if "</head>" in body:
        body = body.replace("</head>", PRINT_CSS + "</head>")
    else:
        body = PRINT_CSS + body
    with open(TMP_HTML, "w", encoding="utf-8") as f:
        f.write(body)
    return os.path.abspath(TMP_HTML)


def html_to_pdf(html_path: str):
    from playwright.sync_api import sync_playwright

    url = "file:///" + html_path.replace("\\", "/")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        page.pdf(
            path=OUT_PDF,
            format="Letter",
            print_background=True,
            margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
        )
        browser.close()


if __name__ == "__main__":
    html_path = build_html()
    html_to_pdf(html_path)
    os.remove(TMP_HTML)
    size_kb = os.path.getsize(OUT_PDF) / 1024
    print(f"wrote {OUT_PDF} ({size_kb:.0f} KB)")
