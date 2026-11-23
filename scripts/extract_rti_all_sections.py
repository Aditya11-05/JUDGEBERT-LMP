import re
import csv
from pathlib import Path

source = Path("datasets/processed/official_laws/RTI_English.txt")
output = Path("datasets/processed/rti_all_section_candidates.csv")

text = source.read_text(encoding="utf-8")

# Start from the actual Act body, not the table of contents.
start = text.find("1. Short title, extent and commencement.—")
if start == -1:
    start = text.find("1. Short title, extent and commencement.")
if start == -1:
    raise RuntimeError("Could not locate the beginning of the Act.")

text = text[start:]

# Match section headings, not numbered amendment footnotes.
pattern = re.compile(
    r"(?m)^[ \t\f]*(3[01]|[12][0-9]|[1-9])\.\s+"
    r"([^—\n]+?)—"
)

matches = list(pattern.finditer(text))
rows = []

for i, match in enumerate(matches):
    section_no = int(match.group(1))
    title = match.group(2).strip()

    body_start = match.end()
    body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[body_start:body_end]

    # Remove obvious amendment footnotes and isolated page numbers.
    body = re.sub(
        r"(?m)^\s*\d+\.\s+(?:The words|Subs\.|Ins\.|Added|Omitted).*$",
        "",
        body
    )
    body = re.sub(r"(?m)^\s*\d+\s*$", "", body)
    body = re.sub(r"\s+", " ", body).strip()

    # Keep provision text intact; do not invent sentence boundaries.
    if len(body) < 20:
        continue

    rows.append({
        "candidate_id": f"RTI-{section_no:02d}",
        "source_document_id": "RTI_ACT_2005",
        "section_number": section_no,
        "section_title": title,
        "language": "en",
        "original_section_text": body,
        "simplified_text": "",
        "simplification_origin": "",
        "review_status": "needs_sentence_segmentation_and_review"
    })

fields = list(rows[0].keys()) if rows else []

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Sections extracted:", len(rows))
print("Section numbers:", [r["section_number"] for r in rows])
print("Output:", output)
