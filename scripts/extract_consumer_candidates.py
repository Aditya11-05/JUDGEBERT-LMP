import re
import csv
from pathlib import Path

source = Path("datasets/processed/official_laws/Consumer_Protection_English.txt")
output = Path("datasets/processed/consumer_protection_sentence_candidates.csv")

text = source.read_text(encoding="utf-8")

start = text.find("1. Short title, extent, commencement and application.—")
if start == -1:
    raise RuntimeError("Could not locate the start of the Act.")

text = text[start:]

pattern = re.compile(
    r"(?m)^[ \t\f]*(\d{1,3})\.\s+([^—\n]+?)—"
)

matches = list(pattern.finditer(text))
rows = []

for i, match in enumerate(matches):
    section_no = int(match.group(1))
    title = match.group(2).strip()

    body_start = match.end()
    body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[body_start:body_end]

    body = re.sub(
        r"(?m)^\s*\d+\.\s+(?:The words|Subs\.|Ins\.|Added|Omitted).*$",
        "",
        body
    )
    body = re.sub(r"(?m)^\s*\d+\s*$", "", body)
    body = re.sub(r"\s+", " ", body).strip()

    # Split into provisional sentence/clause candidates.
    parts = re.split(r"(?<=[.;])\s+", body)

    for part in parts:
        part = part.strip()
        if len(part) < 25:
            continue

        rows.append({
            "pair_id": f"CPA-EN-{len(rows)+1:04d}",
            "source_document_id": "CONSUMER_PROTECTION_ACT_2019",
            "section_number": section_no,
            "section_title": title,
            "language": "en",
            "original_text": part,
            "simplified_text": "",
            "simplification_origin": "",
            "annotation_rating": "",
            "annotation_confidence": "",
            "review_status": "needs_manual_review"
        })

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Sections detected:", len(matches))
print("Sentence/clause candidates:", len(rows))
print("Output:", output)
print("Section numbers:", sorted(set(r["section_number"] for r in rows)))
