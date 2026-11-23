import re
import csv
from pathlib import Path

source = Path("datasets/processed/official_laws/RTI_English_Sections_1_11.txt")
output = Path("datasets/processed/rti_english_sentence_candidates.csv")

text = source.read_text(encoding="utf-8")

# Identify actual section headings (em dash distinguishes headings from footnotes).
pattern = re.compile(
    r"(?m)^\s*(1[01]|[1-9])\.\s+(.+?)—"
)

matches = list(pattern.finditer(text))
rows = []

for i, match in enumerate(matches):
    section_no = int(match.group(1))
    section_title = match.group(2).strip()

    start = match.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[start:end]

    # Remove amendment footnotes and normalize PDF line wrapping.
    body = re.sub(r"(?m)^\s*\d+\.\s+(?:The words|Subs\.|Ins\.).*$", "", body)
    body = re.sub(r"\s+", " ", body).strip()

    # Split into provisional sentence candidates.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", body)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue

        rows.append({
            "pair_id": f"RTI-EN-{len(rows)+1:04d}",
            "source_document_id": "RTI_ACT_2005",
            "section_number": section_no,
            "section_title": section_title,
            "language": "en",
            "original_text": sentence,
            "simplified_text": "",
            "simplification_origin": "",
            "source_url": "",
            "annotation_rating": "",
            "annotation_confidence": "",
            "review_status": "needs_sentence_review"
        })

fields = list(rows[0].keys()) if rows else []

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Sections found: {len(matches)}")
print(f"Sentence candidates: {len(rows)}")
print(f"Saved to: {output}")
