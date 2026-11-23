# Passage-Level Legal Meaning Preservation (LMP) Annotation Guidelines

**Status: Revised draft for pilot annotation**

## 1. Annotation unit

Each annotation concerns one aligned pair:

- **Original passage:** The original legal provision, section, subsection, or other legal text segment.
- **Simplified passage:** The corresponding simplified version of that same passage.

Rate how well the SIMPLIFIED PASSAGE preserves the legally relevant meaning of the ORIGINAL PASSAGE on a scale of 1–10.

Evaluate the complete passage as a whole, not individual sentences in isolation.

A passage may contain multiple sentences, clauses, definitions, conditions, exceptions, or subsections. Consider their combined legal meaning.

Do not assume that a passage is legally equivalent merely because most of its wording is similar.

## 2. Rating scale

| Score | Anchor |
|---|---|
| 9–10 | Duties, rights, conditions, exceptions, definitions, parties, dates, amounts, and scope are preserved throughout the passage. Any differences are immaterial to its legally relevant meaning. |
| 7–8 | Minor imprecision or limited loss of detail, with no clear change to a legally important meaning across the passage. |
| 4–6 | At least one potentially material legal detail is lost, added, weakened, strengthened, or changed. |
| 2–3 | Several serious changes, major omissions, or substantial alteration of the passage's legal effect. |
| 1 | Contradictory, unrelated, or fundamentally meaning-changing passage. |

Use the full scale where justified. Do not automatically assign high scores to passages that are long, fluent, or lexically similar.

## 3. What to examine

Read both passages completely. Check whether the simplified passage preserves:

1. **Obligations and permissions:** must, shall, may, should, is entitled to, is required to.
2. **Negation:** not, unless, except, prohibited, cannot.
3. **Conditions:** prerequisites, triggering events, and conditional requirements.
4. **Exceptions:** exclusions, provisos, exemptions, and carve-outs.
5. **Dates and deadlines:** effective dates, limitation periods, durations, and time limits.
6. **Amounts and numbers:** monetary amounts, percentages, quantities, thresholds, and section references.
7. **Parties and entities:** persons, authorities, officers, institutions, and defined terms.
8. **Scope:** all, any, only, some, each, specified classes, and territorial or subject-matter limits.
9. **Definitions and cross-references:** defined expressions and references to other provisions.
10. **Omissions and additions:** legally relevant content removed from or introduced into the simplified passage.

Record only changes that you actually observe. Do not select a category merely because an automated rule-based system flags it.

## 4. Passage-level scoring procedure

- Read the complete original and simplified passages before rating.
- Compare the legal meaning of the passages as a whole.
- Consider every material change, not just the first or most obvious one.
- A single serious change can justify a low score even if the remaining text is preserved.
- Several smaller changes may collectively alter the passage's legal effect.
- Do not average imagined sentence-level scores.
- Do not infer missing legal meaning from outside knowledge or unstated context.
- If the alignment is unclear or the passage is incomplete, explain the issue in annotation notes rather than inventing a judgment.
- Do not provide legal advice or decide whether a provision is legally valid.

## 5. Confidence and notes

Record:

- `annotator_rating`: integer from 1 to 10.
- `confidence`: integer from 1 to 5.
- `error_categories`: semicolon-separated labels, or `none`.
- `annotation_notes`: a short explanation of the observed passage-level preservation or change.

Confidence scale:

| Score | Meaning |
|---|---|
| 1 | Very low confidence |
| 2 | Low confidence |
| 3 | Moderate confidence |
| 4 | High confidence |
| 5 | Very high confidence |

Notes should identify the relevant part of the passage and explain why it affects meaning. Avoid unsupported legal conclusions.

## 6. Independent annotation and adjudication

- At least two independent annotators must rate each pair.
- Annotators must not discuss ratings with one another before submitting.
- Each annotator must submit their own rating and notes independently.
- Record annotator qualifications honestly (for example, "CS undergraduate, no legal training").
- If ratings differ by more than 2 points, the disagreement must be discussed and an `adjudicated_rating` recorded with resolution notes.
- Preserve the original independent ratings; never overwrite them with the adjudicated rating.
- Report ordinal Krippendorff's alpha for inter-annotator agreement.

AI-generated ratings must never be entered as human judgments.

## 7. Dataset provenance and training eligibility

Annotation does not establish source ownership, licensing, simplification origin, or training eligibility.

- Preserve source provenance and licence status.
- Do not treat unverified licences as verified.
- Do not treat unknown simplification origins as human-written or expert-reviewed.
- Synthetic pairs must retain `simplification_origin=synthetic`.
- Historical ratings on a different scale must not be converted into 1–10 labels.
- Only records passing the project's strict provenance, licence, origin, and adjudication validation may be used for supervised model training.

## 8. Error category vocabulary

Use these labels (semicolon-separated, or `none`) so annotations can be compared with the rule-based explanation module:

- obligation_permission_change
- negation_change
- date_or_deadline_change
- bound_or_direction_change
- monetary_amount_change
- percentage_change
- numerical_value_change
- frequency_change
- party_or_entity_change
- scope_change
- condition_removed
- condition_added
- condition_strength_change
- exception_removed
- exception_added
- possible_omission_or_addition

Human-only extras:

- contradiction
- hallucination
- other

Label what actually changed, not what an automated rule would flag.

## 9. Research reporting

Describe this annotation task as **passage-level legal meaning preservation**.

Report the number of passage pairs, independent annotators, adjudicated pairs, source documents, and eligible training examples separately.

Do not describe the task as sentence-level annotation unless a separately defined sentence-level dataset and protocol are introduced.
