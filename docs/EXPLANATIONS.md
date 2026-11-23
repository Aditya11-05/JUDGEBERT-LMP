# Legal-error explanation module (rule-based)

Status: **English implemented and unit-tested. Hindi is NOT supported** (no rules, no tests; requesting `hi` raises
`UnsupportedLanguageError`). The module is independent of the LMP regression model: it never imports it, and its output
must never be used to set or adjust model predictions.

## What it does
`explain(original, simplified, language="en", min_strength="weak")` returns an `ExplanationResult`:
`findings`, both sentences with `[[highlighted]]` spans, a summary, and a disclaimer. Each `Finding` has
`error_type`, `change_kind` (changed / omitted / added), `original_span`, `simplified_span` (text + character offsets),
`explanation`, `evidence_strength`, `rule_id`. `to_dict()` is JSON-ready; `highlight_html()` gives escaped `<mark>` HTML.

CLI: `python scripts/explain_pair.py "original" "simplified" [--json] [--min-strength moderate]`

## Detected categories (`error_type`)
| Category | What is compared | Context-aware handling |
|---|---|---|
| obligation_permission_change | obligation / permission / prohibition / no-obligation / advice (must, shall, required to, may, can, entitled to, may not, must not, prohibited from, need not ...) | `May` the month is not a modal; `no party may` = prohibition (its `no` is not counted again as a negation); `must` ~ `shall` ~ `required to` are equivalent |
| negation_change | parity of negation cues (not, no, never, without, n't, fail(ure) to, ineligible ...) | `no later than`, `not more than`, `not limited to`, `No. 5` are ignored; modal negations are not double counted |
| date_or_deadline_change | explicit dates, durations, ordinal days ("the fifth day") | `15 March 2026` = `15/03/2026` = `2026-03-15`; missing year tolerated; `thirty (30) days`; 2 weeks = 14 days; 1 year = 12 months; business days are distinct from days |
| bound_or_direction_change | at least / at most / more than / within / after / before attached to a value; before vs after between events | `within` ~ `no later than` ~ `up to` for time values |
| monetary_amount_change | amount + currency (₹, Rs, INR, $, rupees, lakh, crore ...) | `Rs. 10,000` = `10,000 rupees` = `ten thousand rupees`; `words (figures)` counted once |
| percentage_change, numerical_value_change, frequency_change | percentages, plain numbers, daily/monthly/... | `ten per cent` = `10%`; `each month` = `monthly` = `per month` |
| party_or_entity_change | role words (tenant/lessee/renter ...), named entities | synonym groups; passive voice suppresses the order heuristic; `RBI` = `Reserve Bank of India`; sentence-initial capitals ignored; ALL-CAPS text skips entities |
| scope_change | all/every/each/in full vs some/certain/in part vs only/solely | `each month` is a frequency not a scope word; `each other`, `at all`, `only if` excluded |
| condition_removed / added / strength_change | if, only if, provided that, subject to, in the event, as long as | `if` ~ `provided that` ~ `in case` ~ `when/once/upon/failure to` (weak cues stop a "removed" flag); `even if`, `as if` excluded; `only if` vs `if` is flagged |
| exception_removed / added | unless, except, other than, save for, apart from, does not apply ... | `unless X` ~ `if not X` is not flagged |
| possible_omission_or_addition | content-word count ratio only | weak evidence by design |

## Evidence strength
`strong` / `moderate` / `weak` are **categorical heuristics, not probabilities**. Roughly: strong = a clear counterpart
changed (e.g. `must`->`may`, a different amount); moderate = something has no counterpart (dropped amount/condition,
unit mismatch); weak = may well be a harmless paraphrase (dropped modal, party-order swap, length ratio). No numeric
confidence is reported because none has been calibrated against human labels yet.

## Known limitations (several are pinned by tests that assert the current, wrong behaviour)
* **Lexical, not semantic.** Antonyms (`in advance` / `in arrears`), prefix negations (`limited` / `unlimited`),
  hypernyms (`dog` / `pet`), changed trigger events (`of delivery` / `of invoice`) and a dropped prepositional condition
  (`with notice`) are **not detected**.
* **Positive rewrites of prohibitions** (`shall not disclose` -> `must keep secret`) are flagged although meaning may be preserved.
* **Pronouns** replacing parties (`the tenant` -> `you`) cannot be resolved; reported as weak.
* Modal order is ignored when both sentences contain the same set of modals (`must pay ... may sublet` swapped).
* `shall` is sometimes only future tense in legal drafting; `may not` can be ambiguous; `can` may denote ability.
* Negation scope is approximated by parity and modal-prefix handling, not a parse. `no need to` is not modelled.
* Implicit conditions/exceptions (no cue word) are invisible; `where`/`when` are only equivalence cues.
* Money/dates: US-style `MM/DD/YYYY` is only recognised when the day exceeds 12; bare months (`in March`) are not compared.
* Entities: capitalisation-based; a named party at the very start of a sentence is skipped unless multi-word or an acronym.
* Rules were tuned on the author's own examples and a 14-pair probe; **no precision/recall on human-annotated data exists
  yet.** `app/explanations/dataset.py::category_prf` will compute it once annotated pairs exist; until then no accuracy claim is made.
* Rule output is not a faithful explanation of what the transformer computes, and is not a legal conclusion.

## Dataset integration
* Human labels live in `error_categories` (`;`-separated). Vocabulary = `ERROR_TYPES` + `contradiction`, `hallucination`, `other`, `none`.
* `annotate_dataframe(df)` adds `rule_supported`, `rule_error_types` (same `;` format), `rule_n_findings`, `rule_findings` (JSON).
  Unsupported-language rows are marked unsupported, never guessed.
* `category_prf(df)` scores rules against human labels (explanation accuracy), separate from LMP regression metrics.
* `unknown_error_categories(df)` (also exposed from `app.data.validate_dataset`) reports labels outside the vocabulary.

## Adding Hindi (planned, not started)
Language-neutral parts (`types.py`, `alignment.py`, `rules.py`) are reused. A Hindi rule set needs: `lexicon_hi.py::extract()`
(modals such as चाहिए / होगा / सकता है / नहीं / ज़रूरी, Devanagari numerals and numeral words, rupee and date forms,
role words, if/unless cues), an `entity_eq`, registration in `engine.RULESETS`, and its own tests including harmless
paraphrases. `hi` must stay unregistered until those tests pass. The tokenizer already accepts the Devanagari block,
but matras are not yet verified for the regex word boundaries.
