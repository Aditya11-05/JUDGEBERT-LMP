"""Language-neutral comparison rules.

Each rule compares the Features of the original and the simplified sentence and returns Findings.
Rules never look at the LMP regression model and must never be used to set its predictions.
evidence_strength is categorical ("strong" > "moderate" > "weak"); it is NOT a calibrated probability.
"""
from __future__ import annotations
from .alignment import pair_differences
from .types import Features, Finding

_MODAL_LABEL = {"obligation": "an obligation", "permission": "a permission or right", "prohibition": "a prohibition",
                "no_obligation": "an absence of obligation", "advice": "a recommendation"}
_HARD_MODALS = {"obligation", "permission", "prohibition", "no_obligation"}


def _sp(m):
    return m.span if m is not None else None


def _mk(etype, kind, o, s, text, strength, rule_id):
    return Finding(etype, kind, _sp(o), _sp(s), text, strength, rule_id)


def _q(m) -> str:
    return f"'{m.span.text}'"


# ------------------------------------------------------------------ modality
def cmp_modality(fo: Features, fs: Features, rs) -> list[Finding]:
    _, changed, only_a, only_b = pair_differences(fo.get("modality"), fs.get("modality"))
    out = []
    for x, y in changed:
        strength = "strong" if {x.key, y.key} <= _HARD_MODALS else "moderate"
        out.append(_mk("obligation_permission_change", "changed", x, y,
                       f"Potential meaning change: {_q(x)} expresses {_MODAL_LABEL[x.key]}, but {_q(y)} expresses "
                       f"{_MODAL_LABEL[y.key]}. This can change what a party must, may, or must not do.",
                       strength, "modality.changed"))
    for x in only_a:
        if x.key != "advice":
            out.append(_mk("obligation_permission_change", "omitted", x, None,
                           f"{_q(x)} ({_MODAL_LABEL[x.key]}) has no matching modal expression in the simplified sentence. "
                           "The simplified text may no longer state this duty, permission, or prohibition "
                           "(it may also be expressed another way, e.g. as an imperative).", "weak", "modality.omitted"))
    for y in only_b:
        if y.key != "advice":
            out.append(_mk("obligation_permission_change", "added", None, y,
                           f"The simplified sentence adds {_q(y)} ({_MODAL_LABEL[y.key]}) that the original does not express.",
                           "weak", "modality.added"))
    return out


# ------------------------------------------------------------------ negation
def cmp_negation(fo: Features, fs: Features, rs) -> list[Finding]:
    a, b = fo.get("negation"), fs.get("negation")
    if len(a) % 2 != len(b) % 2:
        if len(a) > len(b):
            x = a[len(b)]
            return [_mk("negation_change", "omitted", x, None,
                        f"Potential meaning change: the negation {_q(x)} appears in the original but not in the simplified "
                        "sentence. Dropping a negation can reverse the meaning (e.g. 'is not liable' vs 'is liable').",
                        "strong", "negation.parity")]
        y = b[len(a)]
        return [_mk("negation_change", "added", None, y,
                    f"Potential meaning change: the negation {_q(y)} appears in the simplified sentence but not in the "
                    "original. Adding a negation can reverse the meaning.", "strong", "negation.parity")]
    if len(a) != len(b):
        x, y = (a[len(b)], None) if len(a) > len(b) else (None, b[len(a)])
        return [_mk("negation_change", "omitted" if x else "added", x, y,
                    f"The number of negation words differs ({len(a)} in the original, {len(b)} in the simplified "
                    "sentence). Check that the scope of each negation is unchanged.", "weak", "negation.count")]
    return []


# ------------------------------------------------------------------ quantities
_TEMPORAL = {"duration", "date", "day_of_month"}
_DIRECTIONAL = {"at_least", "at_most", "more_than", "less_than"}
_QUANT = [  # kind, error_type, noun, strength when omitted/added
    ("money", "monetary_amount_change", "monetary amount", "moderate"),
    ("percent", "percentage_change", "percentage", "moderate"),
    ("duration", "date_or_deadline_change", "time period", "moderate"),
    ("date", "date_or_deadline_change", "date", "moderate"),
    ("day_of_month", "date_or_deadline_change", "day of the month", "moderate"),
    ("frequency", "frequency_change", "frequency", "weak"),
    ("number", "numerical_value_change", "number", "weak"),
]


def date_eq(x, y) -> bool:
    (y1, m1, d1), (y2, m2, d2) = x.key, y.key
    return m1 == m2 and d1 == d2 and (y1 is None or y2 is None or y1 == y2)


def _norm_q(kind, q):
    return "deadline_upper" if kind in _TEMPORAL and q == "at_most" else q


def cmp_quantities(fo: Features, fs: Features, rs) -> list[Finding]:
    out = []
    for kind, etype, noun, loose in _QUANT:
        a, b = fo.get(kind), fs.get(kind)
        matched, changed, only_a, only_b = pair_differences(a, b, eq=date_eq if kind == "date" else None)
        for x, y in changed:
            strength = "strong"
            note = ""
            if kind == "duration" and x.key[0] != y.key[0]:
                strength = "moderate"
                note = " The units differ (e.g. days vs months or business days), so check whether they are equivalent."
            if kind == "money" and x.key[0] != y.key[0]:
                note = " The currency also differs."
            out.append(_mk(etype, "changed", x, y,
                           f"Potential meaning change: the {noun} {_q(x)} in the original became {_q(y)} in the simplified "
                           f"sentence.{note}", strength, f"{kind}.changed"))
        for x in only_a:
            out.append(_mk(etype, "omitted", x, None,
                           f"The {noun} {_q(x)} in the original has no counterpart in the simplified sentence.",
                           loose, f"{kind}.omitted"))
        for y in only_b:
            out.append(_mk(etype, "added", None, y,
                           f"The simplified sentence introduces a {noun} {_q(y)} that is not in the original.",
                           loose, f"{kind}.added"))
        if kind in ("frequency",):
            continue
        for x, y in matched:  # same value: did its qualifier (at least / by / within ...) survive?
            qx, qy = _norm_q(kind, x.attrs.get("qualifier")), _norm_q(kind, y.attrs.get("qualifier"))
            if qx == qy:
                continue
            if qx and qy:
                out.append(_mk("bound_or_direction_change", "changed", x, y,
                               f"Potential meaning change: the same {noun} is bounded differently "
                               f"({qx.replace('_', ' ')} vs {qy.replace('_', ' ')}), e.g. a minimum became a maximum or a "
                               "deadline became a start date.", "strong", f"{kind}.bound"))
            else:
                kept, which = (qx, "omitted") if qx else (qy, "added")
                strength = "moderate" if kept in _DIRECTIONAL else "weak"
                out.append(_mk("bound_or_direction_change", which, x if qx else None, None if qx else y,
                               f"The qualifier '{kept.replace('_', ' ')}' for the {noun} was {which} in the simplified "
                               "sentence, which can change whether the value is a limit, a minimum, or a deadline.",
                               strength, f"{kind}.bound_{which}"))
    return out


# ------------------------------------------------------------------ parties / entities
def _ordered_unique(ms):
    seen, out = set(), []
    for m in ms:
        if m.key not in seen:
            seen.add(m.key)
            out.append(m)
    return out


def cmp_parties(fo: Features, fs: Features, rs) -> list[Finding]:
    a, b = fo.get("party"), fs.get("party")
    _, changed, only_a, only_b = pair_differences(a, b)
    out = []
    for x, y in changed:
        out.append(_mk("party_or_entity_change", "changed", x, y,
                       f"Potential meaning change: the party {_q(x)} in the original is {_q(y)} in the simplified sentence. "
                       "Swapping who owes or receives something changes the legal effect.", "strong", "party.changed"))
    for x in only_a:
        out.append(_mk("party_or_entity_change", "omitted", x, None,
                       f"The party {_q(x)} is not named in the simplified sentence (it may be replaced by a pronoun such as "
                       "'you', which this rule cannot resolve).", "moderate" if b else "weak", "party.omitted"))
    for y in only_b:
        out.append(_mk("party_or_entity_change", "added", None, y,
                       f"The simplified sentence names a party {_q(y)} that the original does not.",
                       "moderate" if a else "weak", "party.added"))
    if not (changed or only_a or only_b) and len({m.key for m in a}) >= 2 and not (fo.passive or fs.passive):
        oa, ob = _ordered_unique(a), _ordered_unique(b)
        if [m.key for m in oa] != [m.key for m in ob]:
            out.append(_mk("party_or_entity_change", "changed", oa[0], ob[0],
                           "The order in which parties are mentioned differs. In active sentences this can mean the actor "
                           "and the recipient were swapped (heuristic; harmless re-ordering is possible).",
                           "weak", "party.order"))
    return out


def cmp_entities(fo: Features, fs: Features, rs) -> list[Finding]:
    _, changed, only_a, only_b = pair_differences(fo.get("entity"), fs.get("entity"), eq=rs.entity_eq)
    out = []
    for x, y in changed:
        out.append(_mk("party_or_entity_change", "changed", x, y,
                       f"Potential meaning change: the named entity {_q(x)} became {_q(y)}.", "strong", "entity.changed"))
    for x in only_a:
        out.append(_mk("party_or_entity_change", "omitted", x, None,
                       f"The named entity {_q(x)} is missing from the simplified sentence (it may be abbreviated or "
                       "described differently).", "moderate", "entity.omitted"))
    for y in only_b:
        out.append(_mk("party_or_entity_change", "added", None, y,
                       f"The simplified sentence mentions {_q(y)}, which is not in the original (possible added detail).",
                       "moderate", "entity.added"))
    return out


# ------------------------------------------------------------------ scope
def cmp_scope(fo: Features, fs: Features, rs) -> list[Finding]:
    _, changed, only_a, only_b = pair_differences(fo.get("scope"), fs.get("scope"))
    out = []
    for x, y in changed:
        strength = "strong" if {x.key, y.key} == {"universal", "partial"} else "moderate"
        out.append(_mk("scope_change", "changed", x, y,
                       f"Potential meaning change: the scope word {_q(x)} ({x.key}) became {_q(y)} ({y.key}), which can "
                       "widen or narrow who or what is covered.", strength, "scope.changed"))
    for x in only_a:
        out.append(_mk("scope_change", "omitted", x, None,
                       f"The scope word {_q(x)} ({x.key}) is not carried over to the simplified sentence.",
                       "moderate" if x.key == "restrictive" else "weak", "scope.omitted"))
    for y in only_b:
        out.append(_mk("scope_change", "added", None, y,
                       f"The simplified sentence adds the scope word {_q(y)} ({y.key}).",
                       "moderate" if y.key == "restrictive" else "weak", "scope.added"))
    return out


# ------------------------------------------------------------------ conditions / exceptions
def cmp_conditions(fo: Features, fs: Features, rs) -> list[Finding]:
    so, ss = fo.get("condition"), fs.get("condition")
    wo, ws = fo.get("condition_weak"), fs.get("condition_weak")
    unless_o = any(m.attrs.get("unless") for m in fo.get("exception"))
    unless_s = any(m.attrs.get("unless") for m in fs.get("exception"))
    out = []
    if so and not ss and not ws and not unless_s:
        out.append(_mk("condition_removed", "omitted", so[0], None,
                       f"Potential meaning change: the condition {_q(so[0])} has no counterpart in the simplified sentence, "
                       "so the duty or right may now apply unconditionally.", "moderate", "condition.removed"))
    elif len(so) > len(ss) > 0:
        out.append(_mk("condition_removed", "omitted", so[len(ss)], None,
                       "The original has more conditional clauses than the simplified sentence; one may have been dropped "
                       "or merged.", "weak", "condition.fewer"))
    if not so and ss and not wo and not unless_o:
        out.append(_mk("condition_added", "added", None, ss[0],
                       f"The simplified sentence adds a condition {_q(ss[0])} that the original does not contain.",
                       "moderate", "condition.added"))
    elif len(ss) > len(so) > 0:
        out.append(_mk("condition_added", "added", None, ss[len(so)],
                       "The simplified sentence has more conditional clauses than the original.", "weak", "condition.more"))
    if so and ss and any(m.attrs["necessary"] for m in so) != any(m.attrs["necessary"] for m in ss):
        nec_o = any(m.attrs["necessary"] for m in so)
        x = next(m for m in so if m.attrs["necessary"] == nec_o)
        y = next(m for m in ss if m.attrs["necessary"] != nec_o)
        out.append(_mk("condition_strength_change", "changed", x, y,
                       "'only if' (a necessary condition) and plain 'if' do not mean the same thing; the strength of the "
                       "condition may have changed.", "moderate", "condition.strength"))
    return out


def cmp_exceptions(fo: Features, fs: Features, rs) -> list[Finding]:
    eo, es = fo.get("exception"), fs.get("exception")
    unless_o = any(m.attrs.get("unless") for m in eo)
    unless_s = any(m.attrs.get("unless") for m in es)
    neg_cond_s = bool(fs.get("condition") and fs.get("negation"))  # "unless X" -> "if not X"
    neg_cond_o = bool(fo.get("condition") and fo.get("negation"))
    out = []
    if eo and not es and not (unless_o and neg_cond_s):
        out.append(_mk("exception_removed", "omitted", eo[0], None,
                       f"Potential meaning change: the exception {_q(eo[0])} has no counterpart in the simplified sentence, "
                       "so the rule may now apply in cases the original excluded.", "strong", "exception.removed"))
    elif len(eo) > len(es) > 0:
        out.append(_mk("exception_removed", "omitted", eo[len(es)], None,
                       "The original has more exceptions than the simplified sentence.", "weak", "exception.fewer"))
    if es and not eo and not (unless_s and neg_cond_o):
        out.append(_mk("exception_added", "added", None, es[0],
                       f"The simplified sentence adds an exception {_q(es[0])} that the original does not contain.",
                       "moderate", "exception.added"))
    elif len(es) > len(eo) > 0:
        out.append(_mk("exception_added", "added", None, es[len(eo)],
                       "The simplified sentence has more exceptions than the original.", "weak", "exception.more"))
    return out


def cmp_sequence(fo: Features, fs: Features, rs) -> list[Finding]:
    """'before X' vs 'after X' when no number or date carries the qualifier (those go through cmp_quantities)."""
    a, b = fo.get("sequence"), fs.get("sequence")
    ko, ks = {m.key for m in a}, {m.key for m in b}
    if ko and ks and not ko & ks:
        return [_mk("bound_or_direction_change", "changed", a[0], b[0],
                    f"Potential meaning change: the ordering word {_q(a[0])} became {_q(b[0])}, so the duty or right may "
                    "now apply at a different point in time.", "strong", "sequence.changed")]
    return []


def cmp_length(fo: Features, fs: Features, rs) -> list[Finding]:
    co, cs = fo.content_tokens, fs.content_tokens
    if co >= 8 and cs / co < 0.5:
        return [_mk("possible_omission_or_addition", "omitted", None, None,
                    f"The simplified sentence is much shorter ({cs} vs {co} content words), so material detail may have "
                    "been omitted. Length alone is weak evidence.", "weak", "length.shorter")]
    if co >= 4 and cs / co > 2.0:
        return [_mk("possible_omission_or_addition", "added", None, None,
                    f"The simplified sentence is much longer ({cs} vs {co} content words), so details may have been added. "
                    "Explanatory additions are common in simplification; check them against the source.",
                    "weak", "length.longer")]
    return []


ALL_RULES = (cmp_modality, cmp_negation, cmp_quantities, cmp_parties, cmp_entities, cmp_scope,
             cmp_conditions, cmp_exceptions, cmp_sequence, cmp_length)
