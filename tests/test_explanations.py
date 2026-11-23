"""Unit tests for the rule-based legal-error explanation module (English only)."""
import json
import pytest
from app.explanations import (ERROR_TYPES, SUPPORTED_LANGUAGES, STRENGTHS, UnsupportedLanguageError, explain,
                              find_changes, highlight_html)
from app.explanations.alignment import mark, pair_differences, to_html
from app.explanations.lexicon_en import extract, qty_value


def types(o, s, min_strength="weak"):
    return {f.error_type for f in find_changes(o, s, "en", min_strength)}


def rule_ids(o, s):
    return {f.rule_id for f in find_changes(o, s, "en")}


# ------------------------------------------------------------ modality
def test_deck_example_must_to_may_and_deadline():
    r = explain("The tenant must pay the rent by the fifth day of each month.",
                "The tenant may pay the rent by the tenth day of each month.")
    assert {f.error_type for f in r.findings} == {"obligation_permission_change", "date_or_deadline_change"}
    modal = next(f for f in r.findings if f.error_type == "obligation_permission_change")
    assert (modal.original_span.text, modal.simplified_span.text) == ("must", "may")
    assert modal.evidence_strength == "strong"
    assert "[[must]]" in r.highlighted_original and "[[may]]" in r.highlighted_simplified
    assert "[[fifth day]]" in r.highlighted_original and "[[tenth day]]" in r.highlighted_simplified


@pytest.mark.parametrize("o,s", [
    ("The tenant shall pay rent.", "The tenant may pay rent."),
    ("The tenant may sublet the flat.", "The tenant must sublet the flat."),
    ("The tenant must enter the flat.", "The tenant must not enter the flat."),
    ("The tenant may sublet the flat.", "The tenant may not sublet the flat."),
    ("The tenant must pay rent.", "The tenant is not required to pay rent."),
    ("The tenant is required to pay rent.", "The tenant is allowed to pay rent."),
    ("The tenant must pay rent.", "The tenant should pay rent."),
])
def test_modal_changes_flagged(o, s):
    assert "obligation_permission_change" in types(o, s)


@pytest.mark.parametrize("o,s", [
    ("The seller shall deliver the goods.", "The seller must deliver the goods."),
    ("The seller shall deliver the goods.", "The seller is required to deliver the goods."),
    ("The seller must deliver the goods.", "The seller has to deliver the goods."),
    ("The employee may not disclose secrets.", "The employee is prohibited from disclosing secrets."),
    ("The employee cannot disclose secrets.", "The employee must not disclose secrets."),
    ("The tenant may terminate the lease.", "The tenant can terminate the lease."),
    ("The tenant may terminate the lease.", "The tenant is entitled to terminate the lease."),
    ("No party may assign this agreement.", "A party may not assign this agreement."),
    ("Nobody shall enter the premises.", "Nobody must enter the premises."),
])
def test_modal_harmless_paraphrases_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


def test_negator_before_modal_is_prohibition_not_permission():
    f = extract("No tenant may sublet the flat.")
    assert [m.key for m in f.get("modality")] == ["prohibition"]
    assert f.get("negation") == []  # the 'no' is already accounted for by the prohibition


def test_may_month_is_not_a_modal_verb():
    for s in ("Rent is due in May.", "Rent is due on May 5.", "The lease ends on 5 May 2027."):
        assert extract(s).get("modality") == []
    assert [m.key for m in extract("The tenant may pay in May.").get("modality")] == ["permission"]


def test_uppercase_and_curly_apostrophe_modals():
    assert [m.key for m in extract("THE TENANT MAY NOT SUBLET.").get("modality")] == ["prohibition"]
    assert [m.key for m in extract("The tenant mustn\u2019t sublet.").get("modality")] == ["prohibition"]


def test_modal_omitted_is_weak_only():
    fs = find_changes("The tenant must pay rent.", "The tenant pays rent.", "en")
    assert [f.evidence_strength for f in fs if f.error_type == "obligation_permission_change"] == ["weak"]


# ------------------------------------------------------------ negation
@pytest.mark.parametrize("o,s", [
    ("The company is liable for the damage.", "The company is not liable for the damage."),
    ("The company is not liable for the damage.", "The company is liable for the damage."),
    ("The tenant has paid.", "The tenant has never paid."),
    ("Payment is made with notice.", "Payment is made without notice."),
])
def test_negation_changes(o, s):
    assert "negation_change" in types(o, s)


def test_negation_span_and_strength():
    f = next(f for f in find_changes("The firm is liable.", "The firm is not liable.", "en") if f.error_type == "negation_change")
    assert f.change_kind == "added" and f.simplified_span.text == "not" and f.evidence_strength == "strong"


@pytest.mark.parametrize("o,s", [
    ("Pay rent no later than the fifth day.", "Pay rent by the fifth day."),
    ("The fee is not more than 500 rupees.", "The fee is at most 500 rupees."),
    ("The tenant must pay, including but not limited to rent.", "The tenant must pay, including rent."),
    ("Without notice the contract ends.", "The contract ends with no notice."),
])
def test_negation_idioms_not_flagged(o, s):
    assert "negation_change" not in types(o, s)


def test_double_negation_parity_preserved():
    assert "negation_change" not in types("The tenant is not unable to pay.", "The tenant is able to pay.", "moderate")


# ------------------------------------------------------------ dates and periods
@pytest.mark.parametrize("o,s", [
    ("Pay by 15 March 2026.", "Pay by 16 March 2026."),
    ("Pay by 15 March 2026.", "Pay by 15 April 2026."),
    ("Pay by 15 March 2026.", "Pay by 15 March 2027."),
    ("The notice period is 30 days.", "The notice period is 60 days."),
    ("The notice period is thirty days.", "The notice period is 45 days."),
    ("The notice period is 30 days.", "The notice period is 30 business days."),
    ("The notice period is 30 days.", "The notice period is one month."),
    ("Pay by the fifth day of each month.", "Pay by the tenth day of each month."),
    ("Rent is due monthly.", "Rent is due annually."),
])
def test_date_period_changes_flagged(o, s):
    assert types(o, s) & {"date_or_deadline_change", "frequency_change"}


@pytest.mark.parametrize("o,s", [
    ("Pay by 15 March 2026.", "Pay by 15/03/2026."),
    ("Pay by 15 March 2026.", "Pay by 2026-03-15."),
    ("Pay by 15 March.", "Pay by 15 March 2026."),
    ("Pay by March 15, 2026.", "Pay by 15 March 2026."),
    ("The notice period is thirty (30) days.", "The notice period is 30 days."),
    ("The notice period is two weeks.", "The notice period is 14 days."),
    ("The licence is valid for one year.", "The licence is valid for 12 months."),
    ("Pay by the fifth day of each month.", "Pay by the 5th of every month."),
    ("Rent is due each month.", "Rent is due monthly."),
    ("Rent is due every year.", "Rent is due annually."),
])
def test_date_period_equivalent_forms_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


def test_unit_mismatch_is_moderate_not_strong():
    f = next(f for f in find_changes("Notice is 30 days.", "Notice is one month.", "en") if f.rule_id == "duration.changed")
    assert f.evidence_strength == "moderate" and "units differ" in f.explanation


def test_date_added_and_omitted():
    assert any(f.change_kind == "omitted" for f in find_changes("Pay by 15 March 2026 in full.", "Pay in full.", "en"))
    assert any(f.change_kind == "added" for f in find_changes("Pay in full.", "Pay by 15 March 2026 in full.", "en"))


# ------------------------------------------------------------ bounds / direction
@pytest.mark.parametrize("o,s,expected", [
    ("Give at least 30 days notice.", "Give at most 30 days notice.", "bound_or_direction_change"),
    ("Pay within 30 days.", "Pay after 30 days.", "bound_or_direction_change"),
    ("Pay before 15 March 2026.", "Pay after 15 March 2026.", "bound_or_direction_change"),
    ("The fine is at least 500 rupees.", "The fine is 500 rupees.", "bound_or_direction_change"),
])
def test_bound_changes(o, s, expected):
    assert expected in types(o, s)


@pytest.mark.parametrize("o,s", [
    ("Pay within 30 days.", "Pay no later than 30 days."),
    ("Pay within 30 days.", "Pay up to 30 days."),
    ("Pay by 15 March 2026.", "Pay before 15 March 2026."),
    ("The fine is at least 500 rupees.", "The fine is no less than 500 rupees."),
])
def test_bound_equivalents_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


# ------------------------------------------------------------ money / percent / numbers
@pytest.mark.parametrize("o,s", [
    ("Rent is \u20b910,000.", "Rent is \u20b91,000."),
    ("Rent is Rs. 10,000.", "Rent is Rs. 100,000."),
    ("The fee is $100.", "The fee is \u20b9100."),
    ("The fee is ten thousand rupees.", "The fee is Rs. 1,000."),
])
def test_money_changes(o, s):
    assert "monetary_amount_change" in types(o, s)


@pytest.mark.parametrize("o,s", [
    ("Rent is Rs. 10,000.", "Rent is 10,000 rupees."),
    ("Rent is \u20b910,000.", "Rent is INR 10000."),
    ("The fee is ten thousand rupees.", "The fee is \u20b910,000."),
    ("The fine is Rs 2 lakh.", "The fine is Rs. 200000."),
    ("The fine is ten thousand rupees (Rs. 10,000).", "The fine is Rs. 10,000."),
    ("Rent is Rs. 10,000 per month.", "Rent is \u20b910,000 monthly."),
])
def test_money_equivalent_forms_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


def test_money_omitted_and_added_are_moderate():
    assert [f.evidence_strength for f in find_changes("Pay Rs. 500 now.", "Pay now.", "en")] == ["moderate"]
    assert [f.change_kind for f in find_changes("Pay now.", "Pay Rs. 500 now.", "en")] == ["added"]


@pytest.mark.parametrize("o,s,flag", [
    ("Interest is 10%.", "Interest is 12%.", True),
    ("Interest is ten per cent.", "Interest is 10%.", False),
    ("Interest is 10 percent.", "Interest is 10%.", False),
    ("Interest is 2.5%.", "Interest is 25%.", True),
])
def test_percent(o, s, flag):
    assert ("percentage_change" in types(o, s)) is flag


@pytest.mark.parametrize("o,s,flag", [
    ("Both parties sign three copies.", "Both parties sign four copies.", True),
    ("See Section 138.", "See Section 139.", True),
    ("Both parties sign three copies.", "Both parties sign 3 copies.", False),
])
def test_plain_numbers(o, s, flag):
    assert ("numerical_value_change" in types(o, s)) is flag


def test_qty_value_parsing():
    assert qty_value("one hundred and fifty") == 150
    assert qty_value("twenty-five") == 25
    assert qty_value("two lakh fifty thousand") == 250000
    assert qty_value("10,00,000") == 1_000_000
    assert qty_value("3 crore") == 3e7


# ------------------------------------------------------------ parties and entities
@pytest.mark.parametrize("o,s", [
    ("The tenant shall repair the roof.", "The landlord shall repair the roof."),
    ("The buyer shall inspect the goods.", "The seller shall inspect the goods."),
    ("The employer shall give notice to the employee.", "The employee shall give notice to the employer."),
])
def test_party_changes(o, s):
    assert "party_or_entity_change" in types(o, s)


@pytest.mark.parametrize("o,s", [
    ("The lessee shall repair the roof.", "The tenant shall repair the roof."),
    ("The lessor shall give notice to the lessee.", "The landlord shall give notice to the tenant."),
    ("The policyholder shall report the loss.", "The insured shall report the loss."),
    ("The landlord shall notify the tenant.", "The tenant shall be notified by the landlord."),
])
def test_party_synonyms_and_passive_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


def test_party_order_swap_is_weak_heuristic():
    fs = find_changes("The landlord shall pay the tenant.", "The tenant shall pay the landlord.", "en")
    assert [(f.rule_id, f.evidence_strength) for f in fs] == [("party.order", "weak")]


def test_pronoun_replacing_party_is_weak():
    fs = find_changes("The tenant shall pay rent.", "You must pay rent.", "en")
    party = [f for f in fs if f.error_type == "party_or_entity_change"]
    assert party and party[0].evidence_strength == "weak"


def test_named_entities():
    assert "party_or_entity_change" in types("Payment goes to Ramesh Kumar.", "Payment goes to Suresh Kumar.")
    assert "party_or_entity_change" in types("Appeal lies to the Delhi High Court.", "Appeal lies to the court.", "moderate")
    assert find_changes("The Reserve Bank of India issues the rules.", "The RBI issues the rules.", "en") == []
    assert find_changes("Rent is due.", "The rent is due.", "en") == []  # sentence-initial capital is not a name


def test_all_caps_text_does_not_create_entities():
    assert extract("THE TENANT SHALL PAY RENT.").get("entity") == []


# ------------------------------------------------------------ scope
@pytest.mark.parametrize("o,s", [
    ("All tenants must pay rent.", "Some tenants must pay rent."),
    ("Every employee must wear a badge.", "Certain employees must wear a badge."),
    ("The fee applies to all vehicles.", "The fee applies only to vehicles."),
    ("The tenant may use the garden.", "The tenant may use only the garden."),
])
def test_scope_changes(o, s):
    assert "scope_change" in types(o, s)


@pytest.mark.parametrize("o,s", [
    ("All employees must wear badges.", "Every employee must wear a badge."),
    ("Each tenant must sign.", "Every tenant must sign."),
    ("Rent is due each month.", "Rent is due monthly."),
    ("The parties help each other.", "The parties help one another."),
    ("Not at all liable.", "Not at all liable."),
])
def test_scope_harmless(o, s):
    assert "scope_change" not in types(o, s)


def test_only_if_is_a_condition_not_scope():
    f = extract("The tenant may leave only if notified.")
    assert f.get("scope") == [] and f.get("condition")[0].attrs["necessary"] is True


# ------------------------------------------------------------ conditions
def test_condition_removed_and_added():
    fs = find_changes("The tenant may leave if the landlord is notified.", "The tenant may leave.", "en")
    c = next(f for f in fs if f.error_type == "condition_removed")
    assert c.evidence_strength == "moderate" and c.original_span.text.startswith("if the landlord")
    assert "condition_added" in types("The tenant may leave.", "The tenant may leave if the landlord is notified.")


@pytest.mark.parametrize("o,s", [
    ("The tenant may leave if notified.", "The tenant may leave provided that he is notified."),
    ("The tenant may leave if notified.", "The tenant may leave in case he is notified."),
    ("The tenant may leave if notified.", "The tenant may leave subject to notice."),
    ("The tenant may leave if notified.", "The tenant may leave when notified."),
    ("The tenant may leave when notified.", "The tenant may leave if notified."),
    ("The tenant is bound even if it rains.", "The tenant is bound even if it rains."),
])
def test_condition_paraphrases_not_flagged(o, s):
    assert find_changes(o, s, "en") == []


def test_only_if_vs_if_is_strength_change():
    assert "condition_strength_change" in types("Rent is waived only if the landlord agrees.",
                                                "Rent is waived if the landlord agrees.")


def test_even_if_and_as_if_are_not_conditions():
    assert extract("He is bound even if it rains.").get("condition") == []
    assert extract("He acts as if he owns it.").get("condition") == []


# ------------------------------------------------------------ exceptions
def test_exception_removed_and_added():
    fs = find_changes("The tenant must pay rent unless the landlord waives it.", "The tenant must pay rent.", "en")
    e = next(f for f in fs if f.error_type == "exception_removed")
    assert e.evidence_strength == "strong" and e.original_span.text.startswith("unless")
    assert "exception_added" in types("The tenant must pay rent.", "The tenant must pay rent except in March.")


@pytest.mark.parametrize("o,s", [
    ("The tenant must pay rent unless the landlord waives it.", "The tenant must pay rent except if the landlord waives it."),
    ("The rule applies to all vehicles other than buses.", "The rule applies to all vehicles except buses."),
    ("The tenant must pay rent unless the landlord waives it.", "The tenant must pay rent if the landlord does not waive it."),
])
def test_exception_paraphrases_not_flagged(o, s):
    assert not types(o, s) & {"exception_removed", "exception_added", "condition_removed", "condition_added"}


def test_does_not_apply_counts_as_exception_without_negation_double_count():
    f = extract("This clause does not apply to minors.")
    assert f.get("exception") and f.get("negation") == []


# ------------------------------------------------------------ omission heuristic
def test_length_heuristic_weak_only():
    o = ("The contractor shall complete the renovation of the kitchen bathroom and garage according to the approved "
         "drawings schedule and budget agreed between everyone involved")
    fs = [f for f in find_changes(o, "The contractor shall renovate.", "en") if f.error_type == "possible_omission_or_addition"]
    assert fs and fs[0].evidence_strength == "weak" and fs[0].original_span is None


# ------------------------------------------------------------ whole-pair behaviour
@pytest.mark.parametrize("s", [
    "The tenant must pay the rent by the fifth day of each month.",
    "Rent is Rs. 10,000 per month, payable to the landlord unless the court orders otherwise.",
    "All tenants may use the garden only if the owner agrees.",
])
def test_identical_pairs_produce_no_findings(s):
    assert find_changes(s, s, "en") == []


def test_unrelated_pair_flags_something_but_makes_no_lmp_claim():
    r = explain("The tenant must pay rent by the fifth day.", "Gardens are lovely in spring.")
    assert r.findings  # rules fire; whether the pair is 'unrelated' is the regression model's job
    assert not hasattr(r, "lmp_score")


def test_harmless_paraphrase_pair_from_deck():
    assert find_changes("The tenant must pay the rent by the fifth day of each month.",
                        "The tenant is required to pay rent no later than the fifth day of every month.", "en") == []


def test_multiple_changes_in_one_pair_all_reported():
    r = explain("All tenants must pay Rs. 10,000 by the fifth day unless the landlord agrees.",
                "Some tenants may pay Rs. 1,000 by the tenth day.")
    got = {f.error_type for f in r.findings}
    assert {"scope_change", "obligation_permission_change", "monetary_amount_change", "date_or_deadline_change",
            "exception_removed"} <= got


def test_findings_sorted_strongest_first_and_deterministic():
    o, s = "All tenants must pay Rs. 10,000 by the fifth day.", "Some tenants may pay Rs. 1,000 by the tenth day."
    a, b = find_changes(o, s, "en"), find_changes(o, s, "en")
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]
    order = [STRENGTHS.index(f.evidence_strength) for f in a]
    assert order == sorted(order)


def test_min_strength_filter():
    o, s = "The tenant must pay rent.", "The tenant pays rent."
    assert find_changes(o, s, "en", "weak") and find_changes(o, s, "en", "moderate") == []


# ------------------------------------------------------------ output structure / highlighting
def test_spans_point_at_the_quoted_text():
    o, s = "The tenant must pay Rs. 10,000 by 15 March 2026.", "The tenant may pay Rs. 1,000 by 16 March 2026."
    for f in find_changes(o, s, "en"):
        if f.original_span:
            assert o[f.original_span.start:f.original_span.end] == f.original_span.text
        if f.simplified_span:
            assert s[f.simplified_span.start:f.simplified_span.end] == f.simplified_span.text


def test_result_is_json_serialisable_and_has_required_fields():
    d = explain("The tenant must pay rent.", "The tenant may pay rent.").to_dict()
    json.dumps(d)
    f = d["findings"][0]
    assert {"error_type", "original_span", "simplified_span", "explanation", "evidence_strength", "change_kind"} <= set(f)
    assert f["error_type"] in ERROR_TYPES and f["evidence_strength"] in STRENGTHS
    assert "not legal advice" in d["disclaimer"] and "regression model" in d["disclaimer"]


def test_no_findings_summary_does_not_claim_preservation():
    r = explain("Rent is due.", "Rent is due.")
    assert "does not show that the meaning is preserved" in r.summary


def test_html_highlighting_escapes_markup():
    o, s = "The tenant must pay <b>rent</b>.", "The tenant may pay <b>rent</b>."
    ho, hs = highlight_html(o, s, find_changes(o, s, "en"))
    assert "<mark>must</mark>" in ho and "&lt;b&gt;" in ho and "<b>" not in ho


def test_mark_merges_overlapping_spans():
    from app.explanations import Span
    t = "abcdefgh"
    assert mark(t, [Span("bcd", 1, 4), Span("def", 3, 6)]) == "a[[bcdef]]gh"
    assert to_html("a<b", []) == "a&lt;b"


def test_pair_differences_alignment():
    from app.explanations.types import Mention, Span
    mk = lambda k: Mention("x", k, Span("", 0, 0))
    matched, changed, oa, ob = pair_differences([mk(1), mk(2), mk(3)], [mk(1), mk(9)])
    assert len(matched) == 1 and len(changed) == 1 and len(oa) == 1 and ob == []


# ------------------------------------------------------------ validation / languages
@pytest.mark.parametrize("o,s", [("", "x"), ("x", ""), ("   ", "x"), ("x", "\n\t")])
def test_empty_inputs_rejected(o, s):
    with pytest.raises(ValueError):
        explain(o, s)


def test_non_string_and_too_long_rejected():
    with pytest.raises(ValueError):
        explain(None, "x")
    with pytest.raises(ValueError):
        explain("x" * 20_001, "y")
    explain("word " * 3000, "word " * 3000)  # long but allowed


def test_bad_min_strength_rejected():
    with pytest.raises(ValueError):
        explain("a b", "a c", min_strength="certain")


def test_hindi_is_not_supported_yet():
    assert SUPPORTED_LANGUAGES == ("en",)
    with pytest.raises(UnsupportedLanguageError):
        explain("किरायेदार को किराया देना होगा।", "किरायेदार किराया दे सकता है।", language="hi")
    with pytest.raises(UnsupportedLanguageError):
        explain("a", "b", language="fr")
    assert issubclass(UnsupportedLanguageError, ValueError)


def test_explanations_are_independent_of_the_regression_model():
    import subprocess
    import sys

    code = """
import sys
import app.explanations as pkg

assert not any(
    m.split(".")[0] in ("torch", "transformers")
    for m in sys.modules
)

src = open(pkg.__file__).read()
assert "scoring" not in src
assert "TfidfBaseline" not in src
"""

    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------ regressions found by probing untuned pairs
def test_failure_to_is_a_negation_and_weak_condition_not_a_removed_negation():
    o = "Failure to pay rent may result in termination of the lease."
    s = "If the tenant fails to pay rent, the lease may be terminated."
    assert not types(o, s, "moderate") & {"negation_change", "condition_added", "condition_removed"}
    assert "negation_change" in types("The tenant fails to pay rent.", "The tenant pays rent.")


@pytest.mark.parametrize("o,s,flag", [
    ("The seller must refund the buyer in full.", "The seller must refund the buyer in part.", True),
    ("The seller must refund the buyer in full.", "The seller must refund the buyer fully.", False),
    ("The seller must refund the buyer in full.", "The seller must refund the buyer partly.", True),
])
def test_in_full_vs_in_part(o, s, flag):
    assert ("scope_change" in types(o, s, "moderate")) is flag


@pytest.mark.parametrize("o,s,flag", [
    ("The tenant must give notice before leaving.", "The tenant must give notice after leaving.", True),
    ("The tenant must give notice prior to leaving.", "The tenant must give notice before leaving.", False),
    ("The landlord inspects after delivery.", "The landlord inspects following delivery.", False),
    ("Read the following terms.", "Read the terms.", False),
])
def test_before_after_sequence(o, s, flag):
    assert ("bound_or_direction_change" in types(o, s)) is flag


def test_before_after_on_a_date_is_reported_once():
    fs = find_changes("Pay before 15 March 2026.", "Pay after 15 March 2026.", "en")
    assert [f.rule_id for f in fs] == ["date.bound"]


# ------------------------------------------------------------ KNOWN LIMITATIONS (documented in docs/EXPLANATIONS.md)
# These pin down behaviour we know is wrong or blind, so a future change cannot silently alter it.
@pytest.mark.parametrize("o,s", [
    ("The tenant shall pay the rent in advance.", "The tenant shall pay the rent in arrears."),   # antonyms
    ("Liability is limited to direct damages.", "Liability is unlimited."),                        # prefix antonym
    ("The tenant may keep a dog.", "The tenant may keep a pet."),                                  # hypernym
    ("The landlord may enter the premises with notice.", "The landlord may enter the premises."),  # dropped PP condition
    ("The buyer shall pay within 30 days of delivery.", "The buyer shall pay within 30 days of invoice."),  # trigger event
])
def test_known_false_negatives_are_not_detected(o, s):
    assert types(o, s, "moderate") == set()


def test_known_false_positive_rewriting_a_prohibition_positively():
    assert "obligation_permission_change" in types("The employee shall not disclose confidential information.",
                                                   "The employee must keep confidential information secret.")
