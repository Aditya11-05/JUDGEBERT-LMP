"""English extractors: turn one sentence into Features (modals, negation, quantities, parties, ...).

All patterns are surface heuristics. Comparison logic lives in rules.py and is language-neutral,
so a Hindi rule set only needs its own extract() (see docs/EXPLANATIONS.md).
"""
from __future__ import annotations
import re
from .types import Features, Mention, Span

LANGUAGE = "en"


def _norm(t: str) -> str:
    return t.replace("\u2019", "'").replace("\u2018", "'")  # same length, offsets preserved


# ---------------------------------------------------------------- numbers
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7, "million": 1e6, "billion": 1e9}
_SCALE_VAL = dict(_SCALES, k=1e3)
_NUMWORD = r"\b(?:" + "|".join(list(_UNITS) + list(_TENS) + ["hundred"] + list(_SCALES)) + r")\b"
_NUMPHRASE = _NUMWORD + r"(?:[\s-]+(?:and[\s-]+)?" + _NUMWORD + r")*"
_NUMERAL = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_SCALE_RX = r"(?:\s*(?:lakhs?|crores?|million|billion|thousand|k)\b)"
_QTY = "(?:" + _NUMERAL + _SCALE_RX + "?|" + _NUMPHRASE + ")"


def _words_value(s: str) -> float:
    total = cur = 0.0
    for w in re.split(r"[\s-]+", s):
        if w == "and":
            continue
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in _SCALES:
            total += max(cur, 1) * _SCALES[w]
            cur = 0
    return total + cur


def qty_value(s: str) -> float:
    s = s.strip().lower()
    m = re.fullmatch("(" + _NUMERAL + r")\s*(lakhs?|crores?|million|billion|thousand|k)?", s)
    if m:
        v = float(m.group(1).replace(",", ""))
        return v * _SCALE_VAL[m.group(2)] if m.group(2) else v
    return _words_value(s)


# ---------------------------------------------------------------- money / percent / duration
_CUR = r"(?:₹|\$|€|£|Rs\.?|INR|USD|EUR|GBP)"
_CURW = r"(?:rupees?|dollars?|euros?|pounds?|INR|USD|EUR|GBP|Rs\.?)"
_CUR_MAP = {"₹": "INR", "rs": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
            "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
            "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
            "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP"}
MONEY = re.compile(r"(?<![\w.,])(?:(?P<c1>" + _CUR + r")\s*(?P<a1>" + _QTY + r")|(?P<a2>" + _QTY +
                   r")\s*(?P<c2>" + _CURW + r")(?!\w))", re.I)
PERCENT = re.compile(r"(?<![\w.,])(?P<p>" + _QTY + r")\s*(?:%|per\s?cent\b|percent\b|percentage\s+points?\b)", re.I)
DURATION = re.compile(r"(?<![\w.,])(?P<n>" + _QTY + r")(?:\s*\(\s*\d+\s*\))?[\s-]*"
                      r"(?P<pre>calendar\s+|business\s+|working\s+)?(?P<u>days?|weeks?|months?|years?|hours?)\b", re.I)

_QUALS = [
    ("at_least", r"at\s+least|a\s+minimum\s+of|minimum\s+of|not\s+less\s+than|no\s+less\s+than|no\s+fewer\s+than|not\s+fewer\s+than"),
    ("at_most", r"at\s+most|up\s+to|no\s+more\s+than|not\s+more\s+than|a\s+maximum\s+of|maximum\s+of|not\s+exceeding|not\s+exceed"),
    ("more_than", r"more\s+than|over|exceeding|in\s+excess\s+of|greater\s+than"),
    ("less_than", r"less\s+than|under|fewer\s+than|below"),
    ("deadline_upper", r"within|no\s+later\s+than|not\s+later\s+than|by|before|prior\s+to|until|till"),
    ("deadline_lower", r"after|no\s+earlier\s+than|not\s+earlier\s+than|following"),
    ("exactly", r"exactly"),
]
_QUAL_RX = [(n, re.compile(r"\b(?:" + p + r")\s+(?:the\s+|a\s+|an\s+|each\s+)?$", re.I)) for n, p in _QUALS]


def _qualifier(t: str, start: int) -> str | None:
    window = t[max(0, start - 30):start]
    best = None
    for name, rx in _QUAL_RX:
        m = rx.search(window)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


_GAP = re.compile(r"[\s(\[,]*(?:i\.e\.|that is|being)?[\s(\[,]*")


def _is_dup(prev, key, start: int, t: str) -> bool:
    """'ten thousand rupees (Rs. 10,000)' is one amount, not two."""
    return prev is not None and prev[0] == key and _GAP.fullmatch(t[prev[1]:start]) is not None


def _dur_key(n: float, pre: str | None, unit: str):
    u = unit.lower().rstrip("s")
    if (pre or "").strip().lower() in ("business", "working"):
        return (f"business_{u}", n)
    return {"day": ("day", n), "week": ("day", n * 7), "month": ("month", n),
            "year": ("month", n * 12), "hour": ("hour", n)}[u]


# ---------------------------------------------------------------- dates
_MON = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)")
_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
DATE = re.compile(
    r"(?<![\w-])(?P<y4>\d{4})-(?P<m4>\d{2})-(?P<d4>\d{2})(?!\w)"
    r"|(?<![\w/.-])(?P<a>\d{1,2})[/.-](?P<b>\d{1,2})[/.-](?P<y3>\d{4}|\d{2})(?![\w/])"
    r"|(?<![\w])(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>" + _MON + r")\b\.?(?:,?\s*(?P<y1>\d{4}))?"
    r"|(?<![\w])(?P<m2>" + _MON + r")\b\.?\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(?P<y2>\d{4}))?", re.I)

_ORD_W = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
          "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth").split()
_ORD_MAP = {w: i + 1 for i, w in enumerate(_ORD_W[:19])}
_ORD_MAP.update({"twentieth": 20, "thirtieth": 30})
for _t, _b in (("twenty", 20), ("thirty", 30)):
    for _w, _i in zip(_ORD_W[:9], range(1, 10)):
        _ORD_MAP[f"{_t}-{_w}"] = _b + _i
_ORD = (r"(?:\d{1,2}(?:st|nd|rd|th)|(?:twenty|thirty)[\s-](?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|"
        + "|".join(_ORD_W) + r")")
ORD_DAY = re.compile(r"(?<![\w])(?:(?P<o>" + _ORD + r")\s+(?:day|date)\b|(?P<o2>\d{1,2}(?:st|nd|rd|th))\s+of\s+"
                     r"(?:(?P<fq>each|every)|the|that)\s+month\b)", re.I)
FREQ = re.compile(r"\b(?:(?:each|every)\s+(?:calendar\s+)?(?P<u1>day|week|month|quarter|year)|"
                  r"(?P<f2>daily|weekly|monthly|quarterly|annually|yearly)|per\s+(?P<u3>day|week|month|quarter|annum|year))\b", re.I)
_FREQ_CANON = {"day": "daily", "daily": "daily", "week": "weekly", "weekly": "weekly", "month": "monthly",
               "monthly": "monthly", "quarter": "quarterly", "quarterly": "quarterly", "year": "yearly",
               "yearly": "yearly", "annually": "yearly", "annum": "yearly"}


def _ord_value(s: str) -> int:
    s = s.lower().replace(" ", "-")
    return int(re.sub(r"\D", "", s)) if s[0].isdigit() else _ORD_MAP[s]


# ---------------------------------------------------------------- modality
_PROHIB = [r"must\s+not", r"mustn't", r"shall\s+not", r"shan't", r"may\s+not", r"cannot", r"can\s+not", r"can't",
           r"may\s+never", r"(?:is|are)\s+not\s+(?:allowed|permitted|entitled|authori[sz]ed)\s+to",
           r"(?:is|are)\s+(?:prohibited|forbidden|barred)\s+(?:from|to)"]
_NOOBL = [r"need\s+not", r"needn't", r"(?:does|do)\s+not\s+(?:have|need)\s+to", r"doesn't\s+have\s+to",
          r"don't\s+have\s+to", r"(?:is|are)\s+not\s+(?:required|obliged|obligated)\s+to"]
_OBLIG = [r"must", r"shall", r"(?:is|are)\s+(?:required|obliged|obligated|bound)\s+to", r"(?:has|have)\s+to",
          r"needs?\s+to", r"undertakes?\s+to", r"agrees?\s+to"]
_ADVICE = [r"should", r"ought\s+to", r"(?:is|are)\s+(?:expected|encouraged)\s+to"]
_PERM = [r"may", r"can", r"(?:is|are)\s+(?:allowed|permitted|entitled|authori[sz]ed|free)\s+to", r"(?:has|have)\s+the\s+right\s+to"]
MODAL = re.compile(r"\b(?:(?P<prohib>" + "|".join(_PROHIB) + r")|(?P<noobl>" + "|".join(_NOOBL) + r")|(?P<oblig>"
                   + "|".join(_OBLIG) + r")|(?P<advice>" + "|".join(_ADVICE) + r")|(?P<perm>" + "|".join(_PERM) + r"))\b", re.I)
_MODAL_CAT = {"prohib": "prohibition", "noobl": "no_obligation", "oblig": "obligation",
              "advice": "advice", "perm": "permission"}
_NEG_BEFORE_MODAL = re.compile(r"\b(no|nobody|none|neither)\b(?:\s+[\w'-]+){0,3}\s*$", re.I)
_MONTH_BEFORE = re.compile(r"\b(?:in|of|during|until|till|before|after|since|from|early|late|mid|end\s+of|on)\s+$", re.I)

# ---------------------------------------------------------------- negation / exceptions / conditions / scope
_NEG_IDIOM = re.compile(r"\b(?:no|not)\s+(?:later|earlier|less|more|fewer|greater)\s+than\b|\bnot\s+(?:exceeding|exceed|only)\b|"
                        r"\bwhether\s+or\s+not\b|\bnot\s+limited\s+to\b", re.I)
_NEG = re.compile(r"\b(?:not|no|never|neither|none|nothing|nobody|without|cannot)\b|\b\w+n't\b|"
                  r"\b(?:fails?|failed|failing|failure|refus(?:es|ed|ing|e)|unable)\s+to\b|\bineligible\b", re.I)
_EXC = re.compile(r"\b(?:unless|except(?:ing)?(?:\s+(?:for|that|when|where|as|in))?|other\s+than|save\s+(?:for|that|as)|"
                  r"apart\s+from|aside\s+from|with\s+the\s+exception\s+of|excluding|exclusive\s+of|"
                  r"(?:does|do|shall)\s+not\s+apply|not\s+applicable|but\s+not)\b", re.I)
_COND = re.compile(r"\b(?P<only>only\s+if|if\s+and\s+only\s+if)\b|(?<!even )(?<!as )(?<!what )\bif\b|"
                   r"\b(?:provided\s+that|provided\s+(?:however|further)|providing\s+that|subject\s+to|in\s+the\s+event\s+(?:that|of)|"
                   r"in\s+case(?:\s+of)?|on\s+(?:the\s+)?condition\s+that|as\s+long\s+as|so\s+long\s+as)\b", re.I)
_COND_WEAK = re.compile(r"\b(?:when|whenever|once|upon|where|failure\s+to)\b", re.I)
_CLAUSE_END = re.compile(r"[,;]|\.\s+(?=[A-Z])|\.\s*$")
_SCOPE = [
    ("universal", re.compile(r"(?<!at )\b(?:all|every|each|entire|whole|everyone|everybody|always|any\s+and\s+all|in\s+full|fully|entirely|completely|wholly)\b"
                             r"(?!\s+other\b)(?!\s+(?:calendar\s+)?(?:month|year|week|day|quarter)\b)", re.I)),
    ("partial", re.compile(r"\b(?:some|certain|several|a\s+few|most|many|partly|in\s+part|sometimes|occasionally)\b", re.I)),
    ("restrictive", re.compile(r"(?<!not )\b(?:only|solely|exclusively)\b(?!\s+if\b)", re.I)),
]
_SEQ = re.compile(r"\b(?P<before>before|prior\s+to|ahead\s+of)\b|(?<!the )\b(?P<after>after|following|subsequent\s+to)\b", re.I)

# ---------------------------------------------------------------- parties / entities
_ROLES = [  # multiword first so 'insurance company' wins over 'company'
    ("insurer", r"insurance\s+compan(?:y|ies)|insurers?|underwriters?"),
    ("insured", r"policy\s?holders?|insured\s+persons?|insured"),
    ("tenant", r"tenants?|lessees?|renters?"), ("landlord", r"landlords?|landlady|lessors?"),
    ("buyer", r"buyers?|purchasers?"), ("seller", r"sellers?|vendors?"),
    ("employer", r"employers?"), ("employee", r"employees?"),
    ("borrower", r"borrowers?"), ("lender", r"lenders?"), ("creditor", r"creditors?"), ("debtor", r"debtors?"),
    ("licensor", r"licensors?"), ("licensee", r"licensees?"), ("contractor", r"contractors?"),
    ("client", r"clients?"), ("customer", r"customers?"), ("supplier", r"suppliers?"),
    ("guarantor", r"guarantors?|sureties|surety"), ("owner", r"owners?"),
    ("company", r"compan(?:y|ies)|corporations?"), ("court", r"courts?"), ("tribunal", r"tribunals?"),
    ("government", r"governments?"), ("authority", r"authorit(?:y|ies)"), ("trustee", r"trustees?"),
    ("beneficiary", r"beneficiar(?:y|ies)"), ("agent", r"agents?"), ("partner", r"partners?"),
    ("shareholder", r"shareholders?"), ("director", r"directors?"), ("guardian", r"guardians?"),
    ("applicant", r"applicants?"), ("respondent", r"respondents?"), ("plaintiff", r"plaintiffs?"),
    ("defendant", r"defendants?"), ("consumer", r"consumers?"), ("patient", r"patients?"),
]
_ROLE_RX = [(c, re.compile(r"\b(?:" + p + r")\b", re.I)) for c, p in _ROLES]
_PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+\s+by\b", re.I)
_ENT = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:\s+(?:of|for|and|the)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)*)\b")
_CONN = {"of", "for", "and", "the"}
_STOPCAP = {"The", "This", "That", "These", "Those", "Any", "All", "Each", "Every", "If", "Unless", "Except", "Subject",
            "In", "On", "At", "By", "For", "To", "Of", "And", "Or", "Such", "A", "An", "It", "He", "She", "They", "We",
            "You", "Section", "Article", "Clause", "Schedule", "Rule", "Agreement", "Party", "Parties", "Notice",
            "Mr", "Mrs", "Ms", "Dr", "Shri", "Smt", "Rs", "No", "Where", "When", "Once", "Upon", "Provided", "Failure"}
_NOT_ENTITY = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
               "inr", "usd", "eur", "gbp"}
_ABBREV = {"mr", "mrs", "ms", "dr", "sr", "jr", "shri", "smt", "no", "rs", "st", "vs", "v"}
_STOP = {"that", "this", "with", "from", "shall", "must", "will", "have", "been", "were", "they", "their", "them", "which",
         "there", "when", "then", "than", "into", "such", "also", "each", "every", "your", "what", "does", "while", "where", "about"}


def _clause_end(t: str, start: int) -> int:
    m = _CLAUSE_END.search(t, start, min(len(t), start + 140))
    end = m.start() if m else min(len(t), start + 140)
    while end > start and t[end - 1].isspace():
        end -= 1
    return end


def _sentence_starts(t: str) -> set[int]:
    starts = {0}
    for m in re.finditer(r"[.!?]\s+", t):
        prev = re.findall(r"[A-Za-z/]+", t[:m.start() + 1])
        if prev and prev[-1].lower() in _ABBREV:
            continue
        starts.add(m.end())
    return starts


def entity_eq(x: Mention, y: Mention) -> bool:
    a, b = x.key, y.key
    if a == b:
        return True

    def initials(k: str) -> str:
        return "".join(w[0] for w in k.split() if w not in _CONN)
    return (x.attrs.get("acronym") and initials(b) == a.replace(" ", "")) or \
           (y.attrs.get("acronym") and initials(a) == b.replace(" ", ""))


# ---------------------------------------------------------------- main entry
def extract(text: str) -> Features:
    t = _norm(text)
    f = Features(text=text)
    used: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in used)

    def add(kind, key, s, e, **attrs):
        f.add(Mention(kind, key, Span(text[s:e], s, e), attrs))

    # money
    prev = None
    for m in MONEY.finditer(t):
        s, e = m.span()
        cur = (m.group("c1") or m.group("c2")).lower().rstrip(".")
        key = (_CUR_MAP[cur], round(qty_value(m.group("a1") or m.group("a2")), 2))
        used.append((s, e))
        if not _is_dup(prev, key, s, t):
            add("money", key, s, e, qualifier=_qualifier(t, s))
        prev = (key, e)
    # percent
    prev = None
    for m in PERCENT.finditer(t):
        s, e = m.span()
        if not free(s, e):
            continue
        key = round(qty_value(m.group("p")), 4)
        used.append((s, e))
        if not _is_dup(prev, key, s, t):
            add("percent", key, s, e, qualifier=_qualifier(t, s))
        prev = (key, e)
    # explicit dates
    for m in DATE.finditer(t):
        s, e = m.span()
        if not free(s, e):
            continue
        g = m.groupdict()
        if g["y4"]:
            y, mo, d = int(g["y4"]), int(g["m4"]), int(g["d4"])
        elif g["a"]:
            d, mo, y = int(g["a"]), int(g["b"]), int(g["y3"])
            if mo > 12 >= d:
                d, mo = mo, d
            y = y + 2000 if y < 100 else y
        else:
            mon = g["m1"] or g["m2"]
            if mon == "may":  # lowercase 'may' is the modal verb, not the month
                continue
            d = int(g["d1"] or g["d2"])
            mo = _MONTHS[mon[:3].lower()]
            y = int(g["y1"] or g["y2"]) if (g["y1"] or g["y2"]) else None
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            continue
        used.append((s, e))
        add("date", (y, mo, d), s, e, qualifier=_qualifier(t, s))
    # durations
    prev = None
    for m in DURATION.finditer(t):
        s, e = m.span()
        if not free(s, e):
            continue
        key = _dur_key(qty_value(m.group("n")), m.group("pre"), m.group("u"))
        used.append((s, e))
        add("duration", key, s, e, qualifier=_qualifier(t, s))
    # ordinal day of month ("the fifth day", "5th of each month")
    for m in ORD_DAY.finditer(t):
        s, e = m.span()
        if not free(s, e):
            continue
        used.append((s, e))
        add("day_of_month", _ord_value(m.group("o") or m.group("o2")), s, e, qualifier=_qualifier(t, s))
        if m.group("fq"):  # "5th of every month" also states a monthly frequency
            add("frequency", "monthly", m.start("fq"), e)
    # frequency
    for m in FREQ.finditer(t):
        s, e = m.span()
        if not free(s, e):
            continue
        used.append((s, e))
        add("frequency", _FREQ_CANON[(m.group("u1") or m.group("f2") or m.group("u3")).lower()], s, e)

    # modality
    for m in MODAL.finditer(t):
        grp = m.lastgroup
        s, e = m.span()
        if grp == "perm" and m.group(0) == "May":
            before, after = t[max(0, s - 14):s], t[e:e + 12]
            if re.match(r"\s+\d", after) or _MONTH_BEFORE.search(before) or re.search(r"\d(?:st|nd|rd|th)?\s+(?:of\s+)?$", before, re.I):
                continue  # the month, not the modal verb
        via = False
        if grp in ("perm", "oblig"):
            nb = _NEG_BEFORE_MODAL.search(t[max(0, s - 40):s])
            if nb:  # "no party may ..." is a prohibition, and that 'no' is already accounted for
                grp, via = "prohib", True
                f.neg_consumed.append((max(0, s - 40) + nb.start(1), max(0, s - 40) + nb.end(1)))
        if grp in ("prohib", "noobl"):
            f.neg_consumed.append((s, e))
        add("modality", _MODAL_CAT[grp], s, e, via_negator=via)

    # exceptions
    for m in _EXC.finditer(t):
        s = m.start()
        e = max(_clause_end(t, s), m.end())
        cue = m.group(0).lower()
        if re.search(r"not\s+apply|not\s+applicable|but\s+not", cue):
            f.neg_consumed.append(m.span())
        add("exception", "exception", s, e, cue=cue, unless=cue.startswith("unless"))

    # negation
    idioms = [mm.span() for mm in _NEG_IDIOM.finditer(t)]
    for m in _NEG.finditer(t):
        s, e = m.span()
        if any(a <= s and e <= b for a, b in idioms + f.neg_consumed):
            continue
        if m.group(0).lower() == "no" and re.match(r"\.?\s*\d", t[e:]):
            continue  # "No. 5"
        add("negation", "neg", s, e)

    # conditions
    for m in _COND.finditer(t):
        s = m.start()
        add("condition", "condition", s, max(_clause_end(t, s), m.end()), necessary=bool(m.group("only")))
    for m in _COND_WEAK.finditer(t):
        add("condition_weak", "condition_weak", m.start(), m.end())

    # scope
    for cls, rx in _SCOPE:
        for m in rx.finditer(t):
            add("scope", cls, m.start(), m.end())

    # before/after ordering of events (when not already a qualifier of a number or date)
    qstarts = [m.span.start for k in ("money", "percent", "date", "duration", "day_of_month") for m in f.mentions.get(k, [])]
    for m in _SEQ.finditer(t):
        if not any(m.end() <= q <= m.end() + 8 for q in qstarts):
            add("sequence", m.lastgroup, m.start(), m.end())

    # parties
    role_ranges: list[tuple[int, int]] = []
    for canon, rx in _ROLE_RX:
        for m in rx.finditer(t):
            if any(m.start() < b and m.end() > a for a, b in role_ranges):
                continue
            role_ranges.append(m.span())
            add("party", canon, m.start(), m.end())
    f.passive = bool(_PASSIVE.search(t))

    # generic numbers (not already part of money/percent/date/duration/...)
    prev = None
    for m in re.finditer(r"(?<![\w.,/-])(" + _NUMERAL + r")(?![\w/])|" + _NUMPHRASE, t, re.I):
        s, e = m.span()
        if not free(s, e):
            continue
        raw = m.group(0)
        if raw.lower() == "one":
            continue  # 'one' is mostly a pronoun/article; counted only inside durations/money
        key = qty_value(raw)
        if _is_dup(prev, key, s, t):
            prev = (key, e)
            continue
        add("number", key, s, e)
        prev = (key, e)

    # named entities (skipped for ALL-CAPS text, where capitalisation carries no information)
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) <= 0.6:
        starts = _sentence_starts(t)
        for m in _ENT.finditer(text):
            s, e = m.span()
            toks = [(mm.group(0), s + mm.start()) for mm in re.finditer(r"\S+", m.group(0))]
            at_start = s in starts
            while toks and toks[0][0] in _STOPCAP:
                toks.pop(0)
            if not toks:
                continue
            s2 = toks[0][1]
            words = [w for w, _ in toks]
            if len(words) == 1 and at_start and not words[0].isupper():
                continue  # sentence-initial capital is not evidence of a name
            key = " ".join(w.lower() for w in words)
            if key in _NOT_ENTITY or key in _CUR_MAP or not free(s2, e):
                continue
            if len(words) == 1 and any(s2 < b and e > a for a, b in role_ranges):
                continue  # a bare defined term such as 'Tenant' is a role, not a name
            add("entity", key, s2, e, acronym=len(words) == 1 and words[0].isupper())

    f.content_tokens = sum(1 for w in re.findall(r"[A-Za-z]{4,}", t) if w.lower() not in _STOP)
    return f
