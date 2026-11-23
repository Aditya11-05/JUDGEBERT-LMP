"""Language-neutral helpers: tokenizing, aligning mentions across two sentences, highlighting."""
from __future__ import annotations
import html
import re

# \w plus the Devanagari block, so the same tokenizer can serve Hindi later.
_TOKEN = re.compile(r"[\w\u0900-\u097F]+(?:'[\w]+)?|[^\w\s]")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def pair_differences(a: list, b: list, eq=None):
    """Greedy one-to-one alignment of two mention lists.

    Returns (matched, changed, only_a, only_b):
      matched  - equal pairs
      changed  - leftover items paired in order (same kind of item, different value)
      only_a / only_b - items with no counterpart (omitted / added)
    """
    eq = eq or (lambda x, y: x.key == y.key)
    used: set[int] = set()
    matched, only_a = [], []
    for x in a:
        for j, y in enumerate(b):
            if j not in used and eq(x, y):
                used.add(j)
                matched.append((x, y))
                break
        else:
            only_a.append(x)
    only_b = [y for j, y in enumerate(b) if j not in used]
    n = min(len(only_a), len(only_b))
    return matched, list(zip(only_a[:n], only_b[:n])), only_a[n:], only_b[n:]


def _merge(spans) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted((sp.start, sp.end) for sp in spans if sp is not None):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def mark(text: str, spans, left: str = "[[", right: str = "]]") -> str:
    """Plain-text highlighting: wrap each (merged) span in markers."""
    parts, pos = [], 0
    for s, e in _merge(spans):
        parts += [text[pos:s], left, text[s:e], right]
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


def to_html(text: str, spans) -> str:
    """HTML highlighting with <mark>; all text is escaped."""
    parts, pos = [], 0
    for s, e in _merge(spans):
        parts += [html.escape(text[pos:s]), "<mark>", html.escape(text[s:e]), "</mark>"]
        pos = e
    parts.append(html.escape(text[pos:]))
    return "".join(parts)
