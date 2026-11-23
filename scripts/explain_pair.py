"""CLI: python scripts/explain_pair.py "original" "simplified" [--language en] [--min-strength weak] [--json]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.explanations import STRENGTHS, UnsupportedLanguageError, explain  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule-based legal-change hints for one sentence pair.")
    ap.add_argument("original")
    ap.add_argument("simplified")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-strength", default="weak", choices=STRENGTHS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        r = explain(a.original, a.simplified, a.language, a.min_strength)
    except (ValueError, UnsupportedLanguageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print("ORIGINAL  :", r.highlighted_original)
    print("SIMPLIFIED:", r.highlighted_simplified)
    for f in r.findings:
        print(f"- [{f.evidence_strength}] {f.error_type} ({f.change_kind}): {f.explanation}")
    print(r.summary)
    print(r.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
