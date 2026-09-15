"""
Section 2: Cross-reference resolution.

Constitutional text is full of references like "subject to Article 199" or
"notwithstanding anything contained in Article 8". Pure semantic search can
easily miss the referenced Article if it isn't semantically similar to the
user's query -- it only shows up because the *retrieved* Article happens to
name it. Resolving these into explicit graph edges at ingestion time means
retrieval can deliberately pull in referenced Articles instead of hoping
embeddings stumble onto them.

Usage:
    python -m src.ingestion.resolve_crossref
"""

import argparse
import json
import re

# Captures the whole run of article numbers after "Article(s)", e.g.
# "Articles 8, 9 and 10", "Article 199", "Articles 199 to 203".
ARTICLE_REF_RE = re.compile(
    r"Articles?\s+((?:\d{1,3}[A-Z]?)(?:\s*(?:,|and|to|&)\s*\d{1,3}[A-Z]?)*)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"^\d{1,3}[A-Z]?$")
SPLIT_RE = re.compile(r"\s*(?:,|and|&)\s*", re.IGNORECASE)
RANGE_RE = re.compile(r"(\d{1,3}[A-Z]?)\s+to\s+(\d{1,3}[A-Z]?)", re.IGNORECASE)


def _expand_range(a: str, b: str) -> list[str]:

    if a.isdigit() and b.isdigit():
        lo, hi = int(a), int(b)
        if 0 < hi - lo <= 50:  # guard against a mis-parsed, absurdly wide "range"
            return [str(n) for n in range(lo, hi + 1)]
    return [a, b]


def extract_references(text: str, self_number: str) -> list[str]:

    found: list[str] = []
    for match in ARTICLE_REF_RE.finditer(text):
        for token in SPLIT_RE.split(match.group(1)):
            token = token.strip()
            if not token:
                continue
            range_match = RANGE_RE.match(token)
            if range_match:
                found.extend(_expand_range(*range_match.groups()))
            elif NUMBER_RE.match(token):
                found.append(token)

    deduped = []
    for num in found:
        if num != self_number and num not in deduped:
            deduped.append(num)
    return deduped


def run(input_path: str, output_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        articles = json.load(f)

    by_number = {a["article_number"]: a for a in articles}

    for a in articles:
        refs = extract_references(f"{a['title']} {a['body']}", a["article_number"])
  
        a["references"] = [r for r in refs if r in by_number]
        a["unresolved_references"] = [r for r in refs if r not in by_number]
        a["referenced_by"] = []

    for a in articles:
        for ref in a["references"]:
            by_number[ref]["referenced_by"].append(a["article_number"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    return articles


if __name__ == "__main__":
    input_file = r"data\processed\articles.json"
    output_file = r"data\processed\articles_with_ref.json" 

    result = run(input_file, output_file)
    total_refs = sum(len(a["references"]) for a in result)
    print(f"Resolved {total_refs} cross-references across {len(result)} articles -> {output_file}")
    for a in result:
        if a["references"] or a["unresolved_references"]:
            print(f"  Article {a['article_number']}: refs={a['references']} unresolved={a['unresolved_references']}")