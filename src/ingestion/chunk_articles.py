"""
Section 3: Chunking (parent-child pattern).

Two different things need two different chunk sizes:
- SEARCH needs small, precise units -- a whole Article is too broad, the
  embedding gets diluted and precision drops.
- The LLM needs full legal context -- handing it a lone clause without the
  Article it belongs to is how you get answers that are technically-grounded
  but practically misleading.

So this produces two parallel outputs:
- child_chunks.json  -- clause-level units, what gets embedded and searched
- parent_chunks.json -- one per Article, what actually gets fed to the LLM
  once its child chunk is retrieved

Also applies "contextual chunking": every child chunk's embedding text is
prefixed with its Article number/title/Part/Chapter, not just the bare
clause. Prefixing it with what it's actually about is a well-known,
cheap retrieval-quality win.

Usage:
    python -m src.ingestion.chunk_articles 
"""

import argparse
import json
import re


CLAUSE_MARKER_RE = re.compile(r"\((\d{1,2})\)\s+")

MIN_CHARS_TO_SPLIT = 300  # below this, one clean chunk beats fragmenting


def split_into_clauses(body: str) -> list[tuple[str | None, str]]:
    """Split an Article body into (clause_number, clause_text) pairs."""
    matches = list(CLAUSE_MARKER_RE.finditer(body))
    if not matches:
        return [(None, body.strip())]

    expected = 1
    split_points = []
    for m in matches:
        if int(m.group(1)) == expected:
            split_points.append(m)
            expected += 1

    if not split_points:
        return [(None, body.strip())]

    clauses: list[tuple[str | None, str]] = []
    lead_in = body[: split_points[0].start()].strip()
    if lead_in:
        clauses.append((None, lead_in))

    for i, m in enumerate(split_points):
        start = m.end()
        end = split_points[i + 1].start() if i + 1 < len(split_points) else len(body)
        clauses.append((m.group(1), body[start:end].strip()))

    return clauses


def build_chunks(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    child_chunks: list[dict] = []
    parent_chunks: list[dict] = []

    for a in articles:
        num = a["article_number"]

        heading = f"Article {num}"
        if a.get("part_title"):
            heading += f", Part {a.get('part')} ({a['part_title']})"
        if a.get("chapter_title"):
            heading += f", Chapter {a.get('chapter')} ({a['chapter_title']})"

        parent_chunks.append({
            "parent_id": f"article_{num}",
            "article_number": num,
            "title": a["title"],
            "text": f"Article {num}: {a['title']}\n{a['body']}",
            "part": a.get("part"),
            "part_title": a.get("part_title"),
            "chapter": a.get("chapter"),
            "chapter_title": a.get("chapter_title"),
            "references": a.get("references", []),
            "referenced_by": a.get("referenced_by", []),
            "amendment_history": a.get("amendment_history", []),
            "source_pages": a.get("source_pages", []),
        })

        clauses = split_into_clauses(a["body"])
        if len(a["body"]) < MIN_CHARS_TO_SPLIT:
            clauses = [(None, a["body"].strip())]

        for i, (clause_num, clause_text) in enumerate(clauses):
            if not clause_text:
                continue
            clause_label = f", Clause ({clause_num})" if clause_num else ""
            embedding_text = f"{heading} - {a['title']}{clause_label}: {clause_text}"
            child_chunks.append({
                "chunk_id": f"{num}-c{i}",
                "article_number": num,
                "clause_number": clause_num,
                "text": embedding_text,      # what gets embedded
                "raw_text": clause_text,     # clause text alone, no prefix
                "parent_id": f"article_{num}",
            })

    return child_chunks, parent_chunks


def run(input_path: str, child_output: str, parent_output: str) -> tuple[list[dict], list[dict]]:
    with open(input_path, encoding="utf-8") as f:
        articles = json.load(f)

    child_chunks, parent_chunks = build_chunks(articles)

    with open(child_output, "w", encoding="utf-8") as f:
        json.dump(child_chunks, f, indent=2, ensure_ascii=False)
    with open(parent_output, "w", encoding="utf-8") as f:
        json.dump(parent_chunks, f, indent=2, ensure_ascii=False)

    return child_chunks, parent_chunks


if __name__ == "__main__":
    input_file = r"data\processed\articles_with_ref.json"
    child_output_file = r"data\processed\child_chunks.json"
    parent_output_file = r"data\processed\parent_chunks.json" 

    children, parents = run(input_file, child_output_file, parent_output_file)
    print(f"{len(parents)} parent chunks (1 per Article) -> {parent_output_file}")
    print(f"{len(children)} child chunks -> {child_output_file}")
    for c in children[:5]:
        print(f"  [{c['chunk_id']}] {c['text'][:100]}...")