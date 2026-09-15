"""
Section 1: Ingestion + structure-aware parsing.

Turns a single Constitution PDF into a structured JSON list of Articles,
each tagged with its Part, Chapter, Article number, title, and resolved
amendment history.

This version is calibrated against "The Pakistan Code" formatted edition,
which annotates amended/inserted text with superscript footnote markers
tied to per-page footnote definitions (e.g. "Ins. by the Constitution
(Twenty-sixth Amendment) Act, 2024...").

Usage:
    python -m src.ingestion.parse_pdf
"""

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    article_number: str          # e.g. "25", "9A" (amendment-inserted articles)
    title: str                   # e.g. "Equality of citizens."
    body: str                    # full clause text, footnote markers/brackets stripped
    part: Optional[str] = None
    part_title: Optional[str] = None
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    source_pages: list = field(default_factory=list)
    footnote_refs: list = field(default_factory=list)       # [(page, marker), ...] raw refs collected while parsing
    amendment_history: list = field(default_factory=list)   # resolved human-readable strings, filled in post-process


# ---------------------------------------------------------------------------
# Step 1: raw text extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract raw text per page, preserving page numbers for provenance."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


WATERMARK_RE = re.compile(r"^\s*THE\s+PAKISTAN\s+CODE\s*$", re.IGNORECASE)
PAGE_FOOTER_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)


def clean_page_text(text: str) -> str:
 
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.fullmatch(r"-\s*\d{1,4}\s*-", stripped):
            continue
        if WATERMARK_RE.match(stripped):
            continue
        if PAGE_FOOTER_RE.match(stripped):
            continue
        if "constitution of the islamic republic of pakistan" in stripped.lower() and len(stripped) < 80:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Step 2: structure-aware parsing
# ---------------------------------------------------------------------------

PART_RE = re.compile(r"^PART[\s\-]+([IVXLCDM]+)\s*[:\-]?\s*(.*)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER[\s\-]+(\d+)\s*[:\-]?\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(\d{1,3}[A-Z]?)\.\s+([A-Z][^.]{0,150}\.)\s*(.*)$")


FOOTNOTE_DEF_RE = re.compile(
    r"^([\u2070-\u2079\u00b9\u00b2\u00b3]+|\d{1,2})\s*"
    r"((?:Subs\.|Ins\.|Am\.|Rep\.|Add\.|Omitted).*)$"
)

_SUPERSCRIPT_CHARS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUP_TRANSLATE = str.maketrans(_SUPERSCRIPT_CHARS, "0123456789")
_INLINE_MARKER_RE = re.compile(r"[\u2070-\u2079\u00b9\u00b2\u00b3]+")


def strip_inline_markers(line: str) -> tuple[str, list[str]]:

    markers = [m.translate(_SUP_TRANSLATE) for m in _INLINE_MARKER_RE.findall(line)]
    cleaned = _INLINE_MARKER_RE.sub("", line)
    return cleaned, markers


def parse_structure(full_text: str) -> tuple[list[Article], dict]:

    articles: list[Article] = []
    page_footnotes: dict[int, dict[str, str]] = {}

    current_part = current_part_title = None
    current_chapter = current_chapter_title = None
    current: Optional[Article] = None
    current_page = 0
    awaiting_part_title = False
    awaiting_chapter_title = False

    page_marker_re = re.compile(r"^\x00PAGE:(\d+)\x00$")

    for raw_line in full_text.split("\n"):
        stripped_raw = raw_line.strip("\n")
        page_match = page_marker_re.match(stripped_raw)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        line = stripped_raw.strip()
        if not line:
            continue

        # Footnote DEFINITION line -- capture as data, never as body text.
        footnote_match = FOOTNOTE_DEF_RE.match(line)
        if footnote_match:
            marker, text = footnote_match.groups()
            marker = marker.translate(_SUP_TRANSLATE)
            page_footnotes.setdefault(current_page, {})[marker] = text.strip()
            continue

        # Strip inline reference markers + bracket noise before structural matching.
        cleaned_line, line_markers = strip_inline_markers(line)
        cleaned_line = cleaned_line.replace("[", "").replace("]", "").strip()
        if not cleaned_line:
            continue
        refs = [(current_page, m) for m in line_markers]

        part_match = PART_RE.match(cleaned_line)
        if part_match:
            current_part, inline_title = part_match.groups()
            current_part_title = inline_title.strip() or None
            current_chapter, current_chapter_title = None, None
            awaiting_part_title = current_part_title is None
            awaiting_chapter_title = False
            continue

        chapter_match = CHAPTER_RE.match(cleaned_line)
        if chapter_match:
            current_chapter, inline_title = chapter_match.groups()
            current_chapter_title = inline_title.strip() or None
            awaiting_chapter_title = current_chapter_title is None
            continue

        is_header_line = bool(ARTICLE_RE.match(cleaned_line))

        if awaiting_part_title and not is_header_line:
            current_part_title = cleaned_line
            awaiting_part_title = False
            continue
        awaiting_part_title = False

        if awaiting_chapter_title and not is_header_line:
            current_chapter_title = cleaned_line
            awaiting_chapter_title = False
            continue
        awaiting_chapter_title = False

        article_match = ARTICLE_RE.match(cleaned_line)
        if article_match:
            if current is not None:
                articles.append(current)
            num, title, rest_of_line = article_match.groups()
            body = rest_of_line.strip().lstrip("\u2014-").strip()  # drop leading em-dash
            current = Article(
                article_number=num,
                title=title.strip(),
                body=body,
                part=current_part,
                part_title=current_part_title,
                chapter=current_chapter,
                chapter_title=current_chapter_title,
                source_pages=[current_page],
                footnote_refs=refs,
            )
            continue

        # Continuation of the current article's body.
        if current is not None:
            current.body = (current.body + " " + cleaned_line).strip()
            if current_page not in current.source_pages:
                current.source_pages.append(current_page)
            current.footnote_refs.extend(refs)

    if current is not None:
        articles.append(current)

    return articles, page_footnotes


def resolve_amendment_history(articles: list[Article], page_footnotes: dict) -> None:
    """Fill in each Article's amendment_history by resolving its
    (page, marker) footnote_refs against the per-page footnote definitions."""
    for a in articles:
        for page, marker in a.footnote_refs:
            note = page_footnotes.get(page, {}).get(marker)
            if note and note not in a.amendment_history:
                a.amendment_history.append(note)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_PAGE_MARKER = "\x00PAGE:{}\x00"


def tag_pages_with_markers(pages: list[tuple[int, str]]) -> str:
    parts = []
    for page_num, text in pages:
        parts.append(_PAGE_MARKER.format(page_num))
        parts.append(clean_page_text(text))
    return "\n".join(parts)


def run(pdf_path: str, output_path: str) -> tuple[list[Article], dict]:
    pages = extract_pages(pdf_path)
    full_text = tag_pages_with_markers(pages)
    articles, page_footnotes = parse_structure(full_text)
    resolve_amendment_history(articles, page_footnotes)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, indent=2, ensure_ascii=False)

    return articles, page_footnotes


if __name__ == "__main__":
    input_file = r"data\raw\consitiutation of the pakistan.pdf"
    output_file = r"data\processed\articles.json"

    result, footnotes = run(input_file, output_file)

    print(f"Parsed {len(result)} articles -> {output_file}")

    n_amended = sum(1 for a in result if a.amendment_history)
    print(f"{n_amended} articles have resolved amendment history")

    for a in result[:6]:
        tag = " [AMENDED]" if a.amendment_history else ""
        print(f"  Article {a.article_number} ({a.part_title}): {a.title}{tag}")

        for note in a.amendment_history:
            print(f"      -> {note}")