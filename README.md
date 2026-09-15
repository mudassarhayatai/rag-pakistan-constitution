# RAG System: Constitution of Pakistan

A production-oriented Retrieval-Augmented Generation system for querying the Constitution of the Islamic Republic of Pakistan fully open-source, citation-grounded, and built to say "I don't know" rather than guess on a legal document where a confident wrong answer is worse than no answer.

## Why this isn't just "chat with a PDF"

Single-document RAG demos are common and usually shallow. This one leans into what actually makes a constitution hard to retrieve and answer correctly from:

- **Cross-references everywhere** — Articles constantly point at each other ("subject to Article 8," "notwithstanding Article 199"). Naive chunking destroys these relationships; this system resolves them into an explicit reference graph at ingestion time.
- **Amendments** — the source document annotates amended/inserted text with footnote markers tied to specific amendment Acts (e.g. Article 9A, the right to a clean environment, was inserted by the 26th Amendment in 2024). This system extracts that provenance as structured data, not just prose.
- **High-stakes accuracy** — a wrong answer about a fundamental right isn't a shrug. Every generated claim is required to cite a specific Article, and the system is instructed and evaluated on whether it actually refuses to answer when the retrieved context doesn't cover the question, rather than filling the gap from the model's general knowledge.

## Architecture

![System Architecture](docs\architecture.png)

Served behind a FastAPI layer (`src/api/main.py`) with models loaded once at startup, not per-request.

## Fully open-source stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 | Apache 2.0, strong multilingual retrieval quality |
| Vector store | Qdrant | Self-hosted, native hybrid (dense + sparse) support |
| Keyword search | Hand-rolled BM25Okapi | No `rank_bm25` dependency demonstrates the algorithm, not just a library call |
| Reranker | BAAI/bge-reranker-v2-m3 | Open-weight cross-encoder, meaningful precision gain over vector-only |
| LLM | Qwen3.5 0.8B via Ollama | Fully local generation, no API keys |
| Eval | RAGAS + a dependency-free tier-1 check | Currently not working - model issue |

No proprietary APIs anywhere in the pipeline.

## Key design decisions

- **Parent-child retrieval**: small clause-level chunks are what's embedded and searched (precision), but the LLM always receives the full parent Article (complete legal context) — never a lone fragment.
- **Contextual chunking**: every embedded chunk is prefixed with its Article/Part/Chapter heading before embedding. A bare clause like *"(2) The State shall not make any law which takes away..."* is close to meaningless in isolation; the prefix is what makes it findable.
- **Sequential-number clause splitting**: `(2)` is only treated as a real clause boundary if it's the Article's actual second clause — this avoids false splits on in-text citations like "under clause (2) of Article 8."
- **Hybrid retrieval + RRF fusion**: BM25 catches exact-term/Article-number queries that don't need to be *semantically* similar; dense search catches paraphrased questions. Reciprocal Rank Fusion combines both rankings without having to pick one.
- **Cross-reference expansion**: retrieval doesn't just return top-scoring chunks — it also pulls in Articles the top results explicitly reference, from the graph built in Section 2. Guaranteed complete legal context instead of hoping the embedding model surfaces it too.
- **Citation grounding + refuse-if-unsure**: enforced in the system prompt and checked automatically (both a cheap regex-based check and, when available, RAGAS's faithfulness metric) — not just asserted.

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ingest Constitution data
python -m src.ingestion.parse_pdf
python -m src.ingestion.resolve_crossref
python -m src.ingestion.chunk_articles

# 5. Generate embeddings and build the index
python -m src.ingestion.embed_and_index

# 6. Start FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# 7. In a separate terminal, activate the environment
venv\Scripts\activate

# 8. Start Streamlit frontend
streamlit run src/frontend/app.py



```

## Evaluation results


```bash
python run_eval.py --run-name bm25_only --bm25-only --no-ragas
python run_eval.py --run-name hybrid --no-ragas
python run_eval.py --run-name hybrid_reranking --rerank --no-ragas
```

| Run | Retrieval hit rate | Correct refusal rate | Citation faithfulness (basic) | RAGAS faithfulness | RAGAS answer relevancy |
|---|---|---|---|---|---|
| BM25-only | 0.4 | NA | 0.5 | Model issue | Model issue |
| Hybrid (BM25 + dense) | 0.6 | NA | 0.7 | Model issue | Model issue |
| Hybrid + reranking | 0.9 | NA | 0.9 | Model issue | Model issue |

## Repository structure

```
rag-pakistan-constitution/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── src/
│   ├── ingestion/
│   │   ├── parse_pdf.py          # PDF -> structured Articles (Part/Chapter/amendment history)
│   │   ├── resolve_crossrefs.py  # Article reference graph
│   │   ├── chunk_articles.py     # parent/child chunking
│   │   └── embed_and_index.py    # BGE-M3 embeddings -> Qdrant
│   ├── retrieval/
│   │   ├── hybrid_search.py      # BM25 + dense + RRF + cross-ref expansion
│   │   └── rerank.py             # cross-encoder reranking
│   ├── generation/
│   │   └── generate.py           # citation-grounded prompt + Ollama client + faithfulness check
│   └── api/
│       └── main.py               # FastAPI serving layer
├── eval/
│   ├── golden_dataset.json
│   ├── run_eval.py
│   └── results/                  # versioned eval runs, one file per configuration
└── data/
    ├── raw/                      # source PDF
    └── processed/                # articles.json, chunks, etc. (generated, not committed)
```

## Cost and staleness

- **No per-query API cost** — everything runs locally. The real cost is compute: embedding/reranking latency and whatever you pay to host Ollama + Qdrant if this were deployed rather than run locally.
- **Staleness**: this is a single, mostly-static legal document, but it isn't frozen — new amendments happen. The ingestion pipeline is fully idempotent (re-running `embed_and_index.py` upserts by stable chunk ID rather than duplicating), so refreshing the index after a new amendment Act is a matter of re-running the pipeline on an updated source PDF, not a special-cased migration.

## Known limitations

- The eval golden dataset currently has 8 question/answer pairs — a starter set covering each question category (direct lookup, negation/exception, numeric, amendment-aware, out-of-scope), not the 50-100 needed for statistically meaningful numbers. Expanding it is the highest-priority next step.
- Clause splitting handles top-level numbered clauses `(1)`, `(2)`, `(3)`; lettered sub-clauses `(a)`, `(b)` stay bundled within their parent clause rather than becoming separate chunks.
- Cross-reference resolution matches `Article N` mentions; it doesn't yet resolve references to Schedules, Parts, or bare "this Chapter" phrasing.
- Amendment-footnote parsing is calibrated against this specific PDF edition's formatting; a different source PDF would need the regex patterns in `parse_pdf.py` re-verified against its actual layout.
- No automated re-indexing trigger yet — refreshing after a real amendment is a manual pipeline re-run, not an automated watch/trigger.

## What I'd do with more time

- Expand the golden eval set to full coverage across all Articles, with CI running it on every change and failing the build if faithfulness or refusal-rate regresses
- Extend clause splitting to lettered sub-clauses for finer-grained retrieval on long, list-heavy Articles
- Extend cross-reference resolution to Schedules and Part-level references
- Add an automated re-indexing trigger keyed to source-document checksum changes
