"""
Section 9: Evaluation harness.

Runs the full pipeline against the golden Q&A dataset and computes two
tiers of signal:

  1. A dependency-free "cheap check" (retrieval hit rate against
     expected_articles, and the citation-faithfulness check from Section
     7) -- always available, needs nothing beyond what's already running.
  2. RAGAS metrics (faithfulness, answer_relevancy, context_precision,
     context_recall) -- the standard, defensible numbers, computed with
     the local Ollama model as judge and BGE-M3 as the embedding model so
     the eval stack stays open-source too. Optional: if ragas/langchain
     aren't installed, the harness still runs and reports tier 1.

Every run is saved to eval/results/<run_name>_<timestamp>.json. That
directory, with multiple named runs in it (baseline, with_reranking,
with_hybrid, ...)

Usage:
   python -m eval.eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> makes `src` importable

from src.retrieval.hybrid_search import HybridRetriever, EMBEDDING_MODEL_NAME
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generate import build_prompt, OllamaGenerator, check_citation_faithfulness


def run_pipeline_on_question(question, retriever, reranker, generator, parents_by_id, candidate_k, top_k):
    candidates = retriever.search(question, top_k=candidate_k)
    results = reranker.rerank(question, candidates, top_k=top_k) if reranker is not None else candidates[:top_k]
    system_prompt, user_prompt = build_prompt(question, results, parents_by_id)
    answer = generator.generate(system_prompt, user_prompt)
    contexts = [r["text"] for r in results]
    return answer, contexts, results


def run_predictions(golden, retriever, reranker, generator, parents_by_id, candidate_k=10, top_k=5):
    predictions = []
    for item in golden:
        answer, contexts, results = run_pipeline_on_question(
            item["question"], retriever, reranker, generator, parents_by_id, candidate_k, top_k
        )
        cheap_check = check_citation_faithfulness(answer, results)
        retrieved_articles = {r["article_number"] for r in results}
        expected_articles = set(item.get("expected_articles", []))
        # For out-of-scope questions (expected_articles == []), "hit" means
        # the model correctly found nothing to force-fit an answer to --
        # judged separately below via the refusal check, not retrieval overlap.
        retrieval_hit = expected_articles.issubset(retrieved_articles) if expected_articles else None
        predictions.append({
            "id": item["id"],
            "category": item.get("category"),
            "question": item["question"],
            "ground_truth": item.get("ground_truth"),
            "answer": answer,
            "contexts": contexts,
            "retrieved_articles": sorted(retrieved_articles),
            "expected_articles": sorted(expected_articles),
            "retrieval_hit": retrieval_hit,
            "cheap_faithfulness": cheap_check,
        })
    return predictions


# ---------------------------------------------------------------------------
# Tier 1: dependency-free summary
# ---------------------------------------------------------------------------

REFUSAL_PHRASE = "does not contain enough information"


def summarize_cheap_checks(predictions: list[dict]) -> dict:
    total = len(predictions) or 1

    scoped = [p for p in predictions if p["expected_articles"]]
    out_of_scope = [p for p in predictions if not p["expected_articles"]]

    retrieval_hit_rate = (
        sum(1 for p in scoped if p["retrieval_hit"]) / len(scoped) if scoped else None
    )
    # For out-of-scope questions: did the model actually refuse, or did it
    # hallucinate an answer anyway? This is the single check I flagged
    # back in Section 7 as the most important one to run for real.
    correct_refusal_rate = (
        sum(1 for p in out_of_scope if REFUSAL_PHRASE in p["answer"].lower()) / len(out_of_scope)
        if out_of_scope else None
    )

    return {
        "total_questions": len(predictions),
        "pct_with_citation": sum(1 for p in predictions if p["cheap_faithfulness"]["has_citation"]) / total,
        "pct_passing_basic_faithfulness": sum(1 for p in predictions if p["cheap_faithfulness"]["passes_basic_check"]) / total,
        "retrieval_hit_rate": retrieval_hit_rate,
        "correct_refusal_rate": correct_refusal_rate,
    }


# ---------------------------------------------------------------------------
# Tier 2: RAGAS metrics (optional)
# ---------------------------------------------------------------------------

def compute_ragas_metrics(predictions: list[dict]):
    """Returns RAGAS's standard metrics using the local Ollama model as
    judge and BGE-M3 for embeddings -- or None (with can explanation
    printed) if ragas/langchain aren't installed, so the harness still
    produces the tier-1 summary either way."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset
    except ImportError as e:
        print(f"ragas/langchain not available ({e}) -- skipping RAGAS metrics, tier-1 summary only")
        return None

    ragas_llm = LangchainLLMWrapper(ChatOllama(model="qwen3.5:0.8b"))
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))

    # context_recall needs a ground_truth; skip predictions that don't have one
    scored = [p for p in predictions if p.get("ground_truth")]
    dataset = Dataset.from_list([
        {"question": p["question"], "answer": p["answer"], "contexts": p["contexts"], "ground_truth": p["ground_truth"]}
        for p in scored
    ])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas().to_dict(orient="records")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    golden_path: str,
    run_name: str,
    chunks_path: str ,
    parents_path: str,
    collection: str  ,
    qdrant_path: str,
    qdrant_url: str ,
    bm25_only: bool ,
    ollama_model: str ,
    ollama_host: str ,
    candidate_k: int ,
    top_k: int,
    compute_ragas: bool ,
) -> Path:
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    with open(chunks_path, encoding="utf-8") as f:
        child_chunks = json.load(f)
    with open(parents_path, encoding="utf-8") as f:
        parent_chunks = json.load(f)
    parents_by_id = {p["parent_id"]: p for p in parent_chunks}

    qdrant_client = None
    embedding_model = None
    if not bm25_only:
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            qdrant_client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=qdrant_path)
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError as e:
            print(f"Dense search unavailable ({e}) -- BM25-only eval run")

    retriever = HybridRetriever(
        child_chunks, parent_chunks,
        qdrant_client=qdrant_client, collection=collection, embedding_model=embedding_model,
    )

    reranker = None
    try:
        reranker = CrossEncoderReranker()
    except ImportError as e:
        print(f"Reranker unavailable ({e}) -- skipping rerank stage in eval")

    generator = OllamaGenerator(model=ollama_model, host=ollama_host)

    predictions = run_predictions(golden, retriever, reranker, generator, parents_by_id, candidate_k, top_k)
    cheap_summary = summarize_cheap_checks(predictions)
    ragas_scores = compute_ragas_metrics(predictions) if compute_ragas else None

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{run_name}_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "timestamp": time.time(),
                "cheap_summary": cheap_summary,
                "ragas_scores": ragas_scores,
                "predictions": predictions,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n=== Eval run '{run_name}' ===")
    print(json.dumps(cheap_summary, indent=2))
    print(f"RAGAS scores: {'saved, see file' if ragas_scores else 'not computed'}")
    print(f"Full results -> {output_path}")
    return output_path


if __name__ == "__main__":

    run(
        golden_path=r"eval\golden_dataset.json",
        run_name="baseline",

        chunks_path=r"data\processed\child_chunks.json",
        parents_path=r"data\processed\child_chunks.json",

        collection="constitution_pk",

        qdrant_path="qdrant_local",
        qdrant_url=None,

        bm25_only=False,

        ollama_model="qwen3.5:0.8b",
        ollama_host="http://localhost:11434",

        candidate_k=10,
        top_k=5,

        compute_ragas=True,
    )