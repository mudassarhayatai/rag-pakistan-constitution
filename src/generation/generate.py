"""
Section 7: Generation -- prompt assembly, citation grounding, refuse-if-unsure.

Takes the reranked/retrieved context (Section 6's output) and assembles a
strict, citation-grounded prompt. Two behaviors matter more here , because the domain is legal text where a confidently
wrong answer is actively misleading, not just unhelpful:

  1. Every factual claim must cite a specific Article number from the
     provided context -- never a bare, uncited assertion.
  2. If the retrieved context doesn't actually answer the question, the
     model must say so explicitly rather than filling the gap from
     outside/parametric knowledge about the Constitution.

Uses Qwen3.5:0.8B (open weights) served locally via Ollama -- no API keys,
consistent with the rest of this project's open-source-only constraint.


"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root -> makes `src` importable


SYSTEM_PROMPT = """You are a legal research assistant answering questions about the Constitution of the Islamic Republic of Pakistan.

Rules you must follow:
1. Answer ONLY using the provided context below. Do not use any outside knowledge about the Constitution or Pakistani law, even if you believe you know the answer.
2. Every factual claim in your answer must cite the specific Article number it comes from, e.g. "(Article 25)".
3. If an Article in the context has noted amendment history and the question relates to whether/how a provision changed, mention the amendment explicitly.
4. Be precise and concise. Do not editorialize or offer opinions on the law's merits.
"""


def format_context(results: list[dict], parents_by_id: dict[str, dict]) -> str:
    """Turn retrieval results into the numbered, citable context block the
    LLM sees. Cross-reference-expanded Articles are labeled distinctly so
    the model (and a human reviewer) can tell primary hits apart from
    context pulled in for completeness. Amendment history is looked up
    from parent_chunks here rather than requiring earlier sections to
    carry it through their result dicts."""
    blocks = []
    for r in results:
        label = "Retrieved" if r["source"] == "retrieved" else r["source"]
        block = f"[Article {r['article_number']} -- {label}]\n{r['text']}"
        parent = parents_by_id.get(f"article_{r['article_number']}", {})
        history = parent.get("amendment_history")
        if history:
            block += f"\n(Amendment history: {'; '.join(history)})"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(query: str, results: list[dict], parents_by_id: dict[str, dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) ready to send to the LLM."""
    context = format_context(results, parents_by_id)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, citing Article numbers for every claim."
    )
    return SYSTEM_PROMPT, user_prompt


def extract_cited_articles(answer: str) -> list[str]:
    """Pull out every 'Article N' mention from the generated answer."""
    return sorted(set(re.findall(r"Article (\d{1,3}[A-Z]?)", answer)))


def check_citation_faithfulness(answer: str, results: list[dict]) -> dict:
    """Lightweight, free faithfulness check -- NOT a replacement for the
    RAGAS eval harness (that's a later section), but a fast signal to
    catch obviously ungrounded answers before a human even reviews them:
    any cited Article number that wasn't actually in the retrieved
    context is a strong hallucination signal."""
    cited = extract_cited_articles(answer)
    available = {r["article_number"] for r in results}
    ungrounded = [c for c in cited if c not in available]
    return {
        "cited_articles": cited,
        "ungrounded_citations": ungrounded,
        "has_citation": len(cited) > 0,
        "passes_basic_check": len(ungrounded) == 0 and len(cited) > 0,
    }

class OllamaGenerator:
    """Thin client for a locally-running Ollama server using stdlib urllib."""

    def __init__(
        self,
        model: str = "qwen3.5:0.8b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host.rstrip("/")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1
    ) -> str:

        import urllib.request
        import json
        import time

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,

            # Disable Qwen thinking/reasoning
            "think": False,

            "options": {
                "temperature": temperature,

                # Limit answer length
                "num_predict": 1000
            }
        }

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        print("Sending request to Ollama...", flush=True)
        print(f"Model: {self.model}", flush=True)
        print("Thinking: disabled", flush=True)
        print("Max output tokens: 600", flush=True)

        start_time = time.time()

        try:

            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:

                data = json.loads(
                    response.read().decode("utf-8")
                )

            elapsed = time.time() - start_time

            print(
                f"Ollama response received in "
                f"{elapsed:.2f} seconds",
                flush=True
            )

            return data["message"]["content"]

        except Exception as e:

            elapsed = time.time() - start_time

            print(
                f"Ollama request failed after "
                f"{elapsed:.2f} seconds",
                flush=True
            )

            print(
                f"Error type: {type(e).__name__}",
                flush=True
            )

            print(
                f"Error: {e}",
                flush=True
            )

            raise

if __name__ == "__main__":
    from src.retrieval.hybrid_search import HybridRetriever, EMBEDDING_MODEL_NAME
    from src.retrieval.reranker import CrossEncoderReranker

    # ------------------------------------------------------------------
    # HARD-CODED CONFIGURATION
    # ------------------------------------------------------------------
    chunks_path = r"data\processed\child_chunks.json"
    parents_path = r"data\processed\parent_chunks.json"

    query = "What does Article 25 say about equality of citizens?"
    
    collection = "constitution_pk"

    # Use local Qdrant
    qdrant_path = "qdrant_local"

    # Set to True if you want BM25 only
    bm25_only = False

    # Ollama configuration
    ollama_model = "qwen3.5:0.8b"
    ollama_host = "http://localhost:11434"

    


    # ------------------------------------------------------------------
    # LOAD CHUNKS
    # ------------------------------------------------------------------
    with open(chunks_path, encoding="utf-8") as f:
        child_chunks = json.load(f)

    with open(parents_path, encoding="utf-8") as f:
        parent_chunks = json.load(f)

    parents_by_id = {
        p["parent_id"]: p
        for p in parent_chunks
    }

    # ------------------------------------------------------------------
    # LOAD QDRANT + EMBEDDING MODEL
    # ------------------------------------------------------------------
    qdrant_client = None
    embedding_model = None
    try:
        if not bm25_only:
            try:
                from qdrant_client import QdrantClient
                from sentence_transformers import SentenceTransformer

                qdrant_client = QdrantClient(
                    path=qdrant_path
                )

                embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME
                )

            except ImportError as e:
                print(
                    f"qdrant-client/sentence-transformers not available "
                    f"({e}) -- retrieval falling back to BM25-only"
                )

        # ------------------------------------------------------------------
        # HYBRID RETRIEVAL
        # ------------------------------------------------------------------
        retriever = HybridRetriever(
            child_chunks,
            parent_chunks,
            qdrant_client=qdrant_client,
            collection=collection,
            embedding_model=embedding_model,
        )

        candidates = retriever.search(
            query,
            top_k=10
        )


        # ------------------------------------------------------------------
    # CROSS-ENCODER RERANKING
    # ------------------------------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 2: CROSS-ENCODER RERANKING")
        print("=" * 70, flush=True)

        try:

            print(
                "Loading CrossEncoder model: "
                f"{CrossEncoderReranker.MODEL_NAME}",
                flush=True
            )

            reranker = CrossEncoderReranker()

            print(
                f"CrossEncoder loaded successfully.",
                flush=True
            )

            print(
                f"Input candidates: {len(candidates)}",
                flush=True
            )

            results = reranker.rerank(
                query,
                candidates,
                top_k=5
            )

            print(
                f"Reranking completed successfully.",
                flush=True
            )

            print(
                f"Final results: {len(results)}",
                flush=True
            )

        except Exception as e:

            print("\n" + "=" * 70)
            print("RERANKER ERROR")
            print("=" * 70)

            print(
                f"Type: {type(e).__name__}"
            )

            print(
                f"Error: {e}"
            )

            raise

        # ------------------------------------------------------------------
        # BUILD PROMPT
        # ------------------------------------------------------------------
        system_prompt, user_prompt = build_prompt(
            query,
            results,
            parents_by_id
        )

        # ------------------------------------------------------------------
        # GENERATE ANSWER WITH OLLAMA
        # ------------------------------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 3: OLLAMA GENERATION")
        print("=" * 70)

        print(f"Host : {ollama_host}")
        print(f"Model: {ollama_model}")
        print("Sending request to Ollama...", flush=True)

        try:
            generator = OllamaGenerator(
                model=ollama_model,
                host=ollama_host
            )

            answer = generator.generate(
                system_prompt,
                user_prompt
            )

        except Exception as e:
            print(
                f"\nCould not reach Ollama at {ollama_host}: "
                f"{type(e).__name__}: {e}"
            )

            print("\n--- Prompt that would have been sent ---\n")
            print(system_prompt)
            print(user_prompt)
            raise
            # raise SystemExit(0)


   

        # ------------------------------------------------------------------
        # OUTPUT
        # ------------------------------------------------------------------
        print("=== Answer ===")
        print(answer)

        print("\n=== Basic faithfulness check ===")

        print(
            json.dumps(
                check_citation_faithfulness(
                    answer,
                    results
                ),
                indent=2
            )
        )

    finally:

        if qdrant_client is not None:
            try:
                qdrant_client.close()
                print("\nQdrant client closed.")
            except Exception as e:
                print(f"Qdrant close warning: {e}")
