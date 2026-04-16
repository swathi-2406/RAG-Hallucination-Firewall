"""
scripts/ingest_docs.py
Downloads AI/ML paper abstracts from arXiv (free, no API key needed)
and builds the FAISS vector index.

Run this once before starting the app:
    python scripts/ingest_docs.py

What it does:
  1. Downloads ~200 paper abstracts across key AI/ML topics
  2. Saves them as .txt files in data/sample_docs/
  3. Chunks + embeds them
  4. Builds and saves the FAISS index to data/faiss_index/
"""

import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import arxiv
from tqdm import tqdm

from config.settings import DOCS_DIR, INDEX_DIR
from src.retrieval.chunker import load_and_chunk
from src.retrieval.retriever import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Topics to fetch ───────────────────────────────────────────────────────────
SEARCH_QUERIES = [
    ("retrieval augmented generation RAG", 30),
    ("large language model hallucination", 30),
    ("transformer attention mechanism", 20),
    ("BERT GPT language model pretraining", 20),
    ("vector database embedding similarity search", 20),
    ("chain of thought prompting reasoning", 20),
    ("instruction tuning RLHF fine-tuning", 20),
    ("diffusion models image generation", 15),
    ("graph neural network knowledge graph", 15),
    ("federated learning privacy machine learning", 10),
]


def download_arxiv_papers(output_dir: Path) -> int:
    """
    Fetch paper abstracts from arXiv and save as .txt files.
    Returns total number of papers saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for query, max_results in SEARCH_QUERIES:
        logger.info(f"Fetching: '{query}' (max {max_results})")

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for paper in tqdm(search.results(), total=max_results, desc=f"  {query[:40]}"):
                # Build a clean document from title + abstract
                content = (
                    f"Title: {paper.title}\n\n"
                    f"Authors: {', '.join(str(a) for a in paper.authors[:5])}\n\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d') if paper.published else 'N/A'}\n\n"
                    f"Abstract:\n{paper.summary}\n\n"
                    f"Categories: {', '.join(paper.categories)}\n"
                    f"ArXiv ID: {paper.entry_id}\n"
                )

                # Safe filename from paper ID
                paper_id = paper.entry_id.split("/")[-1].replace(".", "_")
                fname = output_dir / f"{paper_id}.txt"

                if not fname.exists():
                    fname.write_text(content, encoding="utf-8")
                    total_saved += 1

            time.sleep(0.5)  # Be polite to arXiv API

        except Exception as e:
            logger.error(f"Failed to fetch '{query}': {e}")

    return total_saved


def main():
    logger.info("=" * 60)
    logger.info("RAG Hallucination Firewall — Document Ingestion")
    logger.info("=" * 60)

    # ── Step 1: Download papers ───────────────────────────────────────────────
    existing = list(DOCS_DIR.glob("*.txt"))
    if existing:
        logger.info(f"Found {len(existing)} existing docs. Skipping download.")
        logger.info("(Delete data/sample_docs/ to re-download)")
    else:
        logger.info("Downloading arXiv abstracts (this may take 1–2 minutes)...")
        n = download_arxiv_papers(DOCS_DIR)
        logger.info(f"Saved {n} papers to {DOCS_DIR}")

    # ── Step 2: Chunk documents ───────────────────────────────────────────────
    logger.info("\nChunking documents...")
    chunks = load_and_chunk(DOCS_DIR)

    if not chunks:
        logger.error("No chunks created. Check that docs exist in data/sample_docs/")
        sys.exit(1)

    logger.info(f"Created {len(chunks)} chunks from {len(list(DOCS_DIR.glob('*.txt')))} documents")

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    logger.info("\nBuilding FAISS index (embedding model downloads ~90MB on first run)...")
    start = time.perf_counter()
    build_index(chunks, save=True)
    elapsed = time.perf_counter() - start

    logger.info(f"FAISS index built and saved in {elapsed:.1f}s")
    logger.info(f"Index location: {INDEX_DIR / 'faiss_store'}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Ingestion complete! Run the app with:")
    logger.info("   streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
