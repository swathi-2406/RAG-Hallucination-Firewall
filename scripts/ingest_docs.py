"""
scripts/ingest_docs.py
Downloads AI/ML paper abstracts from arXiv and builds the FAISS index.
Run once before starting the app:
    python scripts/ingest_docs.py
"""

import sys
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arxiv
from tqdm import tqdm

from config.settings import DOCS_DIR, INDEX_DIR
from src.retrieval.chunker import load_and_chunk
from src.retrieval.retriever import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    for query, max_results in SEARCH_QUERIES:
        logger.info(f"Fetching: '{query}' (max {max_results})")
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            for paper in tqdm(search.results(), total=max_results, desc=f"  {query[:40]}"):
                content = (
                    f"Title: {paper.title}\n\n"
                    f"Authors: {', '.join(str(a) for a in paper.authors[:5])}\n\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d') if paper.published else 'N/A'}\n\n"
                    f"Abstract:\n{paper.summary}\n\n"
                    f"Categories: {', '.join(paper.categories)}\n"
                    f"ArXiv ID: {paper.entry_id}\n"
                )
                paper_id = paper.entry_id.split("/")[-1].replace(".", "_")
                fname = output_dir / f"{paper_id}.txt"
                if not fname.exists():
                    fname.write_text(content, encoding="utf-8")
                    total_saved += 1
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed: {e}")
    return total_saved


def main():
    logger.info("=" * 60)
    logger.info("RAG Hallucination Firewall — Document Ingestion")
    logger.info("=" * 60)

    existing = list(DOCS_DIR.glob("*.txt"))
    if existing:
        logger.info(f"Found {len(existing)} existing docs. Skipping download.")
    else:
        logger.info("Downloading arXiv abstracts...")
        n = download_arxiv_papers(DOCS_DIR)
        logger.info(f"Saved {n} papers to {DOCS_DIR}")

    logger.info("\nChunking documents...")
    chunks = load_and_chunk(DOCS_DIR)
    if not chunks:
        logger.error("No chunks created.")
        sys.exit(1)
    logger.info(f"Created {len(chunks)} chunks")

    logger.info("\nBuilding FAISS index...")
    start = time.perf_counter()
    build_index(chunks, save=True)
    logger.info(f"Index built in {time.perf_counter()-start:.1f}s")
    logger.info("\n✅ Done! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
