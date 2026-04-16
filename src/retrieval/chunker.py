"""
src/retrieval/chunker.py
Loads documents from disk and splits them into chunks using
LangChain's RecursiveCharacterTextSplitter.
"""

import logging
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR

logger = logging.getLogger(__name__)


def load_documents(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Load all .txt and .pdf files from the documents directory."""
    docs: List[Document] = []
    doc_paths = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.pdf"))

    if not doc_paths:
        logger.warning(f"No documents found in {docs_dir}. Run scripts/ingest_docs.py first.")
        return docs

    for path in doc_paths:
        try:
            if path.suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            # Attach source metadata
            for doc in loaded:
                doc.metadata["source"] = path.name
            docs.extend(loaded)
            logger.info(f"Loaded: {path.name} ({len(loaded)} page(s))")
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Uses a hierarchy of separators: paragraph → sentence → word → character.
    This preserves semantic coherence better than fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,   # Adds 'start_index' to metadata for traceability
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def load_and_chunk(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Convenience function: load + chunk in one call."""
    documents = load_documents(docs_dir)
    if not documents:
        return []
    return chunk_documents(documents)
