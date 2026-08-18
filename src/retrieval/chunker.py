"""
src/retrieval/chunker.py
Loads documents from disk and splits them using RecursiveCharacterTextSplitter.
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
    docs = []
    doc_paths = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.pdf"))
    if not doc_paths:
        logger.warning(f"No documents found in {docs_dir}. Run scripts/ingest_docs.py first.")
        return docs
    for path in doc_paths:
        try:
            loader = PyPDFLoader(str(path)) if path.suffix == ".pdf" else TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path.name
            docs.extend(loaded)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def load_and_chunk(docs_dir: Path = DOCS_DIR) -> List[Document]:
    documents = load_documents(docs_dir)
    if not documents:
        return []
    return chunk_documents(documents)
