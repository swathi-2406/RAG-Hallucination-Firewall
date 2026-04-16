"""
tests/test_retrieval.py
Unit tests for the retrieval pipeline components.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from langchain_core.documents import Document


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_documents_basic(self):
        from src.retrieval.chunker import chunk_documents
        docs = [
            Document(
                page_content="Transformers use self-attention. " * 30,
                metadata={"source": "test.txt"},
            )
        ]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1, "Long document should be split into multiple chunks"

    def test_chunk_metadata_preserved(self):
        from src.retrieval.chunker import chunk_documents
        docs = [
            Document(
                page_content="Short text about RAG. " * 10,
                metadata={"source": "rag_paper.txt"},
            )
        ]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert chunk.metadata["source"] == "rag_paper.txt"

    def test_chunk_overlap(self):
        from src.retrieval.chunker import chunk_documents
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        # Chunk size should be respected approximately
        docs = [Document(page_content="word " * 500, metadata={"source": "t.txt"})]
        chunks = chunk_documents(docs)
        for chunk in chunks[:-1]:  # All but last should be close to CHUNK_SIZE
            assert len(chunk.page_content) <= CHUNK_SIZE + 50

    def test_empty_document_list(self):
        from src.retrieval.chunker import chunk_documents
        chunks = chunk_documents([])
        assert chunks == []


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_embed_query_returns_vector(self):
        from src.retrieval.embedder import embed_query
        vec = embed_query("What is RAG?")
        assert isinstance(vec, list)
        assert len(vec) == 384  # all-MiniLM-L6-v2 dimension

    def test_embed_texts_batch(self):
        from src.retrieval.embedder import embed_texts
        texts = ["Attention is all you need.", "BERT uses masked language modeling."]
        vecs = embed_texts(texts)
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_embeddings_normalized(self):
        from src.retrieval.embedder import embed_query
        vec = np.array(embed_query("Semantic search"))
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, "Embeddings should be unit-normalized"

    def test_similar_texts_high_similarity(self):
        from src.retrieval.embedder import embed_texts
        vecs = embed_texts([
            "Retrieval augmented generation",
            "RAG combines retrieval with generation",
            "Completely unrelated: the cat sat on the mat",
        ])
        v = np.array(vecs)
        sim_related = float(np.dot(v[0], v[1]))
        sim_unrelated = float(np.dot(v[0], v[2]))
        assert sim_related > sim_unrelated, "Related texts should have higher similarity"

    def test_singleton_pattern(self):
        from src.retrieval.embedder import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2, "Embedder should be a singleton"


# ── Retriever tests (no FAISS index required) ─────────────────────────────────

class TestRetriever:
    def test_get_context_string_formatting(self):
        from src.retrieval.retriever import get_context_string
        chunks = [
            Document(page_content="Chunk about attention.", metadata={"source": "paper1.txt"}),
            Document(page_content="Chunk about BERT.", metadata={"source": "paper2.txt"}),
        ]
        ctx = get_context_string(chunks)
        assert "Chunk 1" in ctx
        assert "Chunk 2" in ctx
        assert "paper1.txt" in ctx
        assert "paper2.txt" in ctx

    def test_get_context_string_separator(self):
        from src.retrieval.retriever import get_context_string
        chunks = [
            Document(page_content="A", metadata={"source": "a.txt"}),
            Document(page_content="B", metadata={"source": "b.txt"}),
        ]
        ctx = get_context_string(chunks)
        assert "---" in ctx
