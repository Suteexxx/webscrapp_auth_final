"""
Builds an in-memory FAISS vector index from two kinds of Documents:
  - "text"  chunks: normal scraped page text, split into overlapping windows
  - "pixel" chunks: captions produced by pixel_rag.py from screenshot tiles

Both are embedded with the same local, open-source sentence-transformer, so
they compete fairly in similarity search. Each document's `keyword_weight`
metadata nudges retrieval to prefer results that came from higher-weighted
(more specific) query keywords.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import config


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def make_text_documents(source_url: str, text: str, keyword_weight: float) -> List[Document]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.TEXT_CHUNK_SIZE, chunk_overlap=config.TEXT_CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": source_url, "type": "text", "keyword_weight": keyword_weight},
        )
        for chunk in chunks
    ]


def make_pixel_documents(pixel_chunks, keyword_weight: float) -> List[Document]:
    docs = []
    for pc in pixel_chunks:
        docs.append(
            Document(
                page_content=pc.caption,
                metadata={
                    "source": pc.source_url,
                    "type": "pixel",
                    "tile_index": pc.tile_index,
                    "tile_path": pc.tile_path,
                    "keyword_weight": keyword_weight,
                },
            )
        )
    return docs


def build_index(documents: List[Document]):
    if not documents:
        return None
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def weighted_retrieve(vs, query: str, k: int = config.TOP_K_RETRIEVAL):
    """Similarity search, then re-rank by blending vector similarity with keyword_weight."""
    if vs is None:
        return []
    results = vs.similarity_search_with_relevance_scores(query, k=k * 2)
    reranked = []
    for doc, sim_score in results:
        kw_weight = doc.metadata.get("keyword_weight", 0.5)
        blended = 0.7 * sim_score + 0.3 * min(kw_weight, 1.5) / 1.5
        reranked.append((doc, blended))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:k]
