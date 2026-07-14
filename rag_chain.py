"""
Top-level orchestrator that wires every stage together:

    prompt -> weighted keywords -> web search (domain-weighted)
           -> per source: screenshot + text scrape
           -> pixel-RAG tile captions (vision model) + text chunking
           -> FAISS index -> weighted retrieval
           -> local LLM (Ollama) writes the final engineering research brief

Everything is streamed back through a `progress_cb(stage: str)` callback so
the Streamlit UI can show live progress.
"""

from typing import Callable, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
from keyword_extractor import KeywordExtractor
from web_search import gather_sources, SearchResult
from screenshot_capture import capture_page
from pixel_rag import build_pixel_chunks
from vector_store import make_text_documents, make_pixel_documents, build_index, weighted_retrieve


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior PCB / analog design engineer writing an internal research
brief to help a colleague start a hardware design. Use ONLY the CONTEXT below
(a mix of scraped webpage text and vision-model readings of page screenshots,
including datasheet figures/tables). Cite sources inline like [S3] matching
the numbering given. If the context doesn't cover something, say so plainly
instead of inventing numbers.

DESIGN REQUEST:
{query}

CONTEXT:
{context}

Write a structured research brief covering, where the context supports it:
1. Key building blocks and recommended topology/architecture
2. Relevant component classes and notable parts/families mentioned in sources
3. Design considerations: precision, drift, noise, trimming, protection
4. Estimated performance envelope (temperature drift, noise, long-term stability) — clearly mark any estimate as an estimate
5. Open questions / what still needs a datasheet deep-dive

Keep it dense and engineering-oriented, not marketing fluff."""
)


def _format_context(retrieved) -> str:
    lines = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        tag = f"[S{i}]"
        src = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("type", "text")
        lines.append(f"{tag} ({kind}, {src}, relevance={score:.2f}):\n{doc.page_content}\n")
    return "\n".join(lines)


def run_pipeline(
    query: str,
    use_pixel_rag: bool = config.ENABLE_PIXEL_RAG_DEFAULT,
    max_sources: int = config.MAX_TOTAL_SOURCES,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    log("Extracting & weighting keywords...")
    extractor = KeywordExtractor()
    weighted_keywords = extractor.extract(query)

    log(f"Searching the web for {len(weighted_keywords)} weighted keywords...")
    sources: List[SearchResult] = gather_sources(weighted_keywords, max_total=max_sources)

    all_docs = []
    source_records = []

    for i, src in enumerate(sources, start=1):
        log(f"Fetching source {i}/{len(sources)}: {src.url}")
        capture = capture_page(src.url)

        if not capture.accessible:
            log(f"  -> blocked/unavailable ({capture.block_reason}) — skipping {src.url}")
            source_records.append(
                {
                    "url": src.url,
                    "title": src.title,
                    "keyword": src.keyword,
                    "score": src.score,
                    "screenshot_path": None,
                    "text_chunks": 0,
                    "pixel_chunks": 0,
                    "blocked": True,
                    "block_reason": capture.block_reason,
                }
            )
            continue

        text_docs = make_text_documents(src.url, capture.text, src.keyword_weight)
        all_docs.extend(text_docs)

        pixel_doc_count = 0
        if use_pixel_rag and capture.screenshot_path:
            log(f"  -> Pixel-RAG: tiling & reading screenshot for {src.url}")
            pixel_chunks = build_pixel_chunks(src.url, capture.screenshot_path, query)
            pixel_docs = make_pixel_documents(pixel_chunks, src.keyword_weight)
            all_docs.extend(pixel_docs)
            pixel_doc_count = len(pixel_docs)

        source_records.append(
            {
                "url": src.url,
                "title": src.title,
                "keyword": src.keyword,
                "score": src.score,
                "screenshot_path": capture.screenshot_path,
                "text_chunks": len(text_docs),
                "pixel_chunks": pixel_doc_count,
                "blocked": False,
                "block_reason": None,
            }
        )

    log("Building vector index (local embeddings)...")
    vs = build_index(all_docs)

    log("Retrieving most relevant chunks...")
    retrieved = weighted_retrieve(vs, query, k=config.TOP_K_RETRIEVAL) if vs else []

    log("Generating research brief with Groq LLM...")
    if not config.GROQ_API_KEY:
        return {
            "weighted_keywords": weighted_keywords,
            "sources": source_records,
            "retrieved": retrieved,
            "answer": (
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it as an environment "
                "variable (or in a .env file) before running the app."
            ),
        }

    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0.2)
    chain = ANSWER_PROMPT | llm
    context_str = _format_context(retrieved)
    answer = ""
    if context_str.strip():
        answer = chain.invoke({"query": query, "context": context_str}).content
    else:
        answer = "No usable context was retrieved (check network/search setup)."

    return {
        "weighted_keywords": weighted_keywords,
        "sources": source_records,
        "retrieved": retrieved,
        "answer": answer,
    }
