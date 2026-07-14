# PCB Design Research RAG

A RAG pipeline that takes a PCB/electronics design prompt, extracts and
**weights** the technical keywords in it, searches the web (biased toward
trustworthy electronics sources), reads each source **both as text and as
pixels** (screenshot tiling + a vision-language model), and synthesizes an
engineering research brief — all wrapped in a Streamlit UI.

**Backend:** Groq API for text generation, Hugging Face Inference API for
the vision model. Both serve open-weight models (e.g. Llama) rather than a
closed proprietary model — no OpenAI/Anthropic/etc. API is used. Web search
and embeddings stay fully local and key-less.

## Architecture

```
prompt
  │
  ▼
keyword_extractor.py   KeyBERT (local embeddings) + rarity/multiword weighting
  │                    "voltage" → low weight, "buried-zener reference" → high weight
  ▼
web_search.py          DuckDuckGo search (ddgs, no API key) per keyword,
  │                    re-ranked by keyword_weight + PRIORITY_DOMAINS bonus
  │                    (Wikipedia, TI, Analog Devices, IEEE, All About Circuits...)
  ▼
screenshot_capture.py  Playwright: full-page screenshot + plain text scrape
  │
  ├── text  ──────────► vector_store.py: chunk + embed (local sentence-transformers)
  │
  └── pixel_rag.py      tile the screenshot into overlapping crops,
       │                each tile read by an open-weight vision model
       │                (config.HF_VISION_MODEL, via Hugging Face Inference API)
       │                looking for schematics/tables/specs the query cares about
       ▼
       vector_store.py  captions become their own retrievable Documents
  │
  ▼
FAISS index (in-memory) → weighted_retrieve() blends vector similarity
  │                        with the originating keyword's weight
  ▼
rag_chain.py            ChatGroq (open-weight LLM, e.g. Llama 3.3 70B) writes
                         the final structured research brief, citing [S1], [S2]...
  │
  ▼
app.py (Streamlit)      shows weighted keywords, sources + screenshots,
                         retrieved chunks, and the final brief
```

## Why these choices (per your constraints)

- **Open-source models only, served via API instead of run locally.** Groq
  hosts open-weight LLMs (Llama 3.3, etc.) with very fast inference and a
  free tier. Hugging Face's Inference API hosts open-weight vision-language
  models. Neither is a closed proprietary model API.
- Web search uses `ddgs` (DuckDuckGo, key-less, no account needed). Text
  embeddings are a local, open-source sentence-transformer
  (`all-MiniLM-L6-v2`) — that one download is a one-time open-weights pull,
  not an API call, so there are no rate limits or costs on retrieval.
- **Keyword weighting.** `COMMON_ELECTRONICS_TERMS` in `config.py` is a
  discount list — generic words like "voltage", "power", "resistor" get
  their KeyBERT relevance score multiplied down. Multi-word technical
  phrases and terms outside that list get boosted. Tune the list/multipliers
  in `config.py` to taste.
- **Domain-weighted search.** `PRIORITY_DOMAINS` in `config.py` gives a
  score bonus to Wikipedia, TI, Analog Devices, Microchip, IEEE Xplore,
  All About Circuits, etc. Add/remove domains freely.
- **Pixel RAG.** Many datasheet specs, schematics, and comparison tables on
  electronics sites are images/canvases/PDF-embeds that plain HTML scraping
  misses. `pixel_rag.py` screenshots the whole page, cuts it into overlapping
  tiles, and asks a local vision model to transcribe anything relevant to
  the query in each tile. Those tile captions become first-class retrievable
  chunks alongside normal text chunks.

## Setup

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Playwright's headless browser (one-time)
playwright install chromium

# 3. API keys — both have free tiers, no local model install needed
cp .env.example .env
# then edit .env and fill in:
#   GROQ_API_KEY  -> https://console.groq.com/keys
#   HF_TOKEN      -> https://huggingface.co/settings/tokens

# 4. Run the app
streamlit run app.py
```

The Streamlit sidebar shows a ✅/❌ next to each key so you can immediately
see if one isn't loading (e.g. `.env` not found, or env var not exported).

## Notes / tuning

- **Bot-protected vendor sites (Access Denied).** Sites like Analog Devices,
  TI, etc. sometimes run Akamai-style bot detection and return an
  "Access Denied" page instead of content. `screenshot_capture.py` sends a
  realistic browser User-Agent/headers and a gentler load-wait strategy,
  detects blocked responses explicitly, and retries once — but it does not
  attempt to defeat deliberate bot protection (no CAPTCHA solving, no proxy
  rotation). Blocked sources are skipped and surfaced in a "blocked/
  unavailable" section in the UI instead of silently failing or crashing
  the run.
- **Pixel-RAG is the slowest, heaviest stage** (one vision-model API call
  per tile, per source, plus it costs free-tier quota). Toggle it off in
  the sidebar for a fast pure-text run, or lower `MAX_TILES_PER_PAGE` /
  `MAX_TOTAL_SOURCES` in `config.py`.
- Swap `GROQ_MODEL` in `config.py` for any model Groq currently serves
  (check `https://console.groq.com/docs/models` for the current list —
  it changes over time). Swap `HF_VISION_MODEL` similarly.
- **Gated vision models**: `Llama-3.2-11B-Vision-Instruct` requires
  accepting Meta's license on its Hugging Face model page before your
  token can call it. If you get a 403, either accept the license there or
  switch `HF_VISION_MODEL` to a non-gated alternative (see comments in
  `config.py`, e.g. `Qwen/Qwen2-VL-7B-Instruct`).
- `weighted_retrieve()` in `vector_store.py` blends 70% vector similarity /
  30% keyword weight — adjust that split if you want keyword weighting to
  dominate ranking more or less.
- This was built and syntax-checked in a sandboxed environment without live
  internet access, so the search/scrape/API-inference path hasn't been run
  end-to-end — test it in your own environment. The individual pieces
  (keyword weighting, ranking math, chunking, prompt structure) were
  verified locally. Expect the usual first-run rough edges: rate limits on
  free-tier keys, occasional site scrape timeouts, or a gated model needing
  its license accepted.

## File map

| File | Role |
|---|---|
| `config.py` | All tunable knobs in one place |
| `keyword_extractor.py` | Weighted keyword/keyphrase extraction |
| `web_search.py` | Key-less web search + domain-priority ranking |
| `screenshot_capture.py` | Playwright screenshot + text scrape |
| `pixel_rag.py` | Screenshot tiling + vision-model chunk reading |
| `vector_store.py` | FAISS index build + weighted retrieval |
| `rag_chain.py` | Orchestrates the full pipeline, LangChain LLM call |
| `app.py` | Streamlit UI |
