"""
Central configuration for the PCB-Design Research RAG pipeline.
Everything here is a knob you can tune from one place instead of hunting
through the codebase.
"""

# ---------------------------------------------------------------------------
# TEXT GENERATION -> Groq API (serves open-weight models: Llama, etc.)
# Get a free key at https://console.groq.com/keys and set it as the
# GROQ_API_KEY environment variable (or put it in a .env file).
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv()  # picks up a local .env file if present

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"   # swap for "llama-3.1-8b-instant" for a faster/cheaper run

# ---------------------------------------------------------------------------
# VISION (Pixel-RAG tile reading) -> Hugging Face Inference API
# Serves an open-weight vision-language model. Get a token at
# https://huggingface.co/settings/tokens and set HF_TOKEN (or .env).
# Note: some vision models are "gated" and require accepting their license
# on the model page before your token can call them. Non-gated alternatives
# are noted below if you hit a 403.
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# Alternatives if that one 403s for your token / isn't hosted on your provider:
#   "Qwen/Qwen2-VL-7B-Instruct"
#   "llava-hf/llava-v1.6-mistral-7b-hf"

# Text embeddings — open source, downloaded once from HuggingFace, run LOCALLY
# (not an API call — no rate limits, no key needed for this part).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# KEYWORD EXTRACTION & WEIGHTING
# ---------------------------------------------------------------------------
MAX_KEYWORDS = 12
KEYPHRASE_NGRAM_RANGE = (1, 4)          # allow up to 4-word technical phrases

# Generic electronics vocabulary -> gets its importance DISCOUNTED.
# Anything NOT in this list is treated as more specific/technical by default.
COMMON_ELECTRONICS_TERMS = {
    "voltage", "current", "power", "circuit", "resistor", "capacitor",
    "inductor", "diode", "transistor", "signal", "ground", "supply",
    "output", "input", "frequency", "design", "module", "board", "pcb",
    "component", "amplifier", "regulator", "battery", "switch", "load",
    "noise", "gain", "filter", "wire", "trace", "layer", "connector",
    "temperature", "resistance", "capacitance", "energy", "device",
    "system", "layout", "schematic",
}
COMMON_TERM_WEIGHT_MULTIPLIER = 0.35   # discount factor for common terms
MULTIWORD_BOOST = 1.35                 # multi-word technical phrases matter more
RARE_TERM_BOOST = 1.15                 # extra nudge for anything outside the common set

# ---------------------------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------------------------
RESULTS_PER_KEYWORD = 4
MAX_TOTAL_SOURCES = 18

# Domains we trust more for electronics/PCB engineering content.
# Matches get a score bonus so they rank above generic blogs/forums.
PRIORITY_DOMAINS = {
    "en.wikipedia.org": 0.6,
    "www.ti.com": 0.9,
    "ti.com": 0.9,
    "www.analog.com": 0.9,
    "analog.com": 0.9,
    "www.allaboutcircuits.com": 0.8,
    "www.electronics-tutorials.ws": 0.75,
    "www.electronicdesign.com": 0.7,
    "www.eeweb.com": 0.6,
    "www.eevblog.com": 0.5,
    "www.digikey.com": 0.7,
    "www.microchip.com": 0.85,
    "www.st.com": 0.8,
    "www.onsemi.com": 0.75,
    "ieeexplore.ieee.org": 0.85,
    "www.electropages.com": 0.5,
    "www.eepower.com": 0.6,
    "www.maximintegrated.com": 0.8,
    "www.renesas.com": 0.75,
}
DEFAULT_DOMAIN_BONUS = 0.0

# ---------------------------------------------------------------------------
# SCREENSHOT / "PIXEL RAG"
# ---------------------------------------------------------------------------
SCREENSHOT_DIR = "screenshots"
PAGE_LOAD_TIMEOUT_MS = 20000
TILE_SIZE = 900             # px, square tiles cut from the full-page screenshot
TILE_OVERLAP = 120          # px overlap between adjacent tiles
MAX_TILES_PER_PAGE = 8      # cap so one giant page doesn't dominate compute
ENABLE_PIXEL_RAG_DEFAULT = True

# ---------------------------------------------------------------------------
# VECTOR STORE / RETRIEVAL
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = "data/faiss_index"
TEXT_CHUNK_SIZE = 900
TEXT_CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 10
