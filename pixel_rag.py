"""
"Pixel RAG": instead of relying only on scraped HTML text (which can miss
schematics, plots, tables rendered as images/canvas, equations, etc.), we:

  1. Take a full-page screenshot of the source (screenshot_capture.py).
  2. Slice it into overlapping square tiles.
  3. Feed each tile to a local, open-source VISION model (via Ollama, e.g.
     llava) and ask it to describe/transcribe anything in that tile relevant
     to the user's engineering query.
  4. Each tile's caption becomes its own retrievable chunk, tagged with the
     source URL and tile position, so the answer can still cite where it
     came from.

The vision step calls the Hugging Face Inference API (huggingface_hub's
InferenceClient), pointed at an open-weight vision-language model
(config.HF_VISION_MODEL). This still uses an "open source model", just
hosted rather than run locally -- no OpenAI/Anthropic/etc. API is used.
"""

import base64
import os
from dataclasses import dataclass
from typing import List

from huggingface_hub import InferenceClient
from PIL import Image

import config

_hf_client = InferenceClient(token=config.HF_TOKEN) if config.HF_TOKEN else None


@dataclass
class PixelChunk:
    source_url: str
    tile_index: int
    tile_path: str
    caption: str


def _tile_image(image_path: str, tile_size: int, overlap: int, max_tiles: int) -> List[str]:
    """Slice a (tall) full-page screenshot into overlapping square tiles, saved to disk."""
    img = Image.open(image_path)
    w, h = img.size
    stride = tile_size - overlap
    tiles = []

    y = 0
    idx = 0
    base, ext = os.path.splitext(image_path)
    while y < h and idx < max_tiles:
        x = 0
        while x < w and idx < max_tiles:
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = img.crop(box)
            if tile.size[0] > 40 and tile.size[1] > 40:  # skip slivers
                tile_path = f"{base}_tile{idx}{ext}"
                tile.save(tile_path)
                tiles.append(tile_path)
                idx += 1
            x += stride
        y += stride

    return tiles


def _describe_tile_with_vision_model(tile_path: str, query: str) -> str:
    """Send one image tile + the user's engineering query to the HF-hosted vision model."""
    if _hf_client is None:
        print("[pixel_rag] HF_TOKEN not set — skipping vision step for this tile.")
        return "NOT_RELEVANT"

    with open(tile_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_b64}"

    prompt = (
        "You are reading a cropped screenshot tile from a technical webpage about "
        "electronics/PCB design. Extract ONLY information relevant to this design "
        f"task: \"{query}\". If the tile contains a schematic, formula, table, graph, "
        "or numeric spec (e.g. ppm/°C, µA, dB, tolerances), transcribe those values "
        "precisely. If the tile is irrelevant (ads, navigation, unrelated text), reply "
        "with exactly: NOT_RELEVANT."
    )

    try:
        completion = _hf_client.chat_completion(
            model=config.HF_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[pixel_rag] HF vision model call failed for {tile_path}: {e}")
        return "NOT_RELEVANT"


def build_pixel_chunks(source_url: str, screenshot_path: str, query: str) -> List[PixelChunk]:
    if not screenshot_path or not os.path.exists(screenshot_path):
        return []

    tile_paths = _tile_image(
        screenshot_path,
        tile_size=config.TILE_SIZE,
        overlap=config.TILE_OVERLAP,
        max_tiles=config.MAX_TILES_PER_PAGE,
    )

    chunks = []
    for i, tile_path in enumerate(tile_paths):
        caption = _describe_tile_with_vision_model(tile_path, query)
        if caption and "NOT_RELEVANT" not in caption.upper():
            chunks.append(
                PixelChunk(source_url=source_url, tile_index=i, tile_path=tile_path, caption=caption)
            )
    return chunks


if __name__ == "__main__":
    from screenshot_capture import capture_page

    cap = capture_page("https://en.wikipedia.org/wiki/Voltage_reference")
    chunks = build_pixel_chunks(cap.url, cap.screenshot_path, "low power voltage reference design")
    for c in chunks:
        print(c.tile_index, c.caption[:120])
