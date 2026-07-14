"""
Captures a full-page screenshot of each source URL (for Pixel-RAG) and, as a
cheap fallback/complement, also pulls the raw visible text (for normal
text-RAG). Uses Playwright (open source, local browser automation) -- no API.

Some vendor/electronics sites (Analog Devices, TI, etc.) run bot-protection
(commonly Akamai) that blocks obviously-automated requests with an
"Access Denied" page. This module:
  - sends a realistic browser User-Agent + headers so we look like a normal
    visitor rather than a bare automation client,
  - uses a gentler page-load wait strategy (avoids "networkidle", which some
    protections flag as non-human timing),
  - detects blocked/error responses explicitly and marks the source as
    inaccessible instead of silently returning garbage or crashing the run,
  - retries once with a fresh context before giving up.

This does NOT attempt to defeat deliberate bot-protection (no CAPTCHA
solving, no proxy rotation, no fingerprint spoofing beyond looking like an
ordinary browser). If a site actively blocks automated access, we skip it
and move on -- the pipeline should degrade gracefully, not fight the site.

Run once before first use:
    playwright install chromium
"""

import os
import re
import hashlib
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

import config

# A realistic, current desktop Chrome UA. Some sites block requests whose
# UA identifies them as headless/automated (e.g. "HeadlessChrome").
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Phrases that show up on block/denial pages -- used to detect a blocked
# fetch even when the HTTP status code itself looks fine.
BLOCK_INDICATORS = [
    "access denied",
    "permission to access",
    "request could not be satisfied",
    "reference #",
    "are you a robot",
    "captcha",
    "unusual traffic",
]


@dataclass
class PageCapture:
    url: str
    screenshot_path: Optional[str]
    text: str
    accessible: bool = True
    block_reason: Optional[str] = None


def _safe_filename(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:40]
    return f"{slug}_{h}.png"


def _looks_blocked(status: Optional[int], text: str) -> Optional[str]:
    if status is not None and status in (401, 403, 429, 451):
        return f"HTTP {status}"
    lowered = text.lower()[:2000]
    for phrase in BLOCK_INDICATORS:
        if phrase in lowered:
            return f"blocked-page detected ('{phrase}')"
    return None


def _attempt_capture(url: str, screenshot_path: str) -> PageCapture:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()

        response = page.goto(
            url, timeout=config.PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded"
        )
        # Give lazy-loaded content a brief moment without waiting for full
        # network idle (which reads as "non-human" to some bot detectors).
        page.wait_for_timeout(1500)

        status = response.status if response else None

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

        block_reason = _looks_blocked(status, text)

        if block_reason is None:
            page.screenshot(path=screenshot_path, full_page=True)

        browser.close()

        if block_reason:
            return PageCapture(url=url, screenshot_path=None, text="", accessible=False, block_reason=block_reason)
        return PageCapture(url=url, screenshot_path=screenshot_path, text=text, accessible=True)


def capture_page(url: str, out_dir: str = config.SCREENSHOT_DIR) -> PageCapture:
    os.makedirs(out_dir, exist_ok=True)
    screenshot_path = os.path.join(out_dir, _safe_filename(url))

    last_error = None
    for attempt in range(2):  # one retry with a fresh browser context
        try:
            result = _attempt_capture(url, screenshot_path)
            if result.accessible:
                return result
            last_error = result.block_reason
        except Exception as e:
            last_error = str(e)
            print(f"[screenshot_capture] attempt {attempt + 1} failed for {url}: {e}")

    print(f"[screenshot_capture] giving up on {url}: {last_error}")
    return PageCapture(url=url, screenshot_path=None, text="", accessible=False, block_reason=last_error)


if __name__ == "__main__":
    cap = capture_page("https://en.wikipedia.org/wiki/Voltage_reference")
    print(cap.accessible, cap.screenshot_path, len(cap.text))
