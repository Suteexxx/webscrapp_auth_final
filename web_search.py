"""
Free, key-less web search using the `ddgs` (DuckDuckGo Search) library, with
results re-ranked using:
    final_score = keyword_weight + domain_priority_bonus

This is where "top research websites (Wikipedia, TI, Analog Devices,
All About Circuits, IEEE...) should rank higher" gets implemented.
"""

from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse
from ddgs import DDGS

import config


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    keyword: str
    keyword_weight: float
    domain_bonus: float = 0.0
    score: float = 0.0

    def __post_init__(self):
        domain = urlparse(self.url).netloc
        self.domain_bonus = config.PRIORITY_DOMAINS.get(domain, config.DEFAULT_DOMAIN_BONUS)
        self.score = round(self.keyword_weight + self.domain_bonus, 4)


def search_keyword(keyword: str, weight: float, max_results: int = config.RESULTS_PER_KEYWORD) -> List[SearchResult]:
    results: List[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(keyword, max_results=max_results):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href") or r.get("link", ""),
                        snippet=r.get("body", ""),
                        keyword=keyword,
                        keyword_weight=weight,
                    )
                )
    except Exception as e:
        print(f"[web_search] search failed for '{keyword}': {e}")
    return results


def gather_sources(weighted_keywords: List[tuple], max_total: int = config.MAX_TOTAL_SOURCES) -> List[SearchResult]:
    """Run a search per weighted keyword, merge, dedupe by URL, sort by score."""
    all_results: List[SearchResult] = []
    seen_urls = set()

    for keyword, weight in weighted_keywords:
        for res in search_keyword(keyword, weight):
            if res.url and res.url not in seen_urls:
                seen_urls.add(res.url)
                all_results.append(res)

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:max_total]


if __name__ == "__main__":
    demo = [("zero-drift buffer amplifier", 1.2), ("voltage reference", 0.4)]
    for r in gather_sources(demo):
        print(f"{r.score:5.2f}  {r.url}")
