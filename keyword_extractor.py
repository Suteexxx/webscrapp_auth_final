"""
Extracts and WEIGHTS keywords/keyphrases from a PCB-design prompt.

Design choice (per your spec): common electronics vocabulary ("voltage",
"resistor", "power"...) should matter LESS than specific technical phrases
("zero-drift buffer amplifier", "buried-zener reference"). We do this with:

  1. KeyBERT (embedding-similarity based keyphrase extraction) to get
     candidate phrases + a base relevance score.
  2. A rarity/specificity multiplier: phrases built from generic terms are
     discounted; phrases containing terms OUTSIDE the common vocabulary,
     and multi-word technical phrases, are boosted.

No paid API is used — KeyBERT runs on a local open-source sentence-transformer.
"""

from typing import List, Tuple
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

import config


class KeywordExtractor:
    def __init__(self, embedding_model_name: str = config.EMBEDDING_MODEL):
        self._st_model = SentenceTransformer(embedding_model_name)
        self._kw_model = KeyBERT(model=self._st_model)

    def _specificity_multiplier(self, phrase: str) -> float:
        words = phrase.lower().split()
        common_hits = sum(1 for w in words if w in config.COMMON_ELECTRONICS_TERMS)
        common_ratio = common_hits / max(len(words), 1)

        multiplier = 1.0
        # Discount phrases dominated by generic terms
        if common_ratio >= 0.5:
            multiplier *= config.COMMON_TERM_WEIGHT_MULTIPLIER
        # Boost phrases that are mostly outside the common vocabulary
        if common_ratio == 0:
            multiplier *= config.RARE_TERM_BOOST
        # Multi-word technical phrases carry more specific engineering intent
        if len(words) >= 2:
            multiplier *= config.MULTIWORD_BOOST
        return multiplier

    def extract(self, prompt: str, top_n: int = config.MAX_KEYWORDS) -> List[Tuple[str, float]]:
        """Returns [(keyword, weight)] sorted by weight descending, weight in ~[0, 1.5+]."""
        candidates = self._kw_model.extract_keywords(
            prompt,
            keyphrase_ngram_range=config.KEYPHRASE_NGRAM_RANGE,
            stop_words="english",
            use_mmr=True,
            diversity=0.6,
            top_n=top_n * 2,  # over-generate, then re-rank with our weighting
        )

        weighted = []
        for phrase, base_score in candidates:
            mult = self._specificity_multiplier(phrase)
            weighted.append((phrase, round(base_score * mult, 4)))

        weighted.sort(key=lambda x: x[1], reverse=True)
        # de-dup near-identical phrases (substrings of each other)
        deduped: List[Tuple[str, float]] = []
        for phrase, w in weighted:
            if not any(phrase in kept[0] or kept[0] in phrase for kept in deduped):
                deduped.append((phrase, w))
        return deduped[:top_n]


if __name__ == "__main__":
    sample = (
        "Design an ultra-low-power precision voltage reference module producing "
        "selectable outputs of 2.5 V, 5 V, and 10 V while consuming less than 500 uW. "
        "Use a buried-zener or low-power bandgap reference, zero-drift buffer amplifier, "
        "ultra-low-TCR resistor networks, and ultra-low-IQ LDO regulators. Include "
        "trimming capability, reverse-polarity protection, and estimate temperature "
        "drift, output noise, and long-term stability."
    )
    ke = KeywordExtractor()
    for kw, w in ke.extract(sample):
        print(f"{w:6.3f}  {kw}")
