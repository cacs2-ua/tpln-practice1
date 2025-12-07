
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class BoWConfig:
    """
    Configuration for a BoW (CountVectorizer) run.

    Required toggles for the assignment:
      - lowercase: ON vs OFF
      - stop_words: None vs 'english'

    NOTE (important for the unit tests):
      - sklearn default token_pattern drops 1-character tokens (e.g., "a", "b").
      - We set token_pattern to accept 1+ char tokens to avoid "empty vocabulary"
        on synthetic test cases.
    """
    name: str
    lowercase: bool = True
    stop_words: Optional[str] = None
    max_features: Optional[int] = None
    ngram_range: Tuple[int, int] = (1, 1)

    # Accept single-character tokens too (fixes Section 7-style synthetic tests)
    token_pattern: str = r"(?u)\b\w+\b"


def _validate_texts(texts: Sequence[str]) -> None:
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"texts must be a list/tuple of strings, got {type(texts)}")
    if len(texts) == 0:
        raise ValueError("texts is empty (N=0).")
    if any(not isinstance(t, str) for t in texts):
        bad = next(type(t) for t in texts if not isinstance(t, str))
        raise TypeError(f"All texts must be strings. Found non-string: {bad}")
    if any(len(t) == 0 for t in texts):
        raise ValueError("texts contains empty strings (unexpected for this dataset).")


def build_bow_vectors(
    texts: Sequence[str],
    config: BoWConfig,
):
    """
    Fit a CountVectorizer and return:
      - vectorizer (fitted)
      - X (document-term matrix, sparse)
    """
    _validate_texts(texts)

    vectorizer = CountVectorizer(
        lowercase=config.lowercase,
        stop_words=config.stop_words,
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        token_pattern=config.token_pattern,
    )

    try:
        X = vectorizer.fit_transform(texts)
    except Exception as e:
        # DEBUG ONLY ON FAILURE (so it doesn't spam normal runs/tests)
        print("\n[section4_bow] ERROR in build_bow_vectors()", file=sys.stderr)
        print(f"  - Exception: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"  - Config.name     : {config.name}", file=sys.stderr)
        print(f"  - lowercase       : {config.lowercase}", file=sys.stderr)
        print(f"  - stop_words      : {config.stop_words}", file=sys.stderr)
        print(f"  - max_features    : {config.max_features}", file=sys.stderr)
        print(f"  - ngram_range     : {config.ngram_range}", file=sys.stderr)
        print(f"  - token_pattern   : {config.token_pattern}", file=sys.stderr)
        print(f"  - N_texts         : {len(texts)}", file=sys.stderr)
        sample = list(texts[:2])
        print(f"  - sample texts[0:2]: {[s[:120] for s in sample]}", file=sys.stderr)
        print("------------------------------------------------------------\n", file=sys.stderr)
        raise

    return vectorizer, X


def sparse_matrix_stats(X) -> Dict[str, Any]:
    """Compute basic inspection stats for a sparse doc-term matrix."""
    n_docs, n_vocab = X.shape
    nnz = int(X.nnz)
    total = int(n_docs) * int(n_vocab)
    density = float(nnz / total) if total > 0 else 0.0
    avg_nnz_per_doc = float(nnz / n_docs) if n_docs > 0 else 0.0

    return {
        "n_docs": int(n_docs),
        "n_vocab": int(n_vocab),
        "shape": (int(n_docs), int(n_vocab)),
        "nnz": nnz,
        "density": density,
        "avg_nnz_per_doc": avg_nnz_per_doc,
        "matrix_type": type(X).__name__,
    }


def feature_examples(vectorizer: CountVectorizer, n: int = 20) -> List[str]:
    feats = vectorizer.get_feature_names_out()
    return feats[:n].tolist()


def case_sensitive_stopword_survivors(vectorizer: CountVectorizer) -> int:
    """
    Detect the nuance: with lowercase=False and stop_words='english',
    capitalized variants of stopwords (e.g., 'The') can survive.

    Returns how many vocabulary terms are "stopwords after lowercasing"
    but are not exactly in the stopword list (case mismatch).
    """
    stop = vectorizer.get_stop_words()
    if not stop:
        return 0

    feats = vectorizer.get_feature_names_out()
    stop_set = set(stop)

    survivors = 0
    for f in feats:
        fl = f.lower()
        if fl in stop_set and f not in stop_set:
            survivors += 1
    return survivors


def run_bow_grid(texts: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Run the 2x2 grid required by the assignment:
      - lowercase ON/OFF
      - stopwords OFF/ON ('english')
    Returns a dict keyed by config name with vectorizer, X, and stats.
    """
    configs = [
        BoWConfig(name="bow_lc_on_sw_off", lowercase=True,  stop_words=None),
        BoWConfig(name="bow_lc_on_sw_on",  lowercase=True,  stop_words="english"),
        BoWConfig(name="bow_lc_off_sw_off",lowercase=False, stop_words=None),
        BoWConfig(name="bow_lc_off_sw_on", lowercase=False, stop_words="english"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        vect, X = build_bow_vectors(texts, cfg)
        stats = sparse_matrix_stats(X)
        survivors = case_sensitive_stopword_survivors(vect) if cfg.stop_words else 0

        results[cfg.name] = {
            "config": cfg,
            "vectorizer": vect,
            "X": X,
            "stats": stats,
            "case_sensitive_stopword_survivors": survivors,
        }

    return results
