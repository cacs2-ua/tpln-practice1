from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class TopKResult:
    query_index: int
    top_indices: List[int]
    top_scores: List[float]


def compute_similarity_matrix(query_vectors, paper_vectors) -> np.ndarray:
    """
    Returns a dense similarity matrix of shape (n_queries, n_papers).
    Works with sparse inputs too (sklearn handles it).
    """
    if getattr(query_vectors, "shape", None) is None or getattr(paper_vectors, "shape", None) is None:
        raise TypeError("query_vectors and paper_vectors must expose a .shape attribute")

    if query_vectors.shape[1] != paper_vectors.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: query_vectors has {query_vectors.shape[1]} features, "
            f"paper_vectors has {paper_vectors.shape[1]} features."
        )

    return cosine_similarity(query_vectors, paper_vectors)


def top_k_indices(scores: Sequence[float], k: int = 3) -> np.ndarray:
    """
    Deterministic top-k indices by:
      1) higher score first
      2) lower index first (tie-break)
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores is empty")

    k = min(k, s.size)
    idx = np.arange(s.size)
    order = np.lexsort((idx, -s))  # primary: -score asc => score desc, tie: idx asc
    return order[:k]


def top_k_for_all_queries(similarity_matrix: np.ndarray, k: int = 3) -> List[TopKResult]:
    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    results: List[TopKResult] = []
    for qi in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[qi]
        top_idx = top_k_indices(scores, k=k)
        results.append(
            TopKResult(
                query_index=qi,
                top_indices=top_idx.tolist(),
                top_scores=[float(scores[j]) for j in top_idx],
            )
        )
    return results


def validate_text_list(texts: Sequence[str], name: str) -> None:
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple of strings")
    if len(texts) == 0:
        raise ValueError(f"{name} is empty")
    if any(not isinstance(t, str) for t in texts):
        raise TypeError(f"{name} must contain only strings")
