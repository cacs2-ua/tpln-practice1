from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import sys
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
    """
    For each query (row) in similarity_matrix, return a TopKResult with:
      - query_index
      - top_indices (length k, or <=k if fewer papers)
      - top_scores  (aligned with top_indices)

    Includes extra debug prints ONLY if something crashes inside this function.
    """
    if getattr(similarity_matrix, "ndim", None) is None:
        raise TypeError("similarity_matrix must be array-like and expose .ndim/.shape")

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    n_queries, n_papers = similarity_matrix.shape
    results: List[TopKResult] = []

    for qi in range(n_queries):
        try:
            # Force 1D numeric row even if input is np.matrix or weird array-like
            scores = np.asarray(similarity_matrix[qi], dtype=float).ravel()

            if scores.size != n_papers:
                raise ValueError(
                    f"Row size mismatch at query {qi}: got {scores.size} scores, expected {n_papers}"
                )

            top_idx = top_k_indices(scores, k=k)
            top_scores = [float(scores[j]) for j in top_idx]

            results.append(
                TopKResult(
                    query_index=int(qi),
                    top_indices=[int(x) for x in top_idx.tolist()],
                    top_scores=top_scores,
                )
            )

        except Exception as e:
            # ---- DEBUG (only when something actually fails) ----
            print("\n[section7_similarity] ERROR inside top_k_for_all_queries()", file=sys.stderr)
            print(f"  - Exception: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  - similarity_matrix type : {type(similarity_matrix)}", file=sys.stderr)
            print(f"  - similarity_matrix shape: {getattr(similarity_matrix, 'shape', None)}", file=sys.stderr)
            print(f"  - similarity_matrix dtype: {getattr(similarity_matrix, 'dtype', None)}", file=sys.stderr)
            print(f"  - qi (row index)         : {qi}", file=sys.stderr)
            try:
                sample = scores[:10].tolist()  # may fail if scores wasn't created
                print(f"  - scores sample (first 10): {sample}", file=sys.stderr)
            except Exception:
                print("  - scores sample (first 10): <unavailable>", file=sys.stderr)
            print("------------------------------------------------------------\n", file=sys.stderr)
            # ---------------------------------------------
            raise

    return results


def validate_text_list(texts: Sequence[str], name: str) -> None:
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple of strings")
    if len(texts) == 0:
        raise ValueError(f"{name} is empty")
    if any(not isinstance(t, str) for t in texts):
        raise TypeError(f"{name} must contain only strings")
