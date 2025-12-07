from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedDoc:
    rank: int
    doc_index: int
    score: float
    text: str


def top_k_indices(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Return top-k indices sorted by:
      1) score descending
      2) index ascending (tie-break for determinism)
    """
    s = np.asarray(scores).ravel()
    if s.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if k <= 0:
        raise ValueError("k must be >= 1")

    k_eff = int(min(k, s.size))
    # Fast candidate selection
    cand = np.argpartition(-s, k_eff - 1)[:k_eff]
    # Deterministic sorting: score desc, index asc
    order = np.lexsort((cand, -s[cand]))
    return cand[order]


def run_similarity_search(
    *,
    vectorizer: Any,
    paper_vectors: Any,
    paper_texts: Sequence[str],
    query_texts: Sequence[str],
    query_labels: Optional[Sequence[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Core requirement for Section 7:
      - Use vectorizer.transform on queries (NOT fit_transform)
      - cosine similarity(query_vectors, paper_vectors)
      - retrieve top-k docs per query, returning their paper_texts[index] + score
    """
    if query_labels is None:
        query_labels = [f"Query {i+1}" for i in range(len(query_texts))]

    if len(query_texts) != len(query_labels):
        raise ValueError("query_texts and query_labels must have the same length")

    if not hasattr(paper_vectors, "shape"):
        raise TypeError("paper_vectors must be a matrix with .shape")

    n_docs = int(paper_vectors.shape[0])
    if len(paper_texts) != n_docs:
        raise ValueError(f"len(paper_texts)={len(paper_texts)} must match paper_vectors rows={n_docs}")

    # IMPORTANT: transform (re-use learned vocabulary)
    query_vectors = vectorizer.transform(list(query_texts))

    if int(query_vectors.shape[1]) != int(paper_vectors.shape[1]):
        raise RuntimeError(
            f"Vector dimension mismatch: query_vectors has {query_vectors.shape[1]} cols "
            f"but paper_vectors has {paper_vectors.shape[1]} cols"
        )

    sim = cosine_similarity(query_vectors, paper_vectors)  # (n_queries, n_docs) dense
    results: Dict[str, List[RetrievedDoc]] = {}

    for qi, qlabel in enumerate(query_labels):
        idxs = top_k_indices(sim[qi], k=top_k)
        matches: List[RetrievedDoc] = []
        for rank, di in enumerate(idxs, start=1):
            d = int(di)
            matches.append(
                RetrievedDoc(
                    rank=rank,
                    doc_index=d,
                    score=float(sim[qi, d]),
                    text=str(paper_texts[d]),
                )
            )
        results[str(qlabel)] = matches

    return {
        "query_vectors": query_vectors,
        "similarity_matrix": sim,
        "matches": results,
    }


def run_similarity_for_grid(
    *,
    grid_results: Mapping[str, Mapping[str, Any]],
    run_order: Sequence[str],
    representation_name: str,
    paper_texts: Sequence[str],
    query_texts: Sequence[str],
    query_labels: Sequence[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Runs similarity search for each run inside bow_results / tfidf_results.
    Expects each grid_results[run_name] to contain keys: 'vectorizer', 'X', 'config' (optional).
    """
    out: Dict[str, Any] = {"representation": representation_name, "runs": {}}

    for run_name in run_order:
        if run_name not in grid_results:
            raise KeyError(f"run_name '{run_name}' not found in grid_results")

        run_obj = grid_results[run_name]
        vect = run_obj["vectorizer"]
        X = run_obj["X"]
        cfg = run_obj.get("config", None)

        res = run_similarity_search(
            vectorizer=vect,
            paper_vectors=X,
            paper_texts=paper_texts,
            query_texts=query_texts,
            query_labels=query_labels,
            top_k=top_k,
        )

        out["runs"][run_name] = {
            "config": cfg,
            "matches": res["matches"],
        }

    return out


def _truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return text
    t = str(text)
    return t if len(t) <= max_chars else (t[:max_chars] + "...")


def flatten_results(
    *,
    bow_block: Dict[str, Any],
    tfidf_block: Dict[str, Any],
    max_chars: Optional[int] = 350,
) -> List[Dict[str, Any]]:
    """
    Flatten nested results into rows (nice for pandas display).
    """
    rows: List[Dict[str, Any]] = []

    for block in (bow_block, tfidf_block):
        rep = block["representation"]
        for run_name, run_data in block["runs"].items():
            cfg = run_data.get("config", None)
            lowercase = getattr(cfg, "lowercase", None)
            stop_words = getattr(cfg, "stop_words", None)

            matches_by_query = run_data["matches"]
            for qlabel, matches in matches_by_query.items():
                for m in matches:
                    rows.append(
                        {
                            "representation": rep,
                            "run": run_name,
                            "lowercase": lowercase,
                            "stop_words": stop_words,
                            "query": qlabel,
                            "rank": m.rank,
                            "score": m.score,
                            "doc_index": m.doc_index,
                            "paper_text": _truncate(m.text, max_chars=max_chars),
                        }
                    )

    return rows


def results_to_dataframe(rows: List[Dict[str, Any]]):
    """
    Optional: convert rows to a pandas DataFrame for clean comparison in Colab.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    return pd.DataFrame(rows)


def print_results(
    *,
    bow_block: Dict[str, Any],
    tfidf_block: Dict[str, Any],
    max_chars: Optional[int] = 350,
) -> None:
    """
    Console-style report, organized to compare BoW vs TF-IDF and preprocessing toggles.
    """
    for block in (bow_block, tfidf_block):
        rep = block["representation"]
        print(f"\n==================== {rep} ====================")
        for run_name, run_data in block["runs"].items():
            cfg = run_data.get("config", None)
            lc = getattr(cfg, "lowercase", None)
            sw = getattr(cfg, "stop_words", None)

            print(f"\n--- {run_name} (lowercase={lc}, stop_words={sw}) ---")
            matches_by_query = run_data["matches"]
            for qlabel, matches in matches_by_query.items():
                print(f"\n{qlabel}:")
                for m in matches:
                    txt = _truncate(m.text, max_chars=max_chars)
                    print(f"{m.rank}. Text: '{txt}' (Score: {m.score:.4f})")
