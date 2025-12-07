from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise ImportError("section8_plots.py requires pandas. Install/enable pandas first.") from e

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Utilities (deterministic top-k)
# -----------------------------
def _top_k_indices(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Deterministic top-k indices by:
      1) higher score first
      2) lower index first (tie-break)
    """
    s = np.asarray(scores, dtype=float).ravel()
    idx = np.arange(s.size)
    order = np.lexsort((idx, -s))  # primary: -score, tie: idx
    return order[: min(k, s.size)]


def _safe_str(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def _get_paper_meta(papers_df: Optional["pd.DataFrame"], doc_id: int) -> Dict[str, object]:
    """
    Fetch title/venue/year/url/abstract_snippet from papers_df if available,
    else return empty placeholders.
    """
    if papers_df is None:
        return {
            "title": "",
            "venue": "",
            "year": "",
            "url": "",
            "abstract_snippet": "",
        }

    row = papers_df.iloc[int(doc_id)]
    title = _safe_str(row.get("title", ""))
    venue = _safe_str(row.get("venue", ""))
    year = row.get("year", "")
    url = _safe_str(row.get("url", ""))
    abstract = _safe_str(row.get("abstract", ""))

    snippet = abstract[:240] + ("..." if len(abstract) > 240 else "")
    return {
        "title": title,
        "venue": venue,
        "year": year,
        "url": url,
        "abstract_snippet": snippet,
    }


# -----------------------------
# Build results table (72 rows)
# -----------------------------
def build_results_table(
    *,
    bow_results: Dict,
    tfidf_results: Dict,
    query_texts: Sequence[str],
    query_labels: Sequence[str],
    paper_texts: Sequence[str],
    papers_df: Optional["pd.DataFrame"] = None,
    run_order_bow: Optional[List[str]] = None,
    run_order_tfidf: Optional[List[str]] = None,
    k: int = 3,
) -> "pd.DataFrame":
    """
    Recomputes and returns a single tidy DataFrame with top-k results for:
      - BoW (4 runs)
      - TF-IDF (4 runs)
    Total rows: 2 * 4 * 3 * k = 72 (when k=3 and 3 queries).
    """
    if len(query_texts) != 3 or len(query_labels) != 3:
        raise ValueError("Expected exactly 3 queries (query_texts/query_labels length must be 3).")
    if len(paper_texts) == 0:
        raise ValueError("paper_texts is empty.")

    if run_order_bow is None:
        run_order_bow = ["bow_lc_on_sw_off", "bow_lc_on_sw_on", "bow_lc_off_sw_off", "bow_lc_off_sw_on"]
    if run_order_tfidf is None:
        run_order_tfidf = ["tfidf_lc_on_sw_off", "tfidf_lc_on_sw_on", "tfidf_lc_off_sw_off", "tfidf_lc_off_sw_on"]

    rows: List[Dict[str, object]] = []

    def _collect_for_family(repr_name: str, results_dict: Dict, run_order: List[str]) -> None:
        for run_name in run_order:
            if run_name not in results_dict:
                raise KeyError(f"Run '{run_name}' not found in {repr_name} results dict.")

            out = results_dict[run_name]
            cfg = out["config"]
            vectorizer = out["vectorizer"]
            X_docs = out["X"]

            X_q = vectorizer.transform(list(query_texts))
            S = cosine_similarity(X_q, X_docs)  # shape: (3, N_docs)

            for qi in range(3):
                top_idx = _top_k_indices(S[qi], k=k)
                for rank_i, doc_id in enumerate(top_idx, start=1):
                    score = float(S[qi, doc_id])
                    meta = _get_paper_meta(papers_df, int(doc_id))

                    rows.append(
                        {
                            "repr": repr_name,
                            "run": run_name,
                            "lowercase": bool(getattr(cfg, "lowercase", False)),
                            "stop_words": getattr(cfg, "stop_words", None),
                            "query": query_labels[qi],
                            "rank": int(rank_i),
                            "doc_id": int(doc_id),
                            "score": score,
                            **meta,
                        }
                    )

    _collect_for_family("BoW", bow_results, run_order_bow)
    _collect_for_family("TF-IDF", tfidf_results, run_order_tfidf)

    df = pd.DataFrame(rows)

    # stable sorting for nicer plots
    df["repr"] = pd.Categorical(df["repr"], categories=["BoW", "TF-IDF"], ordered=True)
    df["rank"] = df["rank"].astype(int)
    df["doc_id"] = df["doc_id"].astype(int)
    df["score"] = df["score"].astype(float)

    return df.sort_values(["repr", "run", "query", "rank"]).reset_index(drop=True)


# -----------------------------
# Plot A: Score separation
# -----------------------------
def plot_score_separation(
    results_df: "pd.DataFrame",
    *,
    query_label: str,
    run_order: Optional[List[str]] = None,
    repr_filter: Optional[str] = None,  # None => both, else "BoW" or "TF-IDF"
) -> None:
    """
    Line plot of top-1/top-2/top-3 scores per run for a given query.
    """
    df = results_df.copy()
    df = df[df["query"] == query_label]

    if repr_filter is not None:
        df = df[df["repr"] == repr_filter]

    if run_order is None:
        # default: whatever appears, in the DataFrame order
        run_order = list(dict.fromkeys(df["run"].tolist()))

    # pivot: index=run, columns=rank, values=score
    pivot = (
        df.pivot_table(index="run", columns="rank", values="score", aggfunc="first")
        .reindex(run_order)
    )

    x = np.arange(len(pivot.index))
    plt.figure(figsize=(11, 4.2))
    for r in [1, 2, 3]:
        if r in pivot.columns:
            plt.plot(x, pivot[r].values, marker="o", label=f"rank {r}")

    title = f"Score separation (top-3) — {query_label}"
    if repr_filter:
        title += f" — {repr_filter}"
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel("Cosine similarity")
    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot B: Jaccard overlap heatmap
# -----------------------------
def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def plot_jaccard_heatmap(
    results_df: "pd.DataFrame",
    *,
    query_label: str,
    run_order: Optional[List[str]] = None,
    repr_filter: Optional[str] = None,  # None => both, else "BoW" or "TF-IDF"
    k: int = 3,
) -> None:
    """
    Heatmap of Jaccard overlap between runs, based on the set of top-k doc_ids.
    """
    df = results_df.copy()
    df = df[df["query"] == query_label]
    df = df[df["rank"] <= k]

    if repr_filter is not None:
        df = df[df["repr"] == repr_filter]

    if run_order is None:
        run_order = list(dict.fromkeys(df["run"].tolist()))

    top_sets: Dict[str, set] = {}
    for run in run_order:
        s = set(df[df["run"] == run]["doc_id"].tolist())
        top_sets[run] = s

    M = np.zeros((len(run_order), len(run_order)), dtype=float)
    for i, ri in enumerate(run_order):
        for j, rj in enumerate(run_order):
            M[i, j] = _jaccard(top_sets[ri], top_sets[rj])

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(M)
    plt.colorbar()
    plt.title(f"Top-{k} Jaccard overlap (doc_id sets) — {query_label}" + (f" — {repr_filter}" if repr_filter else ""))
    plt.xticks(np.arange(len(run_order)), run_order, rotation=45, ha="right")
    plt.yticks(np.arange(len(run_order)), run_order)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Optional: BoW vs TF-IDF diversity summary
# -----------------------------
def plot_diversity_bow_vs_tfidf(
    results_df: "pd.DataFrame",
    *,
    query_label: str,
    k: int = 3,
) -> None:
    """
    Bar chart: number of unique doc_ids in top-k across runs, grouped by repr (BoW vs TF-IDF).
    """
    df = results_df.copy()
    df = df[(df["query"] == query_label) & (df["rank"] <= k)]

    counts = (
        df.groupby("repr")["doc_id"]
        .nunique()
        .reindex(["BoW", "TF-IDF"])
    )

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Diversity of retrieved doc_ids (unique in top-{k}) — {query_label}")
    plt.xlabel("Representation")
    plt.ylabel("Unique doc_ids across runs")
    plt.tight_layout()
    plt.show()
