from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False


@dataclass(frozen=True)
class PrepConfig:
    title_key: str = "title"
    abstract_key: str = "abstract"
    joiner: str = " "
    make_dataframe: bool = True


def _as_text(x: Any) -> str:
    """Minimal, non-aggressive coercion: keep text as-is (no casing/punct stripping)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def build_paper_texts(
    records: Sequence[Dict[str, Any]],
    config: PrepConfig = PrepConfig(),
) -> Tuple[List[str], Optional["pd.DataFrame"]]:
    """
    Build the working 'text' field exactly as required:
        text = title + " " + abstract

    Document unit: one paper = one document.
    Returns:
      - paper_texts: list[str] with one concatenated document per paper
      - df (optional): pandas DataFrame including title/abstract/url/venue/year/text
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(f"records must be a non-string Sequence of dicts, got {type(records)}")

    # Enforce assignment requirement strictly
    if config.joiner != " ":
        raise ValueError("PrepConfig.joiner must be exactly one space (' ') to satisfy the spec.")

    paper_texts: List[str] = []
    rows_for_df: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"Each record must be a dict. Found {type(rec)} at index {i}.")

        title = _as_text(rec.get(config.title_key, ""))
        abstract = _as_text(rec.get(config.abstract_key, ""))

        # Exact construction rule: title + " " + abstract
        text = title + config.joiner + abstract
        paper_texts.append(text)

        if config.make_dataframe:
            rows_for_df.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "url": rec.get("url", ""),
                    "venue": rec.get("venue", ""),
                    "year": rec.get("year", ""),
                    "text": text,
                }
            )

    df_out = None
    if config.make_dataframe:
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for make_dataframe=True, but it is not available.")
        df_out = pd.DataFrame(rows_for_df)

    return paper_texts, df_out


def validate_paper_texts(paper_texts: Sequence[str], expected_n: int) -> None:
    """Sanity checks for Section 3 outputs."""
    if not isinstance(expected_n, int) or expected_n < 0:
        raise AssertionError(f"expected_n must be a non-negative int, got {expected_n!r}")

    if len(paper_texts) != expected_n:
        raise AssertionError(f"len(paper_texts)={len(paper_texts)} != expected_n={expected_n}")

    if any(not isinstance(t, str) for t in paper_texts):
        raise AssertionError("paper_texts must contain only strings.")

    # Stronger: reject empty or whitespace-only documents (e.g., '' or '   ')
    if any(len(t.strip()) == 0 for t in paper_texts):
        raise AssertionError("paper_texts contains empty/whitespace-only documents (unexpected for this dataset).")
