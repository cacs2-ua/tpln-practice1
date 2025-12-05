from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class QueryItem:
    label: str
    title: str
    abstract: str
    text: str


def _as_text(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def build_query_item(label: str, title: str, abstract: str, joiner: str = " ") -> QueryItem:
    """
    Query construction MUST match dataset documents (Section 3):
        text = title + " " + abstract
    No extra cleaning here.
    """
    if not isinstance(joiner, str) or joiner != " ":
        raise ValueError('joiner must be exactly one space: " "')

    t = _as_text(title)
    a = _as_text(abstract)

    if len(t.strip()) == 0:
        raise ValueError(f"{label}: title is empty")
    if len(a.strip()) == 0:
        raise ValueError(f"{label}: abstract is empty")

    text = t + joiner + a
    return QueryItem(label=label, title=t, abstract=a, text=text)


def get_default_queries(joiner: str = " ") -> List[QueryItem]:
    """
    The three queries EXACTLY as provided by the assignment (title + abstract).
    Stored in fixed order: Query 1, Query 2, Query 3.
    """
    q1_title = "QUALES: Machine translation quality estimation via supervised and unsupervised machine learning."
    q1_abstract = (
        "The automatic quality estimation (QE) of machine translation consists in measuring the quality of translations "
        "without access to human references, usually via machine learning approaches. A good QE system can help in three "
        "aspects of translation processes involving machine translation and post-editing: increasing productivity (by ruling "
        "out poor quality machine translation), estimating costs (by helping to forecast the cost of post-editing) and selecting "
        "a provider (if several machine translation systems are available). Interest in this research area has grown significantly "
        "in recent years, leading to regular shared tasks in the main machine translation conferences and intense scientific activity. "
        "In this article we review the state of the art in this research area and present project QUALES, which is under development."
    )

    q2_title = "Learning to Ask Unanswerable Questions for Machine Reading Comprehension"
    q2_abstract = (
        "Machine reading comprehension with unanswerable questions is a challenging task. In this work, we propose a data augmentation "
        "technique by automatically generating relevant unanswerable questions according to an answerable question paired with its corresponding "
        "paragraph that contains the answer. We introduce a pair-to-sequence model for unanswerable question generation, which effectively captures "
        "the interactions between the question and the paragraph. We also present a way to construct training data for our question generation models "
        "by leveraging the existing reading comprehension dataset. Experimental results show that the pair-to-sequence model performs consistently better "
        "compared with the sequence-to-sequence baseline. We further use the automatically generated unanswerable questions as a means of data augmentation "
        "on the SQuAD 2.0 dataset, yielding 1.9 absolute F1 improvement with BERT-base model and 1.7 absolute F1 improvement with BERT-large model."
    )

    q3_title = "Unsupervised Neural Text Simplification"
    q3_abstract = (
        "The paper presents a first attempt towards unsupervised neural text simplification that relies only on unlabelled text corpora. "
        "The core framework is composed of a shared encoder and a pair of attentional-decoders, crucially assisted by discrimination-based losses and denoising. "
        "The framework is trained using unlabelled text collected from en-Wikipedia dump. Our analysis (both quantitative and qualitative involving human evaluators) "
        "on public test data shows that the proposed model can perform text-simplification at both lexical and syntactic levels, competitive to existing supervised methods. "
        "It also outperforms viable unsupervised baselines. Adding a few labelled pairs helps improve the performance further."
    )

    queries = [
        build_query_item("Query 1", q1_title, q1_abstract, joiner=joiner),
        build_query_item("Query 2", q2_title, q2_abstract, joiner=joiner),
        build_query_item("Query 3", q3_title, q3_abstract, joiner=joiner),
    ]
    return queries


def get_query_texts(queries: Sequence[QueryItem]) -> List[str]:
    return [q.text for q in queries]


def validate_queries(queries: Sequence[QueryItem]) -> None:
    if not isinstance(queries, (list, tuple)):
        raise TypeError(f"queries must be a list/tuple, got {type(queries)}")
    if len(queries) != 3:
        raise AssertionError(f"Expected exactly 3 queries, got {len(queries)}")

    expected_labels = ["Query 1", "Query 2", "Query 3"]
    got_labels = [q.label for q in queries]
    if got_labels != expected_labels:
        raise AssertionError(f"Query order/labels must be {expected_labels}, got {got_labels}")

    for q in queries:
        if any(not isinstance(x, str) for x in [q.label, q.title, q.abstract, q.text]):
            raise AssertionError("All QueryItem fields must be strings.")
        if len(q.title.strip()) == 0 or len(q.abstract.strip()) == 0 or len(q.text.strip()) == 0:
            raise AssertionError(f"{q.label} has empty fields.")

        # Strict pipeline check: text must be EXACTLY title + " " + abstract
        if q.text != (q.title + " " + q.abstract):
            raise AssertionError(f"{q.label}: text is not exactly title + ' ' + abstract")
