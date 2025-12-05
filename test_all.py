import unittest
import numpy as np

from section3_preprocessing import PrepConfig, build_paper_texts, validate_paper_texts

from section4_bow import (
    BoWConfig,
    build_bow_vectors,
    sparse_matrix_stats as bow_sparse_matrix_stats,
    case_sensitive_stopword_survivors as bow_case_sensitive_stopword_survivors,
)

from section5_tfidf import (
    TfidfConfig,
    build_tfidf_vectors,
    sparse_matrix_stats as tfidf_sparse_matrix_stats,
    case_sensitive_stopword_survivors as tfidf_case_sensitive_stopword_survivors,
    l2_norm_sample_stats,
    top_terms_for_doc,
)

from section6_queries import (
    QueryItem,
    build_query_item,
    get_default_queries,
    get_query_texts,
    validate_queries,
)


class TestSection3Preprocessing(unittest.TestCase):
    def test_basic_concatenation_rule(self):
        records = [
            {"title": "Hello", "abstract": "World", "url": "u", "venue": "v", "year": 2016},
            {"title": "A", "abstract": "B", "url": "u2", "venue": "v2", "year": 2017},
        ]
        cfg = PrepConfig(make_dataframe=False)
        paper_texts, df = build_paper_texts(records, cfg)
        self.assertIsNone(df)
        self.assertEqual(paper_texts, ["Hello World", "A B"])

    def test_stable_length_and_validation(self):
        records = [{"title": "T", "abstract": "X"} for _ in range(5)]
        cfg = PrepConfig(make_dataframe=False)
        paper_texts, _ = build_paper_texts(records, cfg)
        validate_paper_texts(paper_texts, expected_n=5)

    def test_type_coercion_is_minimal_and_safe(self):
        records = [{"title": 123, "abstract": None}]
        cfg = PrepConfig(make_dataframe=False)
        paper_texts, _ = build_paper_texts(records, cfg)
        self.assertEqual(paper_texts[0], "123 ")

    def test_rejects_non_dict_records(self):
        records = [{"title": "OK", "abstract": "OK"}, "not_a_dict"]
        cfg = PrepConfig(make_dataframe=False)
        with self.assertRaises(TypeError):
            build_paper_texts(records, cfg)


class TestSection4BoW(unittest.TestCase):
    def test_build_bow_vectors_shape_and_sparse(self):
        texts = ["cat sat", "dog sat", "cat dog"]
        cfg = BoWConfig(name="t", lowercase=True, stop_words=None)
        vect, X = build_bow_vectors(texts, cfg)
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)
        self.assertTrue(hasattr(X, "nnz"))
        self.assertTrue(hasattr(vect, "vocabulary_"))

    def test_lowercase_merges_tokens(self):
        texts = ["BERT model", "bert model"]

        vect_off, _ = build_bow_vectors(texts, BoWConfig(name="off", lowercase=False, stop_words=None))
        feats_off = set(vect_off.get_feature_names_out().tolist())
        self.assertIn("BERT", feats_off)
        self.assertIn("bert", feats_off)

        vect_on, _ = build_bow_vectors(texts, BoWConfig(name="on", lowercase=True, stop_words=None))
        feats_on = set(vect_on.get_feature_names_out().tolist())
        self.assertNotIn("BERT", feats_on)
        self.assertIn("bert", feats_on)

    def test_stopwords_removed_when_lowercase_true(self):
        texts = ["the cat sat", "the dog sat"]

        vect_no, _ = build_bow_vectors(texts, BoWConfig(name="no", lowercase=True, stop_words=None))
        feats_no = set(vect_no.get_feature_names_out().tolist())
        self.assertIn("the", feats_no)

        vect_yes, _ = build_bow_vectors(texts, BoWConfig(name="yes", lowercase=True, stop_words="english"))
        feats_yes = set(vect_yes.get_feature_names_out().tolist())
        self.assertNotIn("the", feats_yes)

    def test_case_sensitive_stopword_survivors_detected(self):
        texts = ["The cat and the dog"]
        vect, _ = build_bow_vectors(texts, BoWConfig(name="cs", lowercase=False, stop_words="english"))
        survivors = bow_case_sensitive_stopword_survivors(vect)
        self.assertGreaterEqual(survivors, 1)

    def test_sparse_stats_sane(self):
        texts = ["cat sat", "dog barked"]
        _, X = build_bow_vectors(texts, BoWConfig(name="s", lowercase=True, stop_words=None))
        s = bow_sparse_matrix_stats(X)
        self.assertEqual(s["n_docs"], 2)
        self.assertGreater(s["n_vocab"], 0)
        self.assertGreaterEqual(s["density"], 0.0)
        self.assertLessEqual(s["density"], 1.0)


class TestSection5Tfidf(unittest.TestCase):
    def test_build_tfidf_vectors_shape_sparse_and_float(self):
        texts = ["cat sat", "dog sat", "cat dog"]
        cfg = TfidfConfig(name="t", lowercase=True, stop_words=None)
        vect, X = build_tfidf_vectors(texts, cfg)
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)
        self.assertTrue(hasattr(X, "nnz"))
        self.assertTrue("float" in str(X.dtype))

        stats = tfidf_sparse_matrix_stats(X)
        self.assertEqual(stats["n_docs"], 3)
        self.assertGreater(stats["n_vocab"], 0)

    def test_tfidf_lowercase_merges_tokens(self):
        texts = ["BERT model", "bert model"]

        vect_off, _ = build_tfidf_vectors(texts, TfidfConfig(name="off", lowercase=False, stop_words=None))
        feats_off = set(vect_off.get_feature_names_out().tolist())
        self.assertIn("BERT", feats_off)
        self.assertIn("bert", feats_off)

        vect_on, _ = build_tfidf_vectors(texts, TfidfConfig(name="on", lowercase=True, stop_words=None))
        feats_on = set(vect_on.get_feature_names_out().tolist())
        self.assertNotIn("BERT", feats_on)
        self.assertIn("bert", feats_on)

    def test_tfidf_stopwords_removed_when_lowercase_true(self):
        texts = ["the cat sat", "the dog sat"]

        vect_no, _ = build_tfidf_vectors(texts, TfidfConfig(name="no", lowercase=True, stop_words=None))
        feats_no = set(vect_no.get_feature_names_out().tolist())
        self.assertIn("the", feats_no)

        vect_yes, _ = build_tfidf_vectors(texts, TfidfConfig(name="yes", lowercase=True, stop_words="english"))
        feats_yes = set(vect_yes.get_feature_names_out().tolist())
        self.assertNotIn("the", feats_yes)

    def test_tfidf_case_sensitive_stopword_survivors_detected(self):
        texts = ["The cat and the dog"]
        vect, _ = build_tfidf_vectors(texts, TfidfConfig(name="cs", lowercase=False, stop_words="english"))
        survivors = tfidf_case_sensitive_stopword_survivors(vect)
        self.assertGreaterEqual(survivors, 1)

    def test_tfidf_l2_norms_are_about_one(self):
        texts = ["cat sat", "dog sat", "cat dog"]
        _, X = build_tfidf_vectors(texts, TfidfConfig(name="n", lowercase=True, stop_words=None))
        norms = l2_norm_sample_stats(X, sample_n=3, seed=0)
        self.assertAlmostEqual(norms["mean"], 1.0, places=6)

    def test_tfidf_idf_downweights_common_terms_when_tf_equal(self):
        texts = ["common rare1", "common rare2"]
        vect, X = build_tfidf_vectors(texts, TfidfConfig(name="idf", lowercase=True, stop_words=None))
        feats = vect.get_feature_names_out().tolist()

        i_common = feats.index("common")
        i_rare1 = feats.index("rare1")

        row0 = X.getrow(0).toarray().ravel()
        self.assertGreater(row0[i_rare1], row0[i_common])

    def test_top_terms_for_doc_returns_sorted(self):
        texts = ["cat sat sat", "dog sat"]
        vect, X = build_tfidf_vectors(texts, TfidfConfig(name="top", lowercase=True, stop_words=None))
        top = top_terms_for_doc(X, vect, doc_index=0, top_n=5)
        self.assertTrue(len(top) > 0)
        weights = [w for _, w in top]
        self.assertTrue(all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1)))


class TestSection6Queries(unittest.TestCase):
    def test_default_queries_exist_and_valid(self):
        queries = get_default_queries(joiner=" ")
        validate_queries(queries)  # should not raise
        self.assertEqual(len(queries), 3)
        self.assertEqual([q.label for q in queries], ["Query 1", "Query 2", "Query 3"])
        self.assertTrue(all(isinstance(q, QueryItem) for q in queries))

    def test_query_texts_are_in_order(self):
        queries = get_default_queries(joiner=" ")
        texts = get_query_texts(queries)
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(isinstance(t, str) and len(t) > 0 for t in texts))

    def test_build_query_item_strict_joiner(self):
        with self.assertRaises(ValueError):
            build_query_item("Query 1", "T", "A", joiner="  ")  # not exactly one space
        with self.assertRaises(ValueError):
            build_query_item("Query 1", "T", "A", joiner="")    # not exactly one space

    def test_validate_queries_rejects_wrong_order(self):
        q1 = build_query_item("Query 1", "T1", "A1", joiner=" ")
        q2 = build_query_item("Query 2", "T2", "A2", joiner=" ")
        q3 = build_query_item("Query 3", "T3", "A3", joiner=" ")
        with self.assertRaises(AssertionError):
            validate_queries([q2, q1, q3])  # wrong order/labels

    def test_validate_queries_rejects_bad_text_pipeline(self):
        bad = QueryItem(label="Query 1", title="T", abstract="A", text="T  A")
        q2 = build_query_item("Query 2", "T2", "A2", joiner=" ")
        q3 = build_query_item("Query 3", "T3", "A3", joiner=" ")
        with self.assertRaises(AssertionError):
            validate_queries([bad, q2, q3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
