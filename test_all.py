import unittest

from section3_preprocessing import PrepConfig, build_paper_texts, validate_paper_texts
from section4_bow import (
    BoWConfig,
    build_bow_vectors,
    sparse_matrix_stats,
    case_sensitive_stopword_survivors,
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
        validate_paper_texts(paper_texts, expected_n=5)  # should not raise

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
        self.assertTrue(hasattr(X, "nnz"))  # sparse matrices expose nnz
        self.assertTrue(hasattr(vect, "vocabulary_"))

    def test_lowercase_merges_tokens(self):
        texts = ["BERT model", "bert model"]

        vect_off, X_off = build_bow_vectors(texts, BoWConfig(name="off", lowercase=False, stop_words=None))
        feats_off = set(vect_off.get_feature_names_out().tolist())
        self.assertIn("BERT", feats_off)
        self.assertIn("bert", feats_off)

        vect_on, X_on = build_bow_vectors(texts, BoWConfig(name="on", lowercase=True, stop_words=None))
        feats_on = set(vect_on.get_feature_names_out().tolist())
        self.assertNotIn("BERT", feats_on)
        self.assertIn("bert", feats_on)

        self.assertEqual(X_off.shape[0], X_on.shape[0])

    def test_stopwords_removed_when_lowercase_true(self):
        texts = ["the cat sat", "the dog sat"]

        vect_no, _ = build_bow_vectors(texts, BoWConfig(name="no", lowercase=True, stop_words=None))
        feats_no = set(vect_no.get_feature_names_out().tolist())
        self.assertIn("the", feats_no)

        vect_yes, _ = build_bow_vectors(texts, BoWConfig(name="yes", lowercase=True, stop_words="english"))
        feats_yes = set(vect_yes.get_feature_names_out().tolist())
        self.assertNotIn("the", feats_yes)

    def test_case_sensitive_stopword_survivors_detected(self):
        texts = ["The cat and the dog"]  # 'The' may survive when lowercase=False with english stopwords
        vect, _ = build_bow_vectors(texts, BoWConfig(name="cs", lowercase=False, stop_words="english"))
        survivors = case_sensitive_stopword_survivors(vect)
        self.assertGreaterEqual(survivors, 1)

    def test_sparse_stats_sane(self):
        texts = ["cat sat", "dog barked"]
        vect, X = build_bow_vectors(texts, BoWConfig(name="s", lowercase=True, stop_words=None))
        s = sparse_matrix_stats(X)
        self.assertEqual(s["n_docs"], 2)
        self.assertGreater(s["n_vocab"], 0)
        self.assertGreaterEqual(s["density"], 0.0)
        self.assertLessEqual(s["density"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
