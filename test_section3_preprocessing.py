import unittest

from section3_preprocessing import PrepConfig, build_paper_texts, validate_paper_texts


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
        # title becomes "123", abstract becomes ""
        self.assertEqual(paper_texts[0], "123 ")

    def test_rejects_non_dict_records(self):
        records = [{"title": "OK", "abstract": "OK"}, "not_a_dict"]
        cfg = PrepConfig(make_dataframe=False)
        with self.assertRaises(TypeError):
            build_paper_texts(records, cfg)

    def test_dataframe_creation_if_enabled(self):
        records = [{"title": "T", "abstract": "A", "url": "u", "venue": "EMNLP", "year": 2016}]
        cfg = PrepConfig(make_dataframe=True)
        paper_texts, df = build_paper_texts(records, cfg)
        self.assertIsNotNone(df)
        self.assertIn("text", df.columns)
        self.assertEqual(df.loc[0, "text"], paper_texts[0])


if __name__ == "__main__":
    unittest.main()
