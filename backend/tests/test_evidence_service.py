import unittest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.evidence_service import EvidenceService


class EvidenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "alpha alpha common report",
            "alpha common report",
            "alpha common",
            "alpha report",
            "common report",
            "unrelated topic",
        ]
        self.service = EvidenceService.__new__(EvidenceService)
        self.service.df = pd.DataFrame(
            {
                "title": [f"Article {index}" for index in range(len(self.documents))],
                "text": self.documents,
                "label": [0, 1, 0, 1, 0, 1],
                "date": ["2024-01-01"] * len(self.documents),
            }
        )
        self.service.vectorizer = TfidfVectorizer().fit(self.documents)
        self.service.matrix = self.service.vectorizer.transform(self.documents)

    def test_returns_three_highest_similarity_records_in_order(self):
        results = self.service.search("alpha common report")

        self.assertEqual(len(results), 3)
        self.assertEqual([result["title"] for result in results], ["Article 1", "Article 0", "Article 2"])
        self.assertEqual(
            [result["similarity"] for result in results],
            sorted((result["similarity"] for result in results), reverse=True),
        )
        self.assertTrue(all(result["evidence_status"] == "RELATED_DATASET_ARTICLE_NOT_VERIFIED" for result in results))
        self.assertEqual(results[0]["date"], "2024-01-01")

    def test_zero_limit_and_no_match_return_empty(self):
        self.assertEqual(self.service.search("alpha common report", k=0), [])
        self.assertEqual(self.service.search("outofvocabularyterm"), [])

    def test_missing_corpus_returns_empty(self):
        self.service.vectorizer = None
        self.service.matrix = None

        self.assertEqual(self.service.search("alpha common report"), [])


if __name__ == "__main__":
    unittest.main()