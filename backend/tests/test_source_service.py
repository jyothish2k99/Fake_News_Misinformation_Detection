import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.services.model_service import ModelService
from app.services import source_service


class PublisherScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = Path(self.temp_dir.name) / "publisher_history.csv"
        self.registry.write_text(
            "publisher,real_articles,fake_articles\n"
            "example.com,40,10\n"
            "small.com,10,5\n",
            encoding="utf-8",
        )
        self.path_patch = patch.object(source_service, "SOURCE_PATH", self.registry)
        self.path_patch.start()
        self.env_patch = patch.dict(os.environ, {"SOURCE_MINIMUM_SAMPLE_SIZE": "30"})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_domain_normalization_and_existing_smoothing(self):
        result = source_service.score_for_url("https://www.example.com/article")

        self.assertTrue(result["available"])
        self.assertEqual(result["domain"], "example.com")
        self.assertEqual(result["real_articles"], 40)
        self.assertEqual(result["fake_articles"], 10)
        self.assertEqual(result["total_articles"], 50)
        self.assertEqual(result["credibility_score"], round((40 + 2) / (50 + 4) * 100, 1))

    def test_insufficient_history_has_no_score(self):
        result = source_service.score_for_url("https://small.com/article")

        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "insufficient_history")
        self.assertNotIn("credibility_score", result)
        self.assertEqual(result["message"], source_service.UNAVAILABLE_MESSAGE)

    def test_unknown_and_spoofed_domains_have_no_score(self):
        unknown = source_service.score_for_url("https://unknown.example/article")
        spoofed = source_service.score_for_url("https://evil-example.com/article")
        unsupported_scheme = source_service.score_for_url("javascript://example.com/article")

        self.assertFalse(unknown["available"])
        self.assertFalse(spoofed["available"])
        self.assertFalse(unsupported_scheme["available"])
        self.assertNotIn("credibility_score", unknown)
        self.assertNotIn("credibility_score", spoofed)

    def test_leaderboard_excludes_insufficient_history(self):
        result = source_service.leaderboard()

        self.assertTrue(result["available"])
        self.assertEqual(result["minimum_sample_size"], 30)
        self.assertEqual([row["domain"] for row in result["sources"]], ["example.com"])


class ModelClassMappingTests(unittest.TestCase):
    def predict_with_classes(self, classes, probabilities):
        classifier = SimpleNamespace(classes_=np.array(classes))
        pipeline = SimpleNamespace(
            named_steps={"classifier": classifier},
            predict_proba=lambda _: np.array([probabilities]),
        )
        service = ModelService.__new__(ModelService)
        service.pipeline = pipeline
        service.explain = lambda _text, limit: []
        return service.predict("Headline", "Article text long enough for a model prediction.")

    def test_fake_probability_uses_actual_class_order(self):
        fake_first = self.predict_with_classes(["FAKE", "REAL"], [0.8, 0.2])
        fake_second = self.predict_with_classes(["REAL", "FAKE"], [0.8, 0.2])

        self.assertEqual(fake_first["label"], "FAKE-LIKE")
        self.assertEqual(fake_first["fake_probability"], 0.8)
        self.assertEqual(fake_second["label"], "REAL-LIKE")
        self.assertEqual(fake_second["fake_probability"], 0.2)

    def test_prediction_reports_missing_explanation(self):
        result = self.predict_with_classes([0, 1], [0.2, 0.8])

        self.assertFalse(result["explanation_available"])
        self.assertTrue(result["explanation_message"])


if __name__ == "__main__":
    unittest.main()