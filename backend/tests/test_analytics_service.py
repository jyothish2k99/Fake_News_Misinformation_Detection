import unittest
from unittest.mock import patch

import pandas as pd

from app.services import analytics_service


class AnalyticsSummaryTests(unittest.TestCase):
    def test_summary_computes_derived_metrics_and_deduplicates(self):
        rows = [
            ("Committee releases health report", "Officials reviewed the medical report and shared the findings with the public.", "health", "2024-01-02", 0),
            ("Local officials publish budget", "The city council published its annual budget after a public meeting on Tuesday.", "politics", "2024-02-02", 0),
            ("Researchers report climate data", "Scientists measured temperatures over ten years and reported a gradual regional increase.", "science", "2024-03-02", 0),
            ("Shocking cure hidden by officials", "A shocking secret cure is being hidden from families, experts warn, and everyone must act now!", "health", "2024-01-03", 1),
            ("Officials caught in election plot", "A shocking election plot proves the officials are hiding the truth from every voter!", "politics", "2024-02-03", 1),
            ("Scientists reveal impossible discovery", "Scientists reveal an impossible discovery that changes everything, according to anonymous sources!", "science", "2024-03-03", 1),
        ]
        columns = ["title", "text", "subject", "date", "label"]
        frame = pd.DataFrame(rows + [rows[0]], columns=columns)

        analytics_service.summary.cache_clear()
        with patch.object(analytics_service, "load_labeled_data", return_value=frame):
            result = analytics_service.summary()

        self.assertTrue(result["available"])
        self.assertEqual(result["total_articles"], 6)
        self.assertEqual(result["duplicate_rows_removed"], 1)
        self.assertEqual(result["conflicting_duplicate_rows_excluded"], 0)
        self.assertEqual(sum(item["total"] for item in result["topics"]), 6)
        self.assertTrue(all(item["avg_flesch_reading_ease"] is not None for item in result["linguistic_comparison"]))
        test = result["emotional_language_test"]
        self.assertEqual(test["real_sample_size"], 3)
        self.assertEqual(test["fake_sample_size"], 3)
        self.assertIsNotNone(test["p_value"])
        self.assertIsNotNone(test["rank_biserial_effect_size"])

    def test_conflicting_duplicate_is_excluded_from_comparison(self):
        shared = ("Same report", "The same article text appears with conflicting labels.", "politics", "2024-01-02")
        frame = pd.DataFrame(
            [(*shared, 0), (*shared, 1), ("Real story", "A city report describes a public meeting held on Monday.", "politics", "2024-01-03", 0)],
            columns=["title", "text", "subject", "date", "label"],
        )

        analytics_service.summary.cache_clear()
        with patch.object(analytics_service, "load_labeled_data", return_value=frame):
            result = analytics_service.summary()

        self.assertEqual(result["conflicting_duplicate_rows_excluded"], 2)
        self.assertEqual(result["total_articles"], 1)
        self.assertIsNone(result["emotional_language_test"]["p_value"])


if __name__ == "__main__":
    unittest.main()