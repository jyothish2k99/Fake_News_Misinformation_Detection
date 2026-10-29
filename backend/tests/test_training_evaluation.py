import unittest

import pandas as pd

from train_model import (
    combine_group_columns,
    near_duplicate_groups,
    source_evaluation,
    stratified_group_split,
    time_evaluation,
)


class TrainingEvaluationTests(unittest.TestCase):
    def test_near_duplicate_articles_share_a_group(self):
        first = "The city council published a public schedule for the annual budget meeting. " * 8
        near_duplicates = [f"{first} Supplement number {index}." for index in range(8)]
        unrelated = "Scientists measured seasonal rainfall across a remote mountain region. " * 4

        groups, audit = near_duplicate_groups(near_duplicates + [unrelated])

        self.assertTrue(all(group == groups[0] for group in groups[:8]))
        self.assertNotEqual(groups[0], groups[-1])
        self.assertEqual(audit["near_duplicate_group_count"], 1)
        self.assertGreaterEqual(audit["near_duplicate_pairs"], 28)

    def test_stratified_group_split_keeps_groups_out_of_both_partitions(self):
        frame = pd.DataFrame(
            {
                "model_text": [f"unique article {index}" for index in range(40)],
                "label": [0] * 20 + [1] * 20,
            }
        )
        groups = [f"duplicate-group-{index // 2}" for index in range(40)]

        train_indices, test_indices = stratified_group_split(frame, groups)

        self.assertFalse(set(groups[index] for index in train_indices) & set(groups[index] for index in test_indices))
        self.assertEqual(set(frame.iloc[test_indices].label), {0, 1})

    def test_source_evaluation_reports_missing_source_data(self):
        frame = pd.DataFrame(
            {
                "source": ["", None, " "],
                "model_text": ["article one", "article two", "article three"],
                "label": [0, 1, 0],
            }
        )

        result = source_evaluation(frame)

        self.assertFalse(result["available"])
        self.assertEqual(result["eligible_rows"], 0)
        self.assertEqual(result["source_groups"], 0)
        self.assertIn("identifiers are missing", result["reason"])

    def test_source_evaluation_reports_metrics_and_group_sample_sizes(self):
        rows = []
        for source_index in range(10):
            for label, class_word in ((0, "real"), (1, "fake")):
                rows.append({
                    "source": f"publisher-{source_index}",
                    "near_duplicate_group": f"unique-{source_index}-{label}",
                    "model_text": f"common report publisher{source_index} class{class_word} verified shared terms",
                    "label": label,
                })

        result = source_evaluation(pd.DataFrame(rows))

        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["source_groups"], 10)
        self.assertEqual(result["eligible_rows"], 20)
        self.assertEqual(result["train_rows"] + result["test_rows"], 20)
        self.assertGreater(result["train_sources"], 0)
        self.assertGreater(result["test_sources"], 0)
        self.assertIn("accuracy", result)
        self.assertIn("confusion_matrix", result)

    def test_time_evaluation_excludes_duplicate_rows_crossing_cutoff(self):
        rows = []
        for day in range(20):
            for label in (0, 1):
                rows.append(
                    {
                        "date": f"2024-01-{day + 1:02d}",
                        "model_text": f"Public records report council budget article {day} label {label}. Common reporting language appears across this article.",
                        "label": label,
                        "source": "",
                        "near_duplicate_group": "shared-future-group" if (day == 1 and label == 0) or (day == 17 and label == 0) else f"group-{day}-{label}",
                    }
                )
        frame = pd.DataFrame(rows)

        result = time_evaluation(frame)

        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["eligible_rows"], 40)
        self.assertEqual(result["missing_or_invalid_date_rows"], 0)
        self.assertEqual(result["near_duplicate_train_rows_excluded"], 1)
        self.assertEqual(result["train_rows"], 31)
        self.assertEqual(result["test_rows"], 8)

    def test_group_merge_combines_source_and_duplicate_links(self):
        frame = pd.DataFrame(
            {
                "source": ["wire-a", "wire-a", "wire-b", "wire-c"],
                "near_duplicate_group": [0, 1, 1, 2],
            }
        )

        groups = combine_group_columns(frame, ["source", "near_duplicate_group"])

        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])


if __name__ == "__main__":
    unittest.main()