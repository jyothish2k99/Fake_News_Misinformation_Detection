import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_transformer import prepare_records, split_records, validate_local_model_path


class TransformerPreparationTests(unittest.TestCase):
    def test_prepare_records_removes_duplicates_and_conflicting_labels(self):
        shared = "An article text that is long enough to remain in the training data."
        frame = pd.DataFrame(
            [
                {"title": "Duplicate", "text": shared, "label": 0},
                {"title": "Duplicate", "text": shared, "label": 0},
                {"title": "Duplicate", "text": shared, "label": 1},
                {"title": "Real article", "text": "A report about local public budget decisions and a council meeting.", "label": 0},
                {"title": "Another real article", "text": "The public office released a routine schedule for next month's council meetings.", "label": 0},
                {"title": "Fake article", "text": "A shocking secret plot is hidden from every family and every citizen.", "label": 1},
                {"title": "Another fake article", "text": "Anonymous posts claim a hidden plot controls every city office without evidence.", "label": 1},
            ]
        )

        cleaned, audit = prepare_records(frame)

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(set(cleaned["label"]), {0, 1})
        self.assertEqual(audit["exact_duplicate_rows_removed"], 0)
        self.assertEqual(audit["conflicting_duplicate_rows_excluded"], 3)

    def test_split_is_reproducible_and_stratified(self):
        frame = pd.DataFrame({"model_text": [f"article {i}" for i in range(20)], "label": [0] * 10 + [1] * 10})

        train_a, test_a = split_records(frame, seed=7)
        train_b, test_b = split_records(frame, seed=7)

        self.assertEqual(train_a["model_text"].tolist(), train_b["model_text"].tolist())
        self.assertEqual(test_a["model_text"].tolist(), test_b["model_text"].tolist())
        self.assertEqual(set(test_a["label"]), {0, 1})
        self.assertAlmostEqual(len(test_a) / len(frame), 0.2, delta=0.1)

        _, larger_test = split_records(frame, test_size=0.4, seed=7)
        self.assertGreater(len(larger_test), len(test_a))

        with self.assertRaises(ValueError):
            split_records(frame, test_size=1.0, seed=7)

    def test_model_path_must_already_exist_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "local-model"
            local_path.mkdir()
            self.assertEqual(validate_local_model_path(local_path), local_path.resolve())

            with self.assertRaises(FileNotFoundError):
                validate_local_model_path("distilbert-base-uncased")


if __name__ == "__main__":
    unittest.main()