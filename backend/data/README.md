Place the supplied `Fake.csv` and `True.csv` files in this directory before training.

Optional `publisher_history.csv` format:
- `publisher`: canonical publisher name/domain
- `real_articles`: count of historical articles labelled real by a documented evaluation process
- `fake_articles`: count of historical articles labelled fake by the same process

Do not use the example rows as real publisher data. Copy the example schema only and replace it with sourced, auditable records.

Article-level scores require at least 30 historical rows by default. Set `SOURCE_MINIMUM_SAMPLE_SIZE` to a positive integer to change the threshold. Scores use the existing Beta(2,2)-smoothed historical REAL-label share; see the root README for the formula and interpretation limits.
