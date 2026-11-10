"""Category search stays on the frame it is given and refuses a read-only fit."""
import os
import unittest

import numpy as np
import pandas as pd

from research.grouping import N_CLASSES, quintile_labels, search_categories
from research.legacy import TARGET


class GroupingTests(unittest.TestCase):
    def test_quintiles_are_balanced_inside_the_month(self):
        dates = np.repeat(pd.Timestamp("2006-01-31"), 20)
        labels = quintile_labels(dates, np.linspace(-0.2, 0.4, 20))
        counts = labels.value_counts().sort_index()
        self.assertEqual(list(counts.index), list(range(N_CLASSES)))
        self.assertTrue(counts.max() - counts.min() <= 1)

    def test_search_uses_only_labels_known_at_the_origin(self):
        frame = _panel()
        labels = sorted(frame.label_available_at.unique())
        asof = labels[19]
        result = search_categories(
            frame, ["f0", "f1", "f2"], asof=asof, n_splits=2, min_train_months=12,
            neighbor_grid=(3, 5), logistic_grid=(1.0,), progress=None,
        )
        metadata = result["metadata"]
        self.assertLessEqual(pd.Timestamp(metadata["asof"]), pd.Timestamp(asof))
        self.assertGreater(pd.Timestamp(metadata["holdout_first_return_month"]), pd.Timestamp(asof))
        self.assertFalse(metadata["official_test_opened"])
        knn = next(row for row in metadata["classification"] if row["model_key"] == "knn")
        self.assertGreater(knn["holdout_accuracy"], knn["chance_accuracy"])
        keys = [row["model_key"] for row in metadata["portfolio"]]
        self.assertEqual(keys, ["none", "knn", "logistic"])
        self.assertIn("annualized_log_growth", metadata["portfolio"][1])

    def test_read_only_refuses_to_fit(self):
        previous = os.environ.get("MLFACTOR_READ_ONLY")
        os.environ["MLFACTOR_READ_ONLY"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                search_categories(_panel(), ["f0", "f1", "f2"], asof="2005-12-31")
        finally:
            if previous is None:
                os.environ.pop("MLFACTOR_READ_ONLY", None)
            else:
                os.environ["MLFACTOR_READ_ONLY"] = previous


def _panel():
    dates = pd.date_range("2004-01-31", periods=36, freq="ME")
    rows = []
    generator = np.random.default_rng(0)
    for date in dates:
        order = np.arange(24)
        returns = order / 24 * 0.15
        for name, ret in zip(order, returns):
            rows.append({
                "date": date,
                "stock_id": int(name),
                "f0": float(name),
                "f1": float(generator.normal()),
                "f2": float(generator.normal()),
                TARGET: float(ret),
                "label_available_at": date + pd.offsets.MonthEnd(1),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
