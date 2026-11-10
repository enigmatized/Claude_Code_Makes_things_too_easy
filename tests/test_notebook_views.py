"""The graded notebook reads saved studies and does not open the stock files."""
import unittest
from unittest.mock import patch

from research.notebook_views import (
    choice_text,
    classifier_growth_table,
    equal_weight_benchmark,
    non_ml_windows_table,
    risk_table,
)


class NotebookViewTests(unittest.TestCase):
    def test_saved_views_do_not_open_a_course_file(self):
        def opened(*_args, **_kwargs):
            raise AssertionError("The notebook views opened a course file.")

        with patch("research.final_books.load_panel", opened), patch("research.data.load_panel", opened):
            frame = equal_weight_benchmark()
            rows = risk_table()
            books = non_ml_windows_table()
            growth = classifier_growth_table()
            text = choice_text()
        official = frame.loc[frame["Period"].eq("Official test")].iloc[0]
        training = frame.loc[frame["Period"].eq("Training")].iloc[0]
        combined = frame.loc[frame["Period"].eq("Combined")].iloc[0]
        self.assertAlmostEqual(float(official["CAGR"]), 0.13096352989640528, places=10)
        self.assertAlmostEqual(float(official["Annual log growth"]), 0.12306995072230081, places=8)
        self.assertEqual(int(training["Months"]) + int(official["Months"]), int(combined["Months"]))
        self.assertEqual(int(combined["Months"]), 228)
        self.assertAlmostEqual(float(combined["Max drawdown"]), float(training["Max drawdown"]), places=10)
        self.assertIn("Annual log growth", rows.columns)
        self.assertEqual(len(rows), 21)
        self.assertGreaterEqual(books["Window"].nunique(), 3)
        self.assertIn("Equal weight", set(books["Book"]))
        self.assertGreaterEqual(growth["Window"].nunique(), 3)
        for name in (
            "Logistic regression",
            "Gaussian naive Bayes",
            "Random forest",
            "Histogram gradient boosting",
        ):
            self.assertIn(name, set(growth["Model"]))
        self.assertIn("factory classifier settings", text)
        self.assertIn("Sharpe is lower", text)
        self.assertIn("annual log growth", text)


if __name__ == "__main__":
    unittest.main()
