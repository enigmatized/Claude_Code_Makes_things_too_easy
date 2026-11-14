"""Robustness readers and submission checks read saved evidence only."""
import unittest
from unittest.mock import patch

import numpy as np

from research.notebook_views import (
    ENGINEERING_STUDY, cost_table, pinned_engineering, risk_table, significance_table, source_note, tuned_final_table,
)
from research.submission_checks import submission_checks


def _no_course_files(*_args, **_kwargs):
    raise AssertionError("A saved-result reader opened a course file.")


class RobustnessViewTests(unittest.TestCase):
    def test_engineering_study_is_pinned_not_newest(self):
        self.assertEqual(pinned_engineering().name, ENGINEERING_STUDY)
        self.assertIn(ENGINEERING_STUDY, source_note("engineering"))

    def test_tuned_table_matches_the_frozen_chosen_book(self):
        with patch("research.data.load_panel", _no_course_files), patch("research.final_books.load_panel", _no_course_files):
            tuned = tuned_final_table()
            risk = risk_table()
        chosen = risk.loc[risk.Window.eq("Official test") & risk.Book.eq("Top 10 positive-vol"), "CAGR"].iloc[0]
        self.assertAlmostEqual(float(tuned.iloc[0]["Factory CAGR"]), float(chosen), places=10)
        self.assertEqual(len(tuned), 9)                       # original + 4 models x 2 overlays
        self.assertTrue(tuned["Tuned − factory CAGR"].abs().lt(0.01).all())

    def test_costs_only_lower_returns_and_zero_cost_matches_gross(self):
        with patch("research.data.load_panel", _no_course_files), patch("research.final_books.load_panel", _no_course_files):
            costs = cost_table("test").set_index("Book")
            risk = risk_table()
        gross = risk.loc[risk.Window.eq("Official test")].set_index("Book")["CAGR"]
        for book in ("Equal weight", "Top 10 positive-vol"):
            self.assertAlmostEqual(costs.loc[book, "CAGR @ 0 bps"], gross[book], places=10)
            self.assertGreater(costs.loc[book, "CAGR @ 0 bps"], costs.loc[book, "CAGR @ 10 bps"])
            self.assertGreater(costs.loc[book, "CAGR @ 10 bps"], costs.loc[book, "CAGR @ 25 bps"])

    def test_significance_is_deterministic_and_well_formed(self):
        first, second = significance_table(), significance_table()
        np.testing.assert_allclose(first.select_dtypes("number"), second.select_dtypes("number"))
        self.assertEqual(list(first.Months), [108, 60, 168])
        self.assertTrue((first["95% CI low"] <= first["Extra return / yr"]).all())
        self.assertTrue((first["Extra return / yr"] <= first["95% CI high"]).all())
        self.assertTrue(first["p-value"].between(0, 1).all())

    def test_report_and_slide_helpers_read_saved_files_only(self):
        from research.notebook_views import glance_table, strong_pairs
        with patch("research.data.load_panel", _no_course_files), patch("research.final_books.load_panel", _no_course_files):
            glance = glance_table()
        self.assertEqual(len(glance), 6)                      # 4 frozen books + 2 Section 7 rows
        strong, total = strong_pairs()
        self.assertEqual(total, 190)                          # 20 factors -> 190 pairs
        self.assertLess(strong, total)

    def test_submission_checks_pass_on_saved_evidence(self):
        checks = submission_checks()
        self.assertGreaterEqual(len(checks), 8)
        self.assertTrue(all(checks.values()), [k for k, v in checks.items() if not v])


if __name__ == "__main__":
    unittest.main()
