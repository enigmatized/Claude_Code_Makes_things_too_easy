"""The part-8 reader only reads saved files and keeps its text tied to the saved numbers."""
import json
import unittest
from unittest.mock import patch

from research.notebook_views import vol_neutral_model_table, vol_neutral_note, vol_neutral_table
from research.legacy import PROJECT_ROOT


class VolNeutralViewTests(unittest.TestCase):
    def test_reads_saved_files_without_opening_course_data(self):
        def opened(*_args, **_kwargs):
            raise AssertionError("The part-8 reader opened a course file.")

        with patch("research.data.load_panel", opened):
            walk = vol_neutral_table("oos_2005_2009")
            test = vol_neutral_table("test_2010_2018")
            models = vol_neutral_model_table()
            note = vol_neutral_note()
        meta = json.loads((PROJECT_ROOT / "0_7_Feature_Engineering" / "part8_meta.json").read_text())
        self.assertEqual(list(walk["Strategy"]), list(test["Strategy"]))
        self.assertIn("Equal weight", set(test["Strategy"]))
        self.assertEqual(len(models), 8)                                   # four models x two labels
        self.assertTrue(models["Test AUC"].between(0.4, 0.6).all())
        self.assertIn(f"{meta['chosen_weights']['momentum']:.2f}", note)  # the note states the saved weight
        self.assertTrue(meta["test_previously_viewed"])
        self.assertFalse(meta["test_used_for_choices"])
        with self.assertRaises(ValueError):
            vol_neutral_table("not_a_window")


if __name__ == "__main__":
    unittest.main()
