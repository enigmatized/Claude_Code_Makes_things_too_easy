"""Search return-quintile categories on the training file.

Does not open the official test CSV and does not register an experiment-catalog run.
"""
import json

from research.grouping import run_grouping_study


def main():
    directory = run_grouping_study(progress=print)
    print(json.dumps({"directory": str(directory)}))


if __name__ == "__main__":
    main()
