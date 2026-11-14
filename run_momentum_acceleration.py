"""Explicitly compute the declared acceleration-to-cash experiment."""

from research.legacy import PROJECT_ROOT
from research.momentum_acceleration import create_acceleration_study

if __name__ == "__main__":
    print(create_acceleration_study(PROJECT_ROOT), flush=True)
