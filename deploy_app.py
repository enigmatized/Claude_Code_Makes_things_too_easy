"""Community Cloud entry point: published results only, no research archive."""
import os

os.environ["MLFACTOR_APP_MODE"] = "deploy"
os.environ["MLFACTOR_READ_ONLY"] = "1"

from app import main

main()
