"""Score saved controller epochs on the validation split only. Never the test split."""
from brats_debate.cli import main

if __name__ == "__main__":
    main("validate_controller")
