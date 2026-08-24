"""TMA-selection entry point using the shared GHIST+ trainer."""

from train import TrainingVariant, run_cli


if __name__ == "__main__":
    run_cli(TrainingVariant.TMA_SELECT)
