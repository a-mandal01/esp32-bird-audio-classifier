"""Backward-compatible utility to replace one failed selected recording."""

import argparse
from pathlib import Path

import pandas as pd

try:
    from preprocessing.dataset_pipeline import choose_replacements
except ModuleNotFoundError:
    from dataset_pipeline import choose_replacements


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace a failed selected Xeno-canto recording.")
    parser.add_argument("--species", required=True, help="Normalized species directory name.")
    parser.add_argument("--failed_id", required=True, type=int, help="Failed Xeno-canto ID.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected_file = Path("data/metadata") / f"{args.species}_selected.csv"
    before = pd.read_csv(selected_file)
    after = choose_replacements(before, pd.read_csv(Path("data/raw") / args.species / "metadata_only.csv"),
                                [args.failed_id], args.seed)
    after.to_csv(selected_file, index=False)
    added = after[~after["id"].isin(before["id"])]
    print(f"Replacement selected: XC{int(added.iloc[0]['id'])}")
    print(f"Saved to: {selected_file}")


if __name__ == "__main__":
    main()
