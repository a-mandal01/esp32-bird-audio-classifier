"""Backward-compatible CLI for diverse selection from existing metadata."""

import argparse
from pathlib import Path

import pandas as pd

try:
    from preprocessing.dataset_pipeline import candidate_statistics, select_recordings
except ModuleNotFoundError:  # Supports `python preprocessing/select_recordings.py`.
    from dataset_pipeline import candidate_statistics, select_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description="Select diverse Xeno-canto recordings from metadata.")
    parser.add_argument("--species", default="song_sparrow", help="Normalized species directory name.")
    parser.add_argument("--recordings", type=int, default=75, help="Target recording count.")
    parser.add_argument("--seed", type=int, default=42, help="Selection random seed.")
    args = parser.parse_args()
    source = Path("data/raw") / args.species / "metadata_only.csv"
    output = Path("data/metadata") / f"{args.species}_selected.csv"
    metadata = pd.read_csv(source)
    stats = candidate_statistics(metadata)
    print(f"Total recordings found: {stats.total}")
    print(f"After quality/playback filter: {stats.non_playback_usable}")
    print(f"After duration filter: {stats.duration_usable}")
    selected = select_recordings(metadata, args.recordings, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"\nSelected recordings: {len(selected)}")
    print(f"Unique recordists: {selected['rec'].nunique(dropna=True)}")
    print(f"Unique locations: {selected['loc'].nunique(dropna=True)}")
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
