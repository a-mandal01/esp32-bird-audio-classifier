"""Backward-compatible CLI for downloading an existing selected CSV."""

import argparse
from pathlib import Path

import pandas as pd

try:
    from preprocessing.dataset_pipeline import DEFAULT_WORKERS, download_recordings
except ModuleNotFoundError:
    from dataset_pipeline import DEFAULT_WORKERS, download_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Xeno-canto recordings.")
    parser.add_argument("--species", required=True, help="Normalized species directory name.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    selected = pd.read_csv(Path("data/metadata") / f"{args.species}_selected.csv")
    print(f"Downloading {len(selected)} recordings for {args.species} with {args.workers} workers.")
    def progress(position, total, status, recording_id, error):
        print(f"[{position}/{total}] XC{recording_id}: {status}" + (f" ({error})" if error else ""))
    report = download_recordings(selected, Path("data/raw") / args.species, args.workers, progress)
    print(f"Downloaded: {report.downloaded}; skipped: {report.skipped}; failed: {len(report.failures)}")


if __name__ == "__main__":
    main()
