"""One-command, non-destructive Xeno-canto dataset collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from dotenv import load_dotenv

from preprocessing.dataset_pipeline import (DEFAULT_WORKERS, candidate_statistics, choose_replacements,
    display_species_name, download_recordings, normalize_species_name, select_recordings, validate_selection,
    verify_metadata_species)

DEFAULT_TARGET = 75
DEFAULT_SEED = 42
DEFAULT_REPLACEMENT_ROUNDS = 3


def print_selection_statistics(selected: pd.DataFrame) -> None:
    print(f"  Selected:          {len(selected)}")
    print(f"  Quality:           {selected['q'].value_counts().to_dict()}")
    print(f"  Playback used:     {selected['playback-used'].value_counts().to_dict()}")
    print(f"  Recording types:   {selected['type'].fillna('unspecified').value_counts().to_dict()}")
    print(f"  Unique recordists: {selected['rec'].nunique(dropna=True)}")
    print(f"  Unique locations:  {selected['loc'].nunique(dropna=True)}")


def print_download_progress(position: int, total: int, status: str, recording_id: int, error: str | None) -> None:
    suffix = "" if status != "failed" else f" ({error})"
    print(f"  [{position}/{total}] XC{recording_id}: {status}{suffix}")


def existing_dataset_is_complete(selected_path: Path, raw_dir: Path, target: int) -> bool:
    if not selected_path.exists():
        return False
    selected = pd.read_csv(selected_path)
    return len(selected) == target and validate_selection(selected, raw_dir).passed


def fetch_metadata(species_name: str, raw_dir: Path) -> Path:
    """Run xcapi using its existing environment configuration; never print it."""
    print("[1/5] Searching Xeno-canto...")
    # Preserve an explicitly exported key while supporting the repository's
    # existing .env configuration. No key is passed on the command line.
    load_dotenv(dotenv_path=Path(".env"), override=False)
    executable = shutil.which("xcapi")
    if executable is None:
        sibling_executable = Path(sys.executable).with_name("xcapi.exe")
        if sibling_executable.exists():
            executable = str(sibling_executable)
    if executable is None:
        raise FileNotFoundError(
            "xcapi was not found. Activate the virtual environment or install "
            "the xenocanto-api dependency."
        )
    subprocess.run([executable, "--en", species_name, "--metadata_only", "--output_dir", str(raw_dir)], check=True)
    metadata_path = raw_dir / "metadata_only.csv"
    if not metadata_path.exists():
        raise RuntimeError(f"xcapi completed but did not create expected metadata: {metadata_path}")
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a curated Xeno-canto dataset for one species.")
    parser.add_argument("species", help='English species name, e.g. "Blue Jay".')
    parser.add_argument("--recordings", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-replacement-rounds", type=int, default=DEFAULT_REPLACEMENT_ROUNDS)
    args = parser.parse_args()
    if args.recordings < 1 or args.workers < 1 or args.max_replacement_rounds < 0:
        parser.error("--recordings and --workers must be at least 1; replacement rounds cannot be negative.")
    species_name = display_species_name(args.species)
    species_key = normalize_species_name(species_name)
    raw_dir = Path("data/raw") / species_key
    selected_path = Path("data/metadata") / f"{species_key}_selected.csv"
    print("=" * 58 + "\nESP32 BIRD AUDIO DATASET BUILDER\n" + "=" * 58)
    print(f"Species: {species_name}\nTarget recordings: {args.recordings}\n")
    try:
        if existing_dataset_is_complete(selected_path, raw_dir, args.recordings):
            print(f"Existing dataset detected and validated: {args.recordings}/{args.recordings} recordings present. Nothing to do.")
            return 0
        # Prevent an incorrect species query from ever being pointed at an existing dataset.
        if selected_path.exists() or (raw_dir.exists() and any(raw_dir.iterdir()) and not (raw_dir / "metadata_only.csv").exists()):
            print(f"Refusing to overwrite incomplete existing data: {raw_dir} or {selected_path}", file=sys.stderr)
            return 2
        metadata_path = raw_dir / "metadata_only.csv"
        if metadata_path.exists():
            print("[1/5] Reusing existing metadata-only result...")
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = fetch_metadata(species_name, raw_dir)
        metadata = pd.read_csv(metadata_path)
        verify_metadata_species(metadata, species_key)
        print("\n[2/5] Evaluating dataset quality...")
        stats = candidate_statistics(metadata)
        print(f"  Total candidates: {stats.total}\n  A/B quality: {stats.quality_usable}\n  A/B and playback=no: {stats.non_playback_usable}\n  Usable after >=5 sec: {stats.duration_usable}\n  Unique recordists: {stats.unique_recordists}\n  Unique locations: {stats.unique_locations}")
        if stats.duration_usable < args.recordings:
            print(f"\nDATASET REJECTED\nSpecies: {species_name}\nUsable after filtering: {stats.duration_usable}\nRequired: {args.recordings}\nReason: Insufficient high-quality, non-playback recordings.\nNo audio was downloaded.")
            return 1
        print("\n[3/5] Selecting diverse recordings...")
        selected = select_recordings(metadata, args.recordings, args.seed)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(selected_path, index=False)
        print_selection_statistics(selected)
        if selected["rec"].nunique(dropna=True) < args.recordings or selected["loc"].nunique(dropna=True) < args.recordings:
            print("  Warning: target met; diversity is reported, not made a hard rejection criterion.")
        print("\n[4/5] Downloading recordings...")
        report = download_recordings(selected, raw_dir, args.workers, print_download_progress)
        failed_ids_seen = set(report.failures)
        print(f"  Downloaded: {report.downloaded}; skipped: {report.skipped}; failed: {len(report.failures)}")
        for round_number in range(1, args.max_replacement_rounds + 1):
            if not report.failures:
                break
            print(f"\nReplacement round {round_number}/{args.max_replacement_rounds}...")
            selected = choose_replacements(
                selected,
                metadata,
                report.failures,
                args.seed + round_number - 1,
                excluded_ids=failed_ids_seen,
            )
            selected.to_csv(selected_path, index=False)
            print(f"  Replaced {len(report.failures)} failed selection(s).")
            report = download_recordings(selected, raw_dir, args.workers, print_download_progress)
            failed_ids_seen.update(report.failures)
            print(f"  Downloaded: {report.downloaded}; skipped: {report.skipped}; failed: {len(report.failures)}")
        print("\n[5/5] Validating dataset...")
        final_report = validate_selection(selected, raw_dir)
        print(f"  {final_report.matched_count}/{len(selected)} selected recordings present\n  Missing: {len(final_report.missing_files)}\n  Unexpected audio files: {len(final_report.extra_audio_files)}\n  Duplicate IDs: {len(final_report.duplicate_ids)}\n  Duplicate filenames: {len(final_report.duplicate_filenames)}")
        if report.failures or len(selected) != args.recordings or not final_report.passed:
            print("DATASET INCOMPLETE", file=sys.stderr)
            if report.failures:
                print(f"Replacement limit reached; latest failed IDs: {list(report.failures)}", file=sys.stderr)
            return 1
        print(f"\n{'=' * 58}\nDATASET COMPLETE\n{'=' * 58}\n{species_name}: {len(selected)} recordings\nRaw audio: {raw_dir}\nMetadata: {selected_path}")
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print(f"\nDATASET COLLECTION FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
