"""Manual integrity check for the completed five-species dataset."""

from pathlib import Path
import sys

import pandas as pd

# Permit the documented `python tests/check_dataset.py` invocation on Windows.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.dataset_pipeline import validate_selection


SPECIES = ["carolina_wren", "mourning_dove", "red_winged_blackbird", "great_horned_owl", "song_sparrow"]
EXPECTED_PER_SPECIES = 75


def check_species(species: str) -> bool:
    """Check exact selected filenames for one species; metadata is not audio."""
    selected_path = Path("data/metadata") / f"{species}_selected.csv"
    raw_dir = Path("data/raw") / species
    print(f"\n{'-' * 70}\nSPECIES: {species}\n{'-' * 70}")
    if not selected_path.exists() or not raw_dir.exists():
        print(f"Missing selection CSV or audio directory: {selected_path}, {raw_dir}")
        return False
    selected = pd.read_csv(selected_path)
    report = validate_selection(selected, raw_dir)
    print(f"Selected recordings:   {report.selected_count}")
    print(f"Selected filenames:    {report.selected_filenames}")
    print(f"Matched recordings:    {report.matched_count}")
    print(f"Missing recordings:    {len(report.missing_files)}")
    print(f"Extra audio files:     {len(report.extra_audio_files)}")
    print(f"Duplicate IDs:         {len(report.duplicate_ids)}")
    print(f"Duplicate filenames:   {len(report.duplicate_filenames)}")
    for filename in sorted(report.missing_files):
        print(f"  Missing: {filename}")
    for filename in sorted(report.extra_audio_files):
        print(f"  Extra audio: {filename}")
    passed = len(selected) == EXPECTED_PER_SPECIES and report.selected_filenames == EXPECTED_PER_SPECIES and report.passed
    print("STATUS: PASS" if passed else "STATUS: FAIL")
    return passed


def main() -> None:
    print("=" * 70 + "\nFINAL DATASET INTEGRITY CHECK\n" + "=" * 70)
    results = {species: check_species(species) for species in SPECIES}
    total = sum(len(pd.read_csv(Path("data/metadata") / f"{species}_selected.csv"))
                for species in SPECIES if (Path("data/metadata") / f"{species}_selected.csv").exists())
    print(f"\nTotal selected recordings: {total}\nExpected recordings: {len(SPECIES) * EXPECTED_PER_SPECIES}")
    print("OVERALL STATUS: PASS" if all(results.values()) and total == len(SPECIES) * EXPECTED_PER_SPECIES
          else "OVERALL STATUS: FAIL")


if __name__ == "__main__":
    main()
