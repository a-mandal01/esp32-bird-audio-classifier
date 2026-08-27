"""Reusable Xeno-canto dataset collection helpers.

This module intentionally operates on metadata and original downloads only. It
does not inspect, transform, or otherwise preprocess audio.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Callable, Iterable

import pandas as pd
import requests


VALID_QUALITIES = ("A", "B")
MIN_DURATION_SECONDS = 5
DEFAULT_WORKERS = 6
MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 1024
TIMEOUT = 120
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass(frozen=True)
class CandidateStatistics:
    total: int
    quality_usable: int
    non_playback_usable: int
    duration_usable: int
    unique_recordists: int
    unique_locations: int


@dataclass(frozen=True)
class DownloadReport:
    downloaded: int
    skipped: int
    failures: tuple[int, ...]


@dataclass(frozen=True)
class ValidationReport:
    selected_count: int
    selected_filenames: int
    matched_count: int
    missing_files: frozenset[str]
    extra_audio_files: frozenset[str]
    duplicate_ids: frozenset[int]
    duplicate_filenames: frozenset[str]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_files
            or self.extra_audio_files
            or self.duplicate_ids
            or self.duplicate_filenames
        )


def normalize_species_name(species: str) -> str:
    """Return a safe, stable directory stem for an English species name."""
    normalized = re.sub(r"[^a-z0-9]+", "_", species.strip().casefold()).strip("_")
    if not normalized:
        raise ValueError("Species name must contain letters or numbers.")
    return normalized


def display_species_name(species: str) -> str:
    """Normalize whitespace without changing the user's capitalization."""
    name = " ".join(species.split())
    if not name:
        raise ValueError("Species name cannot be empty.")
    return name


def verify_metadata_species(df: pd.DataFrame, species_key: str) -> None:
    """Reject cached metadata that clearly belongs to a different species.

    This guard is specifically for the historical failure mode where xcapi was
    accidentally directed at another species' raw directory.  It runs before a
    selection CSV can be created or changed.
    """
    if "en" not in df.columns:
        raise ValueError("Metadata is missing the Xeno-canto English-name column ('en').")
    names = {normalize_species_name(name) for name in df["en"].dropna().astype(str)}
    if not names:
        raise ValueError("Metadata contains no English species names to verify.")
    if names != {species_key}:
        found = ", ".join(sorted(names))
        raise ValueError(f"Metadata species mismatch: expected '{species_key}', found '{found}'.")


def length_to_seconds(length: object) -> int:
    """Convert a Xeno-canto MM:SS duration to seconds; invalid values are zero."""
    if pd.isna(length):
        return 0
    parts = str(length).split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def require_metadata_columns(df: pd.DataFrame) -> None:
    required = {"id", "file", "file-name", "q", "playback-used", "length", "rec", "loc"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata is missing required Xeno-canto columns: {', '.join(missing)}")


def eligible_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's A/B, no-playback, and five-second eligibility rules."""
    require_metadata_columns(df)
    candidates = df[df["q"].isin(VALID_QUALITIES) & (df["playback-used"] == "no")].copy()
    candidates["length_seconds"] = candidates["length"].map(length_to_seconds)
    return candidates[candidates["length_seconds"] >= MIN_DURATION_SECONDS].copy()


def candidate_statistics(df: pd.DataFrame) -> CandidateStatistics:
    require_metadata_columns(df)
    quality = df[df["q"].isin(VALID_QUALITIES)]
    non_playback = quality[quality["playback-used"] == "no"]
    usable = eligible_candidates(df)
    return CandidateStatistics(len(df), len(quality), len(non_playback), len(usable),
                               usable["rec"].nunique(dropna=True), usable["loc"].nunique(dropna=True))


def select_recordings(df: pd.DataFrame, target: int = 75, seed: int = 42) -> pd.DataFrame:
    """Select recordings using the existing shuffled, recordist-first strategy."""
    if target < 1:
        raise ValueError("target must be at least 1")
    candidates = eligible_candidates(df).sample(frac=1, random_state=seed).reset_index(drop=True)
    selected: list[dict] = []
    used_recordists: set[object] = set()
    for _, row in candidates.iterrows():
        if row["rec"] not in used_recordists:
            selected.append(row.to_dict())
            used_recordists.add(row["rec"])
        if len(selected) >= target:
            break
    if len(selected) < target:
        selected_ids = {row["id"] for row in selected}
        remaining = candidates[~candidates["id"].isin(selected_ids)].sample(frac=1, random_state=seed)
        selected.extend(remaining.head(target - len(selected)).to_dict("records"))
    return pd.DataFrame(selected).drop(columns=["length_seconds"], errors="ignore")


def choose_replacements(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    failed_ids: Iterable[int],
    seed: int = 42,
    excluded_ids: Iterable[int] = (),
) -> pd.DataFrame:
    """Replace failed IDs while preferring new recordists, then new locations.

    ``excluded_ids`` keeps recordings that failed in an earlier round out of the
    candidate pool, so an unavailable URL is never retried as a replacement.
    """
    require_metadata_columns(selected)
    original_count = len(selected)
    failed_ids = {int(recording_id) for recording_id in failed_ids}
    selected_ids = {int(recording_id) for recording_id in selected["id"]}
    unknown_ids = failed_ids - selected_ids
    if unknown_ids:
        raise ValueError(f"Cannot replace IDs absent from selection: {sorted(unknown_ids)}")
    updated = selected[~selected["id"].isin(failed_ids)].copy()
    unavailable_ids = failed_ids | {int(recording_id) for recording_id in excluded_ids}
    available = eligible_candidates(candidates)
    available = available[~available["id"].isin(unavailable_ids)].copy()
    for failed_id in sorted(failed_ids):
        ranked = available[~available["id"].isin(set(updated["id"]))].copy()
        if ranked.empty:
            raise RuntimeError(f"No valid replacement recordings remain for XC{failed_id}.")
        ranked["new_recordist"] = ~ranked["rec"].isin(set(updated["rec"]))
        ranked["new_location"] = ~ranked["loc"].isin(set(updated["loc"]))
        ranked = ranked.sample(frac=1, random_state=seed).sort_values(
            by=["new_recordist", "new_location"], ascending=False
        )
        updated = pd.concat([updated, ranked.iloc[[0]]], ignore_index=True)
    updated = updated.drop(columns=["length_seconds", "new_recordist", "new_location"], errors="ignore")
    if len(updated) != original_count or updated["id"].duplicated().any():
        raise RuntimeError("Replacement bookkeeping did not preserve a unique selection of the original size.")
    return updated


def _download_one(row: pd.Series, output_dir: Path) -> tuple[str, int, str, str | None]:
    recording_id = int(row["id"])
    download_url = row["file"]  # Direct file URL, never Xeno-canto's recording-page URL.
    filename = Path(str(row["file-name"])).name
    species_dir = output_dir / f"{row.get('gen', 'unknown')}_{row.get('sp', 'unknown')}"
    species_dir.mkdir(parents=True, exist_ok=True)
    output_file = species_dir / filename
    if output_file.exists() and output_file.stat().st_size > 0:
        return "skipped", recording_id, filename, None
    temp_file = output_file.with_suffix(output_file.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(download_url, stream=True, timeout=TIMEOUT,
                                    headers={"User-Agent": "esp32-bird-audio-classifier/1.0"})
            response.raise_for_status()
            with open(temp_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
            if not temp_file.exists() or temp_file.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty")
            temp_file.replace(output_file)
            return "downloaded", recording_id, filename, None
        except Exception as error:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
            else:
                return "failed", recording_id, filename, str(error)
    raise AssertionError("unreachable")


def download_recordings(selected: pd.DataFrame, output_dir: Path, workers: int = DEFAULT_WORKERS,
                        progress: Callable[[int, int, str, int, str | None], None] | None = None) -> DownloadReport:
    """Download a selection concurrently and return all failed Xeno-canto IDs."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    require_metadata_columns(selected)
    downloaded = skipped = 0
    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_one, row, output_dir) for _, row in selected.iterrows()]
        for position, future in enumerate(as_completed(futures), start=1):
            status, recording_id, _filename, error = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append(recording_id)
            if progress:
                progress(position, len(futures), status, recording_id, error)
    return DownloadReport(downloaded, skipped, tuple(failures))


def validate_selection(selected: pd.DataFrame, audio_folder: Path) -> ValidationReport:
    """Validate exact selected ``file-name`` values; never infer IDs from names."""
    require_metadata_columns(selected)
    selected_filenames = set(selected["file-name"].dropna().astype(str))
    duplicate_ids = frozenset(
        int(recording_id)
        for recording_id in selected.loc[selected["id"].duplicated(keep=False), "id"]
    )
    duplicate_filenames = frozenset(
        selected.loc[selected["file-name"].duplicated(keep=False), "file-name"].dropna().astype(str)
    )
    downloaded = {path.name for path in audio_folder.rglob("*") if path.is_file()} if audio_folder.exists() else set()
    missing = selected_filenames - downloaded
    extra_audio = {filename for filename in downloaded - selected_filenames
                   if Path(filename).suffix.lower() in AUDIO_EXTENSIONS}
    return ValidationReport(
        len(selected),
        len(selected_filenames),
        len(selected_filenames & downloaded),
        frozenset(missing),
        frozenset(extra_audio),
        duplicate_ids,
        duplicate_filenames,
    )
