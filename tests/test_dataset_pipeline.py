"""Offline tests for collection helpers; no Xeno-canto requests or downloads."""

from pathlib import Path
import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from collect_species import existing_dataset_is_complete, fetch_metadata, main
from preprocessing.dataset_pipeline import (candidate_statistics, choose_replacements, normalize_species_name,
    DownloadReport, ValidationReport, download_recordings, select_recordings, validate_selection,
    verify_metadata_species)


def rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "file": "https://example.test/1", "file-name": "CarolinaWren95.mp3", "q": "A", "playback-used": "no", "length": "0:05", "rec": "r1", "loc": "l1", "type": "song"},
        {"id": 2, "file": "https://example.test/2", "file-name": "XC2.wav", "q": "B", "playback-used": "no", "length": "0:06", "rec": "r2", "loc": "l2", "type": "call"},
        {"id": 3, "file": "https://example.test/3", "file-name": "XC3.wav", "q": "C", "playback-used": "no", "length": "0:20", "rec": "r3", "loc": "l3", "type": "call"},
        {"id": 4, "file": "https://example.test/4", "file-name": "XC4.wav", "q": "A", "playback-used": "yes", "length": "0:20", "rec": "r4", "loc": "l4", "type": "call"},
        {"id": 5, "file": "https://example.test/5", "file-name": "XC5.wav", "q": "A", "playback-used": "no", "length": "0:05", "rec": "r3", "loc": "l3", "type": "call"},
    ])


class DatasetPipelineTests(unittest.TestCase):
    def test_normalization_and_selection_statistics(self) -> None:
        self.assertEqual(normalize_species_name(" BLUE  Jay "), "blue_jay")
        with self.assertRaises(ValueError):
            verify_metadata_species(rows().assign(en="Red-winged Blackbird"), "blue_jay")
        stats = candidate_statistics(rows())
        self.assertEqual((stats.total, stats.quality_usable, stats.non_playback_usable, stats.duration_usable), (5, 4, 3, 3))
        self.assertEqual(len(select_recordings(rows(), target=3, seed=42)), 3)

    def test_replacement_and_exact_filename_validation(self) -> None:
        metadata = rows()
        selected = metadata.iloc[:2].copy()
        updated = choose_replacements(selected, metadata, [1])
        self.assertEqual(len(updated), 2)
        self.assertNotIn(1, set(updated["id"]))
        self.assertEqual(set(updated["id"]), {2, 5})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "XC2.wav").write_bytes(b"audio")
            (root / "nested" / "XC5.wav").write_bytes(b"audio")
            report = validate_selection(updated, root)
            self.assertTrue(report.passed)
            self.assertNotIn("CarolinaWren95.mp3", report.missing_files)

    def test_duplicate_selected_ids_or_filenames_fail_validation(self) -> None:
        selected = pd.concat([rows().iloc[[0]], rows().iloc[[0]]], ignore_index=True)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CarolinaWren95.mp3").write_bytes(b"audio")
            report = validate_selection(selected, root)
        self.assertFalse(report.passed)
        self.assertEqual(report.duplicate_ids, frozenset({1}))
        self.assertEqual(report.duplicate_filenames, frozenset({"CarolinaWren95.mp3"}))

    def test_existing_complete_and_incomplete_detection(self) -> None:
        selected = rows().iloc[:2].copy()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected_path = root / "blue_jay_selected.csv"
            selected.to_csv(selected_path, index=False)
            audio_dir = root / "audio" / "nested"
            audio_dir.mkdir(parents=True)
            for filename in selected["file-name"]:
                (audio_dir / filename).write_bytes(b"audio")
            self.assertTrue(existing_dataset_is_complete(selected_path, root / "audio", 2))
            (audio_dir / "XC2.wav").unlink()
            self.assertFalse(existing_dataset_is_complete(selected_path, root / "audio", 2))

    def test_download_success_and_multiple_failures_are_reported(self) -> None:
        class SuccessfulResponse:
            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"original-audio-bytes"

        with TemporaryDirectory() as temporary:
            selected = rows().iloc[:1].copy()
            with patch("preprocessing.dataset_pipeline.requests.get", return_value=SuccessfulResponse()):
                report = download_recordings(selected, Path(temporary), workers=1)
            self.assertEqual(report.downloaded, 1)
            self.assertEqual(report.failures, ())

        failed = rows().iloc[:2].copy()
        with TemporaryDirectory() as temporary:
            with patch("preprocessing.dataset_pipeline.requests.get", side_effect=OSError("server unavailable")), \
                 patch("preprocessing.dataset_pipeline.time.sleep"):
                report = download_recordings(failed, Path(temporary), workers=2)
        self.assertEqual(set(report.failures), {1, 2})

    def test_metadata_collection_uses_xcapi_without_exposing_a_key(self) -> None:
        with TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "blue_jay"
            raw_dir.mkdir()

            def create_metadata(command, check):
                self.assertNotIn("--api_key", command)
                rows().assign(en="Blue Jay").to_csv(raw_dir / "metadata_only.csv", index=False)

            with patch("collect_species.load_dotenv") as load_environment, \
                 patch("collect_species.shutil.which", return_value="xcapi"), \
                 patch("collect_species.subprocess.run", side_effect=create_metadata) as run:
                returned = fetch_metadata("Blue Jay", raw_dir)

            self.assertEqual(returned, raw_dir / "metadata_only.csv")
            load_environment.assert_called_once()
            self.assertEqual(run.call_args.args[0][0], "xcapi")

    def test_no_replacement_available_is_actionable(self) -> None:
        metadata = rows().iloc[:2].copy()
        with self.assertRaisesRegex(RuntimeError, "No valid replacement"):
            choose_replacements(metadata, metadata, [1])

    def test_orchestrator_replaces_a_failed_id_without_retrying_it(self) -> None:
        complete = ValidationReport(2, 2, 2, frozenset(), frozenset(), frozenset(), frozenset())
        with TemporaryDirectory() as temporary:
            previous_directory = Path.cwd()
            os.chdir(temporary)
            try:
                def fake_metadata(_species, raw_dir):
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    path = raw_dir / "metadata_only.csv"
                    rows().assign(en="Blue Jay").to_csv(path, index=False)
                    return path

                reports = [DownloadReport(1, 0, (1,)), DownloadReport(1, 1, ())]
                with patch("collect_species.fetch_metadata", side_effect=fake_metadata), \
                     patch("collect_species.download_recordings", side_effect=reports) as download, \
                     patch("collect_species.validate_selection", return_value=complete), \
                     patch("sys.argv", ["collect_species.py", "Blue Jay", "--recordings", "2"]):
                    self.assertEqual(main(), 0)

                self.assertEqual(download.call_count, 2)
                second_selection = download.call_args_list[1].args[0]
                self.assertNotIn(1, set(second_selection["id"]))
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
