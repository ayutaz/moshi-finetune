import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "tsukuyomi_ojousama"


class TrackedExperimentAssetTests(unittest.TestCase):
    def test_fixed_evaluation_registry_matches_tracked_files(self) -> None:
        registry = json.loads(
            (EXPERIMENT_ROOT / "registry" / "fixed-evaluation-v1.json").read_text(
                encoding="utf-8"
            )
        )

        for artifact in registry["files"]:
            path = REPOSITORY_ROOT / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(path.stat().st_size, artifact["byte_size"], artifact["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact["sha256"],
                artifact["path"],
            )
            row_count = sum(bool(line.strip()) for line in path.read_text().splitlines())
            self.assertEqual(row_count, artifact["rows"], artifact["path"])

    def test_m1_reports_are_green_and_manifest_has_fixed_split(self) -> None:
        corpus_report = json.loads(
            (EXPERIMENT_ROOT / "reports" / "tsukuyomi-corpus-v1-validation.json").read_text()
        )
        evaluation_report = json.loads(
            (EXPERIMENT_ROOT / "reports" / "fixed-evaluation-validation.json").read_text()
        )
        manifest_rows = [
            json.loads(line)
            for line in (EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]

        self.assertEqual(corpus_report["status"], "pass")
        self.assertEqual(corpus_report["split_counts"], {"train": 80, "dev": 10, "test": 10})
        self.assertEqual(evaluation_report["status"], "pass")
        self.assertEqual(evaluation_report["training_leakage_count"], 0)
        self.assertEqual(len(manifest_rows), 100)


if __name__ == "__main__":
    unittest.main()
