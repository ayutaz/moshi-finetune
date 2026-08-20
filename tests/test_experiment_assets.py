import hashlib
import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "tsukuyomi_ojousama"


class TrackedExperimentAssetTests(unittest.TestCase):
    def test_fixed_evaluation_registry_matches_tracked_files(self) -> None:
        registry = json.loads(
            (EXPERIMENT_ROOT / "registry" / "fixed-evaluation-v1.json").read_text(encoding="utf-8")
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


class EvaluationRegistryCoverageTests(unittest.TestCase):
    """The registry test only walks registry -> file.

    An evaluation file added without a registry entry would carry no checksum and no
    row count, and nothing would notice.
    """

    def test_every_evaluation_file_is_registered(self) -> None:
        registry = json.loads(
            (EXPERIMENT_ROOT / "registry" / "fixed-evaluation-v1.json").read_text(encoding="utf-8")
        )
        registered = {
            (REPOSITORY_ROOT / artifact["path"]).resolve() for artifact in registry["files"]
        }
        on_disk = {path.resolve() for path in (EXPERIMENT_ROOT / "eval").glob("*.jsonl")}

        self.assertEqual(on_disk - registered, set(), "evaluation files missing from the registry")


class VoiceEvaluationConsistencyTests(unittest.TestCase):
    """`tools.evaluation_data validate` never sees the voice set.

    Its rows duplicate the corpus manifest, so they can drift out of agreement silently.
    """

    def _rows(self) -> list[dict]:
        path = EXPERIMENT_ROOT / "eval" / "voice-seen-heldout-20.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _manifest(self) -> dict[str, dict]:
        path = EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl"
        return {
            json.loads(line)["artifact_id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def test_rows_agree_with_the_corpus_manifest(self) -> None:
        manifest = self._manifest()

        for row in self._rows():
            entry = manifest.get(row["artifact_id"])
            self.assertIsNotNone(entry, row["artifact_id"])
            self.assertEqual(row["sha256"], entry["sha256"], row["id"])
            self.assertEqual(row["text"], entry["text"], row["id"])
            self.assertEqual(row["source_split"], entry["split"], row["id"])

    def test_seen_comes_from_train_and_held_out_from_test(self) -> None:
        expected_split = {"seen": "train", "held-out": "test"}
        counts = {"seen": 0, "held-out": 0}

        for row in self._rows():
            self.assertEqual(row["source_split"], expected_split[row["partition"]], row["id"])
            counts[row["partition"]] += 1

        self.assertEqual(counts, {"seen": 10, "held-out": 10})

    def test_ids_and_artifacts_are_unique(self) -> None:
        rows = self._rows()

        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertEqual(len({row["artifact_id"] for row in rows}), len(rows))


class RedistributionComplianceTests(unittest.TestCase):
    """`reference/ojousama-talk-script-201.jsonl` is committed to a public repository.

    MIT requires the permission notice to travel with a redistributed derivative, so the
    copyright line in DATA_CREDITS.md is not enough on its own.
    """

    def test_bundled_licence_matches_the_registered_checksum(self) -> None:
        registry = json.loads(
            (EXPERIMENT_ROOT / "registry" / "ojousama-talk-script-dataset.json").read_text(
                encoding="utf-8"
            )
        )
        bundled = registry["bundled_license"]
        path = REPOSITORY_ROOT / bundled["path"]

        self.assertTrue(path.is_file(), bundled["path"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), bundled["sha256"])

    def test_bundled_licence_carries_the_permission_notice(self) -> None:
        registry = json.loads(
            (EXPERIMENT_ROOT / "registry" / "ojousama-talk-script-dataset.json").read_text(
                encoding="utf-8"
            )
        )
        text = (REPOSITORY_ROOT / registry["bundled_license"]["path"]).read_text(encoding="utf-8")

        self.assertIn("Permission is hereby granted", text)
        self.assertIn(registry["credit"], text)

    def test_every_registered_source_is_credited(self) -> None:
        credits = (EXPERIMENT_ROOT / "DATA_CREDITS.md").read_text(encoding="utf-8")

        for path in sorted((EXPERIMENT_ROOT / "registry").glob("*.json")):
            registry = json.loads(path.read_text(encoding="utf-8"))
            if registry.get("used_in_experiment") is False:
                continue  # a source that was never obtained needs no credit line yet
            source_url = registry.get("source_url", "")
            if not source_url.startswith("http"):
                continue  # artifacts produced in this repository need no upstream credit
            # The credit line may point at the upstream's distribution page rather than the
            # exact download URL, so the host is what has to appear.
            host = urlparse(source_url).netloc
            self.assertIn(host, credits, f"{path.name}: {source_url} is not credited")


class RawCorpusIntegrityTests(unittest.TestCase):
    """The raw audio is not redistributable, so it is absent from a fresh clone."""

    def test_every_manifest_row_matches_its_audio_file(self) -> None:
        data_root = REPOSITORY_ROOT / "data" / "experiments" / "tsukuyomi_ojousama"
        if not data_root.is_dir():
            self.skipTest("raw corpus audio is not present in this checkout")

        rows = [
            json.loads(line)
            for line in (EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in rows:
            path = data_root / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(path.stat().st_size, row["byte_size"], row["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"], row["path"]
            )


class DataProvenanceCoverageTests(unittest.TestCase):
    """Every dataset the experiment consumes must be registered before it is used.

    An application-only source such as `tsukuyomi-yoseatsume` is registered with
    `used_in_experiment: false`, so it carries documented terms without any manifest row
    depending on terms that were never accepted.
    """

    def _registries(self) -> dict[str, dict]:
        registries = {}
        for path in sorted((EXPERIMENT_ROOT / "registry").glob("*.json")):
            registry = json.loads(path.read_text(encoding="utf-8"))
            registries[registry["dataset_id"]] = registry
        return registries

    def test_every_manifest_dataset_is_registered_and_marked_used(self) -> None:
        registries = self._registries()
        rows = [
            json.loads(line)
            for line in (EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

        for row in rows:
            registry = registries.get(row["dataset_id"])
            self.assertIsNotNone(registry, f"{row['dataset_id']} has no registry entry")
            self.assertNotEqual(
                registry.get("used_in_experiment"),
                False,
                f"{row['dataset_id']} is marked unused but appears in the manifest",
            )

    def test_every_registry_records_its_terms_and_provenance(self) -> None:
        for dataset_id, registry in self._registries().items():
            self.assertTrue(registry.get("source_url"), f"{dataset_id}: no source_url")
            self.assertTrue(registry.get("source_version"), f"{dataset_id}: no source_version")
            has_terms = registry.get("license_id") or registry.get("terms")
            self.assertTrue(has_terms, f"{dataset_id}: no licence or terms recorded")

    def test_an_unobtained_source_declares_why_it_is_excluded(self) -> None:
        for dataset_id, registry in self._registries().items():
            if registry.get("used_in_experiment") is not False:
                continue
            self.assertIn("decision", registry, f"{dataset_id}: no exclusion decision")
            self.assertTrue(registry["decision"].get("rationale"), f"{dataset_id}: no rationale")
            self.assertTrue(
                registry["decision"].get("reopen_when"), f"{dataset_id}: no reopen condition"
            )
