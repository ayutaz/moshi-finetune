import hashlib
import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from tools.text_stream_audit import (
    TOKENIZE_TEXT_FLAGS,
    NotADialogueParquetError,
    audit_parquet,
)

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

    def _manifest_paths(self) -> list[Path]:
        """Every manifest, not one named one.

        Naming a single file meant a manifest added later - v-real-v1, v-tts-v1 - was
        invisible to this check, and CLAUDE.md's rule that a dataset is registered before
        it is used went unenforced for it.
        """
        paths = sorted((EXPERIMENT_ROOT / "manifests").glob("*.jsonl"))
        self.assertTrue(paths, "no manifests found; the glob or the directory is wrong")
        return paths

    def test_every_manifest_dataset_is_registered_and_marked_used(self) -> None:
        registries = self._registries()
        for manifest in self._manifest_paths():
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows, f"{manifest.name} is empty")
            for row in rows:
                registry = registries.get(row["dataset_id"])
                self.assertIsNotNone(
                    registry, f"{manifest.name}: {row['dataset_id']} has no registry entry"
                )
                self.assertNotEqual(
                    registry.get("used_in_experiment"),
                    False,
                    f"{manifest.name}: {row['dataset_id']} is marked unused but appears in a manifest",
                )

    def test_no_manifest_row_derives_from_a_held_out_corpus_utterance(self) -> None:
        """M3 condition 4 is measured on held-out audio, so held-out audio cannot be trained on.

        Every V dialogue quotes one corpus sentence. If a dev or test sentence reaches a
        training row, the held-out speaker-likeness comparison is measuring audio the model
        already heard, and nothing downstream would notice.
        """
        corpus_split = {
            row["artifact_id"]: row["split"]
            for row in (
                json.loads(line)
                for line in (EXPERIMENT_ROOT / "manifests" / "tsukuyomi-corpus-v1.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }

        for manifest in self._manifest_paths():
            if manifest.name == "tsukuyomi-corpus-v1.jsonl":
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                parents = row.get("derivation") or []
                self.assertTrue(
                    parents,
                    f"{manifest.name}: {row.get('artifact_id')} records no derivation, so "
                    "leakage cannot be checked",
                )
                for parent in parents:
                    parent_id = parent if isinstance(parent, str) else parent.get("artifact_id")
                    split = corpus_split.get(parent_id)
                    if split is None:
                        continue
                    self.assertEqual(
                        split,
                        "train",
                        f"{manifest.name}: {row.get('artifact_id')} derives from {parent_id}, "
                        f"which is in the corpus {split} split",
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


class TokenizeFlagRecordTests(unittest.TestCase):
    """M3 tokenized without `--no_whitespace_before_word` and nobody could tell afterwards.

    The flag is `store_true`, so a run that omits it and a run that was never written down
    look identical, and no artifact it touches - parquet, row counts, startup log - looks
    any different. The eighteen-step review before M3 billed a GPU found thirteen defects
    and missed this one for exactly that reason.

    So every dialogue dataset carries a sidecar in which every flag appears with its
    resolved value, and these tests refuse a dataset that has none, a sidecar that leaves
    the flag unstated, and a sidecar that states `false` without saying that it is a
    defect. Standard library only: CI installs nothing but pytest, so this is the layer
    that will actually run when the next dataset arrives.
    """

    #  The corpus manifest lists raw recordings, which are never tokenized. Every other
    #  manifest in this directory describes a dialogue dataset built for training.
    RAW_CORPUS_MANIFEST = "tsukuyomi-corpus-v1.jsonl"

    def _dialogue_manifests(self) -> list[Path]:
        paths = [
            path
            for path in sorted((EXPERIMENT_ROOT / "manifests").glob("*.jsonl"))
            if path.name != self.RAW_CORPUS_MANIFEST
        ]
        self.assertTrue(paths, "no dialogue manifests found; the glob or the directory is wrong")
        return paths

    def _sidecars(self) -> list[tuple[Path, Path, dict]]:
        records = []
        for manifest in self._dialogue_manifests():
            sidecar = manifest.with_name(f"{manifest.stem}-tokenize.json")
            self.assertTrue(
                sidecar.is_file(),
                f"{manifest.name} has no {sidecar.name}; run "
                "`python -m tools.text_stream_audit record-tokenize` after tokenizing",
            )
            records.append((manifest, sidecar, json.loads(sidecar.read_text(encoding="utf-8"))))
        return records

    def test_every_dialogue_manifest_records_how_it_was_tokenized(self) -> None:
        for manifest, sidecar, record in self._sidecars():
            self.assertEqual(
                (REPOSITORY_ROOT / record["manifest"]).resolve(),
                manifest.resolve(),
                f"{sidecar.name}: points at another manifest",
            )
            self.assertTrue(record.get("invocations"), f"{sidecar.name}: no invocations")
            self.assertIn(record.get("provenance"), {"recorded", "reconstructed"}, sidecar.name)

    def test_every_invocation_states_every_flag_explicitly(self) -> None:
        """A flag left out of the record is a flag nobody can rule out later."""
        for _, sidecar, record in self._sidecars():
            for invocation in record["invocations"]:
                flags = invocation.get("flags", {})
                self.assertEqual(
                    set(flags),
                    set(TOKENIZE_TEXT_FLAGS),
                    f"{sidecar.name}: the flag record is incomplete",
                )
                self.assertIsInstance(
                    flags["no_whitespace_before_word"],
                    bool,
                    f"{sidecar.name}: the whitespace flag is not a stated boolean",
                )

    def test_dropping_the_whitespace_flag_requires_a_written_defect(self) -> None:
        """README-ja.md:109 and README.md:112 both require it for Japanese.

        Shipping without it stays possible - the four M3 parquets already did - but only
        by writing down that the dataset is defective and what replaces it. A sidecar that
        says `false` and nothing else fails here.
        """
        for _, sidecar, record in self._sidecars():
            dropped = [
                invocation
                for invocation in record["invocations"]
                if invocation["flags"]["no_whitespace_before_word"] is False
            ]
            if not dropped:
                continue
            defect = record.get("known_defect")
            self.assertIsNotNone(
                defect,
                f"{sidecar.name}: tokenized without --no_whitespace_before_word and declares "
                "no known_defect",
            )
            self.assertEqual(defect.get("flag"), "no_whitespace_before_word", sidecar.name)
            self.assertTrue(defect.get("report"), f"{sidecar.name}: the defect cites no report")
            self.assertTrue(
                (REPOSITORY_ROOT / defect["report"]).is_file(),
                f"{sidecar.name}: {defect['report']} does not exist",
            )
            self.assertTrue(
                defect.get("superseded_by"), f"{sidecar.name}: nothing supersedes the defect"
            )

    def test_a_reconstructed_record_says_how_it_was_reconstructed(self) -> None:
        """A reconstruction is a claim about the past, and claims carry their evidence.

        M3 left no command line anywhere, so its two sidecars are reconstructions. What
        makes them usable is that the flag values were re-derived by re-running the
        tokenizer against the shipped output, not remembered.
        """
        for _, sidecar, record in self._sidecars():
            if record["provenance"] != "reconstructed":
                continue
            evidence = record.get("evidence", {})
            self.assertTrue(evidence.get("why_reconstructed"), f"{sidecar.name}: no reason given")
            self.assertTrue(
                evidence.get("how_the_flags_were_established"),
                f"{sidecar.name}: the flag values are asserted with no evidence",
            )


class TextStreamGateTests(unittest.TestCase):
    """The share of emitted text tokens that are a bare `U+2581` and nothing else.

    Without `--no_whitespace_before_word`, `tools/tokenize_text.py` puts a space in front
    of every pyopenjtalk word and SentencePiece turns each one into that marker. In the
    four parquets M3 shipped it is 44.6% to 46.6% of everything the tokenizer wrote; the
    same scripts re-tokenized with the flag give 0.0%.

    The threshold is calibrated in `reports/m3-text-stream-audit.json` from measurements
    of both paths, and these tests hold it there: it may not drift above a value a
    compliant dataset produces, nor below one a broken dataset produces. The shipped
    parquets are quarantined by name rather than let through by a looser threshold,
    because a threshold raised until the candidate passes is no threshold at all.
    """

    def _audit(self) -> dict:
        return json.loads(
            (EXPERIMENT_ROOT / "reports" / "m3-text-stream-audit.json").read_text(encoding="utf-8")
        )

    def _shipped_parquets(self) -> list[Path]:
        """Any depth, not `m3/<arm>/parquet/`.

        A gate that only looks where the last dataset happened to sit is a gate the next
        dataset walks past. Files that are not Moshi dialogue parquets are recognised and
        left alone by the measurement below rather than by this glob.
        """
        data_root = REPOSITORY_ROOT / "data" / "experiments" / "tsukuyomi_ojousama"
        return sorted(data_root.rglob("*.parquet"))

    def test_the_threshold_sits_between_the_calibrated_poles(self) -> None:
        """Neither pole may cross the line, in either direction.

        Raising the threshold to admit a flag-dropped dataset fails here, and so does
        lowering it to where a compliant dataset would go red.
        """
        gate = self._audit()["gate"]
        calibration = gate["calibration"]
        self.assertTrue(calibration["compliant"], "no compliant pole recorded")
        self.assertTrue(calibration["non_compliant"], "no non-compliant pole recorded")

        for pole in calibration["compliant"]:
            self.assertLess(
                pole["share"],
                gate["max_share"],
                f"the threshold would fail a compliant dataset: {pole['label']}",
            )
        for pole in calibration["non_compliant"]:
            self.assertGreater(
                pole["share"],
                gate["max_share"],
                f"the threshold would pass a flag-dropped dataset: {pole['label']}",
            )

    def test_every_pole_says_where_it_was_measured(self) -> None:
        calibration = self._audit()["gate"]["calibration"]

        for pole in calibration["compliant"] + calibration["non_compliant"]:
            self.assertTrue(pole.get("source"), f"{pole.get('label')}: no source")
            self.assertGreater(pole["text_tokens"], 0, f"{pole.get('label')}: no tokens counted")

    def test_every_quarantined_parquet_explains_itself(self) -> None:
        """The escape hatch costs a reason and a successor, and shows up in review."""
        audit = self._audit()
        measured = {measurement["path"] for measurement in audit["measurements"]}

        for entry in audit["gate"]["quarantine"]:
            self.assertTrue(entry.get("reason"), f"{entry['path']}: quarantined without a reason")
            self.assertTrue(entry.get("superseded_by"), f"{entry['path']}: nothing supersedes it")
            self.assertTrue(entry.get("sha256"), f"{entry['path']}: no checksum")
            self.assertIn(
                entry["path"], measured, f"{entry['path']}: quarantined but never measured"
            )
            self.assertTrue(
                entry.get("recorded_shares"), f"{entry['path']}: no share recorded to pin"
            )

    def test_every_exemption_from_the_metric_explains_itself(self) -> None:
        """A parquet with no text tokens has an undefined share, not a clean one.

        Returning zero for an empty denominator would hand the best possible score to an
        artifact the metric cannot see. The three eval-prompt sets genuinely carry no text;
        they are named here, and anything else that comes out empty fails below.
        """
        audit = self._audit()
        measured = {measurement["path"] for measurement in audit["measurements"]}

        for entry in audit["gate"]["no_text_stream"]:
            self.assertTrue(entry.get("reason"), f"{entry['path']}: exempted without a reason")
            self.assertEqual(entry.get("text_tokens"), 0, f"{entry['path']}: not actually empty")
            self.assertIn(entry["path"], measured, f"{entry['path']}: exempted but never measured")

    def test_every_shipped_parquet_is_under_the_gate_or_quarantined(self) -> None:
        """The measured half. It runs where the data and the tokenizer both are.

        CI installs only pytest and the parquets are gitignored, so this skips there; the
        declaration tests above are what run in CI. Here it does two things: it fails any
        parquet not on the quarantine list whose text stream is mostly whitespace, and it
        pins each quarantined parquet to the share the report records, so the record
        cannot drift away from the artifact it describes.
        """
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError as error:
            self.skipTest(f"the audit tool's dependencies are absent: {error}")

        parquets = self._shipped_parquets()
        if not parquets:
            self.skipTest("no tokenized parquet is present in this checkout")

        tokenizers = sorted(
            Path.home().glob(
                ".cache/huggingface/hub/models--nu-dialogue--j-moshi-ext/snapshots/*/"
                "tokenizer_spm_32k_3.model"
            )
        )
        if not tokenizers:
            self.skipTest("the j-moshi-ext tokenizer is not in the local hub cache")

        audit = self._audit()
        gate = audit["gate"]
        quarantine = {entry["path"]: entry for entry in gate["quarantine"]}
        end_of_text_padding_id = audit["tokenizer"]["ids"]["end_of_text_padding"]["id"]
        exempt = {entry["path"] for entry in gate["no_text_stream"]}

        for parquet in parquets:
            relative = parquet.relative_to(REPOSITORY_ROOT).as_posix()
            try:
                measurement = audit_parquet(
                    parquet,
                    tokenizers[-1],
                    end_of_text_padding_id=end_of_text_padding_id,
                )
            except NotADialogueParquetError:
                continue  # not written by tools/prepare_dataset.py; it has no text stream

            for speaker, counts in measurement["speakers"].items():
                share = counts["bare_whitespace_share_of_text_tokens"]
                entry = quarantine.get(relative)
                if entry is not None:
                    self.assertAlmostEqual(
                        share,
                        entry["recorded_shares"][speaker],
                        places=9,
                        msg=f"{relative} speaker {speaker}: the quarantine record is stale",
                    )
                    continue
                if counts["text_tokens"] == 0:
                    self.assertIn(
                        relative,
                        exempt,
                        f"{relative} speaker {speaker}: the text stream is empty, so the "
                        "whitespace share is undefined rather than clean. Either this dataset "
                        "should carry text and does not, or it is a prompt set and belongs in "
                        "the report's no_text_stream list with a reason.",
                    )
                    continue
                self.assertLess(
                    share,
                    gate["max_share"],
                    f"{relative} speaker {speaker}: {share:.3f} of the emitted text tokens are a "
                    "bare U+2581. Tokenize with --no_whitespace_before_word "
                    "(README-ja.md:109, README.md:112).",
                )
