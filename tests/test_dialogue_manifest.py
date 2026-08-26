import unittest

from tools.dialogue_manifest import (
    DATASET_METADATA_FIELDS,
    DialogueArtifacts,
    ManifestRowError,
    SequencePlanError,
    SequenceRow,
    backchannel_block,
    manifest_row,
    room_tone_block,
    sequence_plan,
)


def build_report(**overrides):
    report = {
        "sequences": {
            "group_size": 1,
            "splits": {
                "train": {
                    "entries": [
                        {
                            "name": "train-seq-001",
                            "dialogues": ["v-001"],
                            "sha256": "a" * 64,
                            "identical_to_dialogue": True,
                        }
                    ]
                },
                "dev": {
                    "entries": [
                        {
                            "name": "dev-seq-001",
                            "dialogues": ["v-016"],
                            "sha256": "b" * 64,
                            "identical_to_dialogue": True,
                        }
                    ]
                },
            },
        }
    }
    report["sequences"].update(overrides)
    return report


class SequencePlanTests(unittest.TestCase):
    """`group_size=1` is what lets a row be named after its dialogue.

    The basename becomes the `dialogue_id` (`train/v-001`), and that namespace is what joins
    the parquet to the manifest and to M3's rows. A row holding four dialogues has no single
    name to take, so the plan refuses rather than picking one.
    """

    def test_a_one_to_one_report_becomes_a_plan(self) -> None:
        plan = sequence_plan(build_report())

        self.assertEqual(
            sorted((row.split, row.name, row.dialogue) for row in plan),
            [("dev", "dev-seq-001", "v-016"), ("train", "train-seq-001", "v-001")],
        )

    def test_a_grouped_report_is_refused(self) -> None:
        report = build_report(group_size=4)

        with self.assertRaises(SequencePlanError) as raised:
            sequence_plan(report)

        self.assertIn("group_size", str(raised.exception))

    def test_a_row_holding_two_dialogues_is_refused(self) -> None:
        report = build_report()
        report["sequences"]["splits"]["train"]["entries"][0]["dialogues"] = ["v-001", "v-002"]

        with self.assertRaises(SequencePlanError) as raised:
            sequence_plan(report)

        self.assertIn("2 dialogues in one row", str(raised.exception))

    def test_a_row_the_builder_did_not_mark_identical_is_refused(self) -> None:
        report = build_report()
        report["sequences"]["splits"]["train"]["entries"][0]["identical_to_dialogue"] = False

        with self.assertRaises(SequencePlanError) as raised:
            sequence_plan(report)

        self.assertIn("did not mark it identical", str(raised.exception))

    def test_a_row_with_no_sha256_is_refused(self) -> None:
        report = build_report()
        report["sequences"]["splits"]["train"]["entries"][0].pop("sha256")

        with self.assertRaises(SequencePlanError):
            sequence_plan(report)

    def test_one_dialogue_in_two_rows_is_refused(self) -> None:
        """Otherwise it would be copied into both splits and evaluated on its own training row."""
        report = build_report()
        report["sequences"]["splits"]["dev"]["entries"][0]["dialogues"] = ["v-001"]

        with self.assertRaises(SequencePlanError) as raised:
            sequence_plan(report)

        self.assertIn("is also row", str(raised.exception))

    def test_every_problem_is_collected_before_raising(self) -> None:
        """A plan that stops at the first bad entry describes one dialogue out of eighty."""
        report = build_report()
        report["sequences"]["splits"]["train"]["entries"][0]["dialogues"] = []
        report["sequences"]["splits"]["dev"]["entries"][0]["identical_to_dialogue"] = False

        with self.assertRaises(SequencePlanError) as raised:
            sequence_plan(report)

        message = str(raised.exception)
        self.assertIn("train-seq-001", message)
        self.assertIn("dev-seq-001", message)

    def test_a_report_with_no_sequences_block_is_refused(self) -> None:
        with self.assertRaises(SequencePlanError):
            sequence_plan({})


class RoomToneBlockTests(unittest.TestCase):
    def test_the_block_counts_what_the_index_holds(self) -> None:
        block = room_tone_block(
            {
                "index": [{}, {}, {}],
                "total_seconds": 13.25,
                "sources_used": ["a", "b"],
                "excluded_held_out": ["VOICEACTRESS100_026"],
            },
            pool="m3r/roomtone",
            index_sha256="a" * 64,
            segments_sha256="b" * 64,
        )

        self.assertEqual(block["segments"], 3)
        self.assertEqual(block["sources"], 2)
        self.assertEqual(block["held_out_excluded"], ["VOICEACTRESS100_026"])


class BackchannelBlockTests(unittest.TestCase):
    def test_a_dialogue_without_one_gets_none(self) -> None:
        """None and an empty object are different claims: "none" against "not recorded"."""
        self.assertIsNone(backchannel_block(None))

    def test_a_block_without_a_measured_checksum_is_refused(self) -> None:
        with self.assertRaises(ManifestRowError):
            backchannel_block({"dialogue_id": "v-001"}, path="x.wav", sha256=None)

    def test_the_block_carries_the_seed(self) -> None:
        block = backchannel_block(
            {
                "dialogue_id": "v-001",
                "text": "はい、はい。",
                "seconds": 0.99,
                "seed": 2833438582,
                "turn_index": 2,
            },
            path="m3r/turns-B-backchannel/v-001-t2-B.wav",
            sha256="c" * 64,
        )

        self.assertEqual(block["seed"], 2833438582)
        self.assertEqual(block["turn_index"], 2)


METADATA = {field: f"{field}-value" for field in DATASET_METADATA_FIELDS}


def artifacts(**overrides):
    defaults = {
        "path": "m3r/v-real/audio/v-001.wav",
        "sha256": "a" * 64,
        "byte_size": 2114928,
        "audio": {"channels": 2, "sample_rate_hz": 24000},
        "split_audio_path": "m3r/v-real/train/audio/v-001.wav",
        "sequence_audio_path": "m3r/v-real/sequences/train/audio/train-seq-001.wav",
        "word_transcript_path": "m3r/v-real/train/text/v-001.json",
        "word_transcript_sha256": "b" * 64,
        "tok_audio_sha256": "c" * 64,
        "tok_text_sha256": "d" * 64,
        "parquet_path": "m3r/v-real/parquet/train-001-of-001.parquet",
        "parquet_sha256": "e" * 64,
    }
    return DialogueArtifacts(**{**defaults, **overrides})


def dialogue(**overrides):
    return {
        "dialogue_id": "v-001",
        "source_artifact_id": "tsukuyomi-corpus-v1:VOICEACTRESS100_001",
        "split": "train",
        "turns": [
            {"speaker": "B", "text": "こんにちは。", "role": "open"},
            {"speaker": "A", "text": "はい。", "role": "body"},
        ],
        **overrides,
    }


SEQUENCE = SequenceRow(
    split="train", name="train-seq-001", dialogue="v-001", sha256="a" * 64, group_size=1
)


def row(**overrides):
    kwargs = {
        "dialogue": dialogue(),
        "split": "train",
        "sequence": SEQUENCE,
        "artifacts": artifacts(),
        "dataset_id": "v-real-v2",
        "metadata": METADATA,
        "room_tone": {"pool": "m3r/roomtone"},
        "backchannel": None,
    }
    kwargs.update(overrides)
    return manifest_row(**kwargs)


class ManifestRowTests(unittest.TestCase):
    def test_the_dialogue_id_namespace_is_split_slash_group(self) -> None:
        """`train/v-001` is what joins this parquet to the manifest and to M3's rows."""
        self.assertEqual(row()["tokenized"]["dialogue_id"], "train/v-001")

    def test_the_row_carries_every_measured_checksum(self) -> None:
        tokenized = row()["tokenized"]

        self.assertEqual(tokenized["tok_audio_npz_sha256"], "c" * 64)
        self.assertEqual(tokenized["tok_text_npz_sha256"], "d" * 64)
        self.assertEqual(tokenized["parquet_sha256"], "e" * 64)

    def test_the_derivation_starts_with_the_corpus_sentence(self) -> None:
        derivation = row(extra_derivation=["irodori-tts-600m-v3-voicedesign"])["derivation"]

        self.assertEqual(
            derivation,
            ["tsukuyomi-corpus-v1:VOICEACTRESS100_001", "irodori-tts-600m-v3-voicedesign"],
        )

    def test_the_text_joins_every_turn(self) -> None:
        self.assertEqual(row()["text"], "こんにちは。 はい。")

    def test_the_turn_roles_are_kept_in_order(self) -> None:
        self.assertEqual(row()["turn_roles"], ["open", "body"])

    def test_a_split_the_script_disagrees_with_is_refused(self) -> None:
        """Three sources name the split; disagreeing puts one dialogue in two places."""
        with self.assertRaises(ManifestRowError) as raised:
            row(dialogue=dialogue(split="dev"))

        self.assertIn("the split map says", str(raised.exception))

    def test_a_sequence_row_from_another_split_is_refused(self) -> None:
        elsewhere = SequenceRow(
            split="dev", name="dev-seq-001", dialogue="v-001", sha256="a" * 64, group_size=1
        )

        with self.assertRaises(ManifestRowError):
            row(sequence=elsewhere)

    def test_a_sequence_row_for_another_dialogue_is_refused(self) -> None:
        other = SequenceRow(
            split="train", name="train-seq-002", dialogue="v-002", sha256="a" * 64, group_size=1
        )

        with self.assertRaises(ManifestRowError):
            row(sequence=other)

    def test_a_wav_that_differs_from_the_build_report_is_refused(self) -> None:
        """The row would otherwise carry a checksum for a file the report never saw."""
        with self.assertRaises(ManifestRowError) as raised:
            row(artifacts=artifacts(sha256="f" * 64))

        self.assertIn("the build report recorded", str(raised.exception))

    def test_a_turn_with_no_role_is_refused(self) -> None:
        broken = dialogue()
        broken["turns"][1].pop("role")

        with self.assertRaises(ManifestRowError):
            row(dialogue=broken)

    def test_missing_dataset_metadata_is_refused(self) -> None:
        with self.assertRaises(ManifestRowError) as raised:
            row(metadata={**METADATA, "license_id": ""})

        self.assertIn("license_id", str(raised.exception))

    def test_a_dialogue_without_a_source_is_refused(self) -> None:
        stripped = dialogue()
        stripped.pop("source_artifact_id")

        with self.assertRaises(ManifestRowError):
            row(dialogue=stripped)

    def test_the_row_holds_every_field_the_validator_requires(self) -> None:
        from tools.experiment_data import REQUIRED_FIELDS

        self.assertEqual(set(REQUIRED_FIELDS) - set(row()), set())


if __name__ == "__main__":
    unittest.main()
