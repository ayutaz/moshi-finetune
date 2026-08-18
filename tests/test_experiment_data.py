import hashlib
import json
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from tools.experiment_data import (
    ManifestValidationError,
    build_manifest,
    load_voiceactress_transcripts,
    validate_manifest,
)


def _write_wav(path: Path, *, sample_rate: int = 24_000, frames: int = 2_400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_float_wav(path: Path, *, sample_rate: int = 96_000, frames: int = 960) -> None:
    data = b"\x00\x00\x00\x00" * frames
    byte_rate = sample_rate * 4
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        3,
        1,
        sample_rate,
        byte_rate,
        4,
        32,
        b"data",
        len(data),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


class BuildManifestTests(unittest.TestCase):
    def test_build_manifest_is_deterministic_and_keeps_groups_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            source_dir = data_root / "corpus"
            for index in range(10):
                _write_wav(source_dir / f"dialogue-{index:02d}.wav", frames=2_400 + index)

            metadata = {
                "dataset_id": "tsukuyomi-corpus-v1",
                "source_url": "https://example.test/corpus.zip",
                "source_version": "v1",
                "retrieved_at": "2026-08-18T00:00:00+09:00",
                "license_id": "custom-tsukuyomi-corpus",
                "license_url": "https://example.test/terms",
                "credit": "Tsukuyomi-chan Corpus (CV. Rei Yumesaki)",
                "redistribution": "prohibited",
                "generation_method": "recorded-human",
            }

            first = build_manifest(
                data_root=data_root,
                source_dir=source_dir,
                metadata=metadata,
                seed="m1-2026-08-18",
            )
            second = build_manifest(
                data_root=data_root,
                source_dir=source_dir,
                metadata=metadata,
                seed="m1-2026-08-18",
            )

            self.assertEqual(first, second)
            self.assertEqual({row["split"] for row in first}, {"train", "dev", "test"})
            self.assertEqual(
                {split: sum(row["split"] == split for row in first) for split in {"train", "dev", "test"}},
                {"train": 8, "dev": 1, "test": 1},
            )
            self.assertTrue(all(row["audio"]["sample_rate_hz"] == 24_000 for row in first))
            self.assertTrue(all(row["sha256"] == _sha256(data_root / row["path"]) for row in first))


class ValidateManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.audio_path = self.data_root / "raw" / "sample.wav"
        _write_wav(self.audio_path)
        self.valid_row = {
            "schema_version": 1,
            "artifact_id": "tsukuyomi-corpus-v1:sample",
            "dataset_id": "tsukuyomi-corpus-v1",
            "path": "raw/sample.wav",
            "media_type": "audio/wav",
            "byte_size": self.audio_path.stat().st_size,
            "sha256": _sha256(self.audio_path),
            "source_url": "https://example.test/corpus.zip",
            "source_version": "v1",
            "retrieved_at": "2026-08-18T00:00:00+09:00",
            "license_id": "custom-tsukuyomi-corpus",
            "license_url": "https://example.test/terms",
            "credit": "Tsukuyomi-chan Corpus (CV. Rei Yumesaki)",
            "redistribution": "prohibited",
            "generation_method": "recorded-human",
            "derivation": [],
            "group_id": "sample",
            "split": "train",
            "text": "本日は晴天なり。",
            "audio": {
                "sample_rate_hz": 24_000,
                "channels": 1,
                "sample_width_bytes": 2,
                "frame_count": 2_400,
                "duration_seconds": 0.1,
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_manifest_returns_a_machine_readable_summary(self) -> None:
        summary = validate_manifest([self.valid_row], data_root=self.data_root)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["artifact_count"], 1)
        self.assertEqual(summary["coeiroink_artifact_count"], 0)
        self.assertEqual(summary["corrupt_audio_count"], 0)
        self.assertEqual(summary["cross_split_duplicate_count"], 0)

    def test_checksum_mismatch_fails(self) -> None:
        row = dict(self.valid_row, sha256="0" * 64)

        with self.assertRaisesRegex(ManifestValidationError, "checksum"):
            validate_manifest([row], data_root=self.data_root)

    def test_coeiroink_provenance_fails_even_when_flag_is_not_present(self) -> None:
        row = dict(self.valid_row, generation_method="COEIROINK-v2")

        with self.assertRaisesRegex(ManifestValidationError, "COEIROINK"):
            validate_manifest([row], data_root=self.data_root)

    def test_same_group_across_splits_fails(self) -> None:
        second_path = self.data_root / "raw" / "second.wav"
        _write_wav(second_path, frames=2_401)
        second = dict(
            self.valid_row,
            artifact_id="tsukuyomi-corpus-v1:second",
            path="raw/second.wav",
            byte_size=second_path.stat().st_size,
            sha256=_sha256(second_path),
            split="test",
        )

        with self.assertRaisesRegex(ManifestValidationError, "group_id"):
            validate_manifest([self.valid_row, second], data_root=self.data_root)

    def test_exact_duplicate_within_one_split_fails(self) -> None:
        second_path = self.data_root / "raw" / "second.wav"
        shutil.copyfile(self.audio_path, second_path)
        second = dict(
            self.valid_row,
            artifact_id="tsukuyomi-corpus-v1:second",
            path="raw/second.wav",
            group_id="second",
        )

        with self.assertRaisesRegex(ManifestValidationError, "exact duplicate"):
            validate_manifest([self.valid_row, second], data_root=self.data_root)

    def test_near_duplicate_text_across_splits_fails(self) -> None:
        second_path = self.data_root / "raw" / "second.wav"
        _write_wav(second_path, frames=2_401)
        second = dict(
            self.valid_row,
            artifact_id="tsukuyomi-corpus-v1:second",
            path="raw/second.wav",
            byte_size=second_path.stat().st_size,
            sha256=_sha256(second_path),
            group_id="second",
            split="test",
            text="本日は、晴天なり！",
        )

        with self.assertRaisesRegex(ManifestValidationError, "near-duplicate"):
            validate_manifest([self.valid_row, second], data_root=self.data_root)

    def test_corrupt_wave_fails(self) -> None:
        self.audio_path.write_bytes(b"not a wave file")
        row = dict(
            self.valid_row,
            byte_size=self.audio_path.stat().st_size,
            sha256=_sha256(self.audio_path),
        )

        with self.assertRaisesRegex(ManifestValidationError, "audio"):
            validate_manifest([row], data_root=self.data_root)

    def test_ieee_float_wave_used_by_the_official_corpus_is_supported(self) -> None:
        _write_float_wav(self.audio_path)
        row = dict(
            self.valid_row,
            byte_size=self.audio_path.stat().st_size,
            sha256=_sha256(self.audio_path),
            audio={
                "sample_rate_hz": 96_000,
                "channels": 1,
                "sample_width_bytes": 4,
                "frame_count": 960,
                "duration_seconds": 0.01,
            },
        )

        summary = validate_manifest([row], data_root=self.data_root)

        self.assertEqual(summary["corrupt_audio_count"], 0)

    def test_missing_rights_metadata_fails(self) -> None:
        row = dict(self.valid_row)
        del row["credit"]

        with self.assertRaisesRegex(ManifestValidationError, "credit"):
            validate_manifest([row], data_root=self.data_root)


class ManifestJsonTests(unittest.TestCase):
    def test_rows_remain_jsonl_serializable(self) -> None:
        payload = {"schema_version": 1, "text": "ごきげんよう。"}
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(json.loads(serialized), payload)

    def test_official_transcript_parser_requires_and_maps_all_100_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = Path(temp_dir) / "transcript.txt"
            transcript_path.write_text(
                "\n".join(
                    f"VOICEACTRESS100_{index:03d}:発話文{index}。" for index in range(1, 101)
                ),
                encoding="utf-8",
            )

            transcripts = load_voiceactress_transcripts(transcript_path)

            self.assertEqual(len(transcripts), 100)
            self.assertEqual(transcripts["VOICEACTRESS100_001"], "発話文1。")


if __name__ == "__main__":
    unittest.main()
