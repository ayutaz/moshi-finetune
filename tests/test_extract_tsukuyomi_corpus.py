import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from tools.extract_tsukuyomi_corpus import CorpusExtractionError, extract_corpus


def _wav_bytes(frame_count: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        with wave.open(temp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return Path(temp_file.name).read_bytes()


class ExtractCorpusTests(unittest.TestCase):
    def test_extracts_first_variant_to_canonical_names_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "corpus.zip"
            output_dir = root / "selected"
            with zipfile.ZipFile(archive, "w") as zip_file:
                for variant, frames in (("01 original", 10), ("02 processed", 20)):
                    for index in range(1, 101):
                        zip_file.writestr(
                            f"corpus/{variant}/VOICEACTRESS100_{index:03d}.wav",
                            _wav_bytes(frames + index),
                        )
                zip_file.writestr(
                    "corpus/04 台本と補足資料/★台本テキスト/01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt",
                    "VOICEACTRESS100_001:テスト。\n",
                )

            first_report = extract_corpus(archive, output_dir)
            second_report = extract_corpus(archive, output_dir)

            self.assertEqual(first_report, second_report)
            self.assertEqual(first_report["selected_file_count"], 100)
            self.assertEqual(first_report["variant_count"], 2)
            self.assertEqual(first_report["documentation_file_count"], 1)
            self.assertEqual(len(list(output_dir.glob("*.wav"))), 100)
            self.assertEqual(
                (output_dir / "documentation" / "corpus-transcript.txt").read_text(
                    encoding="utf-8"
                ),
                "VOICEACTRESS100_001:テスト。\n",
            )
            with wave.open(str(output_dir / "VOICEACTRESS100_001.wav"), "rb") as wav_file:
                self.assertEqual(wav_file.getnframes(), 11)

    def test_rejects_an_archive_without_all_100_utterances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "corpus.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("VOICEACTRESS100_001.wav", _wav_bytes(10))

            with self.assertRaisesRegex(CorpusExtractionError, "100"):
                extract_corpus(archive, root / "selected")


if __name__ == "__main__":
    unittest.main()
