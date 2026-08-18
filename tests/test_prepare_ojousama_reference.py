import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_ojousama_reference import prepare_reference


class PrepareOjousamaReferenceTests(unittest.TestCase):
    def test_deduplicates_prompts_and_records_rejected_source_rows(self) -> None:
        rows = [
            {"prompt": "ごきげん？ ->", "completion": " よろしくてよ\n"},
            {"prompt": "ごきげん？ ->", "completion": " 元気ですわ\n"},
            {"prompt": "朝食は？ ->", "completion": " 紅茶ですの\n"},
        ]

        accepted, report = prepare_reference(rows, source_commit="abc123")

        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0]["source_row"], 1)
        self.assertEqual(accepted[0]["usage"], "reference-only")
        self.assertEqual(report["status"], "pass-with-deduplication")
        self.assertEqual(report["duplicate_prompt_count"], 1)
        self.assertEqual(report["rejected_source_rows"], [2])

    def test_output_is_jsonl_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.jsonl"
            accepted, _ = prepare_reference(
                [{"prompt": "相談 ->", "completion": " 承りますわ\n"}],
                source_commit="abc123",
            )
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
                encoding="utf-8",
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
