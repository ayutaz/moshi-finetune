import unittest

from tools.evaluation_data import (
    EvaluationValidationError,
    build_voice_evaluation_index,
    validate_fixed_evaluation,
)


def _tts_rows(count: int = 30):
    return [
        {
            "id": f"tts-{index:02d}",
            "text": f"未学習評価文章その{index}を朗読します。",
            "tags": ["clarity"],
        }
        for index in range(1, count + 1)
    ]


def _style_rows(count: int = 50):
    return [
        {
            "id": f"style-{index:02d}",
            "topic": f"topic-{index}",
            "prompt": f"相談内容その{index}について答えてください。",
            "preferred": f"承知いたしましたわ。回答番号は{index}ですの。",
            "dispreferred": f"承知しました。回答番号は{index}です。",
        }
        for index in range(1, count + 1)
    ]


def _general_rows(count: int = 30):
    return [
        {
            "id": f"general-{index:02d}",
            "category": f"category-{index}",
            "user_prompt": f"一般対話課題その{index}に答えてください。",
            "success_criteria": ["質問に直接答える"],
        }
        for index in range(1, count + 1)
    ]


class VoiceEvaluationIndexTests(unittest.TestCase):
    def test_uses_train_for_seen_and_test_for_held_out(self) -> None:
        manifest = []
        for split, count in (("train", 12), ("dev", 2), ("test", 10)):
            for index in range(count):
                manifest.append(
                    {
                        "artifact_id": f"{split}-{index:02d}",
                        "split": split,
                        "text": f"{split} transcript {index}",
                        "sha256": f"{index:064x}",
                    }
                )

        index = build_voice_evaluation_index(manifest)

        self.assertEqual(len(index), 20)
        self.assertEqual(sum(row["partition"] == "seen" for row in index), 10)
        self.assertEqual(sum(row["partition"] == "held-out" for row in index), 10)
        self.assertTrue(
            all(row["source_split"] == "train" for row in index if row["partition"] == "seen")
        )
        self.assertTrue(
            all(row["source_split"] == "test" for row in index if row["partition"] == "held-out")
        )


class FixedEvaluationValidationTests(unittest.TestCase):
    def test_valid_fixed_sets_pass(self) -> None:
        report = validate_fixed_evaluation(
            tts_rows=_tts_rows(),
            style_rows=_style_rows(),
            general_rows=_general_rows(),
            training_manifest=[],
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["tts_count"], 30)
        self.assertEqual(report["style_pair_count"], 50)
        self.assertEqual(report["general_dialogue_count"], 30)
        self.assertEqual(report["training_leakage_count"], 0)

    def test_wrong_cardinality_fails(self) -> None:
        with self.assertRaisesRegex(EvaluationValidationError, "30 TTS"):
            validate_fixed_evaluation(
                tts_rows=_tts_rows(29),
                style_rows=_style_rows(),
                general_rows=_general_rows(),
                training_manifest=[],
            )

    def test_style_preferred_without_marker_fails(self) -> None:
        style_rows = _style_rows()
        style_rows[0] = dict(style_rows[0], preferred="承知しました。")

        with self.assertRaisesRegex(EvaluationValidationError, "style marker"):
            validate_fixed_evaluation(
                tts_rows=_tts_rows(),
                style_rows=style_rows,
                general_rows=_general_rows(),
                training_manifest=[],
            )

    def test_ojousama_invitation_marker_is_accepted(self) -> None:
        style_rows = _style_rows()
        style_rows[0] = dict(style_rows[0], preferred="香り高い紅茶を召し上がりませんこと。")

        report = validate_fixed_evaluation(
            tts_rows=_tts_rows(),
            style_rows=style_rows,
            general_rows=_general_rows(),
            training_manifest=[],
        )

        self.assertEqual(report["status"], "pass")

    def test_training_text_leakage_fails(self) -> None:
        training_manifest = [{"split": "train", "text": _tts_rows()[0]["text"]}]

        with self.assertRaisesRegex(EvaluationValidationError, "training leakage"):
            validate_fixed_evaluation(
                tts_rows=_tts_rows(),
                style_rows=_style_rows(),
                general_rows=_general_rows(),
                training_manifest=training_manifest,
            )


if __name__ == "__main__":
    unittest.main()
