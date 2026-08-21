import unittest

from tools.dialogue_scripts import (
    TimelineSpec,
    project_dialogue_seconds,
    project_frames,
    project_seconds,
    validate_scripts,
)

SPEC = TimelineSpec(lead_in_seconds=0.5, gap_seconds=0.4, frame_rate_hz=12.5)


class ProjectionTests(unittest.TestCase):
    def test_seconds_are_mora_over_rate(self) -> None:
        self.assertAlmostEqual(project_seconds(60, mora_per_second=6.0), 10.0)

    def test_a_faster_rate_shortens_the_projection(self) -> None:
        # The floor gate must assume the FAST end: if the voice speaks faster than
        # projected, the dialogue lands shorter than planned and can slip under the floor.
        fast = project_seconds(60, mora_per_second=8.0)
        slow = project_seconds(60, mora_per_second=5.0)
        self.assertLess(fast, slow)

    def test_a_non_positive_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_seconds(60, mora_per_second=0.0)

    def test_frames_round_down_so_the_gate_never_overstates(self) -> None:
        self.assertEqual(project_frames(1.99, frame_rate_hz=12.5), 24)

    def test_a_three_turn_dialogue_sums_turns_gaps_and_lead_in(self) -> None:
        # 0.5 lead-in + 2.0 + 0.4 + 3.0 + 0.4 + 2.0
        total = project_dialogue_seconds([2.0, 3.0, 2.0], spec=SPEC)
        self.assertAlmostEqual(total, 8.3)

    def test_a_single_turn_dialogue_has_no_gaps(self) -> None:
        self.assertAlmostEqual(project_dialogue_seconds([3.0], spec=SPEC), 3.5)

    def test_an_empty_dialogue_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_dialogue_seconds([], spec=SPEC)


class ValidateScriptsTests(unittest.TestCase):
    CORPUS = {
        "tsukuyomi-corpus-v1:VOICEACTRESS100_001": "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
        "tsukuyomi-corpus-v1:VOICEACTRESS100_002": "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。",
    }

    def _dialogue(self, artifact_id: str, **overrides) -> dict:
        row = {
            "dialogue_id": "v-001",
            "source_artifact_id": artifact_id,
            "turns": [
                {"speaker": "B", "text": "Bの前置き"},
                {"speaker": "A", "text": self.CORPUS[artifact_id]},
                {"speaker": "B", "text": "Bの受け"},
            ],
        }
        row.update(overrides)
        return row

    def test_a_well_formed_pair_of_dialogues_validates(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["a_texts_match_corpus"], True)

    def test_an_altered_a_turn_is_caught(self) -> None:
        # A must speak the corpus sentence verbatim; the recording is the audio.
        rows = [self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001")]
        rows[0]["turns"][1]["text"] = "また、東寺のように、五大明王と呼ばれます。"
        rows.append(self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"))
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["altered_a_turns"]))

    def test_a_missing_corpus_sentence_fails_set_equality(self) -> None:
        # A count check would pass here; only set equality catches a duplicate plus a gap.
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-002"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["a_texts_match_corpus"])

    def test_speaker_order_must_be_b_a_b(self) -> None:
        rows = [self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001")]
        rows[0]["turns"] = [{"speaker": "A", "text": self.CORPUS[list(self.CORPUS)[0]]}]
        rows.append(self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"))
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["bad_turn_structure"]))

    def test_reusing_an_evaluation_sentence_is_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][0]["text"] = "評価用に固定した文です"
        report = validate_scripts(
            rows, corpus_texts=self.CORPUS, eval_texts={"評価用に固定した文です"}
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["eval_overlap_count"], 1)

    def test_duplicate_dialogue_ids_are_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-001"),
        ]
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", report["duplicate_dialogue_ids"])

    def test_an_empty_b_turn_is_caught(self) -> None:
        rows = [
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_001", dialogue_id="v-001"),
            self._dialogue("tsukuyomi-corpus-v1:VOICEACTRESS100_002", dialogue_id="v-002"),
        ]
        rows[0]["turns"][0]["text"] = "   "
        report = validate_scripts(rows, corpus_texts=self.CORPUS, eval_texts=set())
        self.assertEqual(report["status"], "fail")
        self.assertIn("v-001", str(report["empty_turns"]))


if __name__ == "__main__":
    unittest.main()
