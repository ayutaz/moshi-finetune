import tempfile
import unittest
from pathlib import Path

from tools.synthesize_turns import turn_filename, turns_to_render

DIALOGUES = [
    {
        "dialogue_id": "v-001",
        "turns": [
            {"speaker": "B", "text": "前置き"},
            {"speaker": "A", "text": "コーパスの文"},
            {"speaker": "B", "text": "受け"},
        ],
    },
    {
        "dialogue_id": "v-002",
        "turns": [
            {"speaker": "B", "text": "前置き2"},
            {"speaker": "A", "text": "コーパスの文2"},
            {"speaker": "B", "text": "受け2"},
        ],
    },
]


class TurnFilenameTests(unittest.TestCase):
    def test_the_turn_index_distinguishes_a_dialogue_s_two_b_turns(self) -> None:
        self.assertNotEqual(turn_filename("v-001", 0, "B"), turn_filename("v-001", 2, "B"))

    def test_the_name_carries_dialogue_index_and_speaker(self) -> None:
        self.assertEqual(turn_filename("v-001", 2, "B"), "v-001-t2-B.wav")


class TurnsToRenderTests(unittest.TestCase):
    def test_it_selects_only_the_requested_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(DIALOGUES, speaker="B", out_dir=Path(tmp))
        self.assertEqual(len(pending), 4)
        self.assertEqual({turn["speaker"] for turn in pending}, {"B"})

    def test_speaker_a_yields_one_turn_per_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(DIALOGUES, speaker="A", out_dir=Path(tmp))
        self.assertEqual(len(pending), 2)

    def test_already_rendered_turns_are_skipped(self) -> None:
        # A three-hour render must not restart from zero after an interruption.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / turn_filename("v-001", 0, "B")).write_bytes(b"")
            pending = turns_to_render(DIALOGUES, speaker="B", out_dir=out)
        self.assertEqual(len(pending), 3)
        self.assertNotIn(turn_filename("v-001", 0, "B"), [t["filename"] for t in pending])

    def test_order_follows_the_script_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(DIALOGUES, speaker="B", out_dir=Path(tmp))
        self.assertEqual([t["dialogue_id"] for t in pending], ["v-001", "v-001", "v-002", "v-002"])

    def test_the_text_travels_with_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(DIALOGUES, speaker="A", out_dir=Path(tmp))
        self.assertEqual(pending[0]["text"], "コーパスの文")


if __name__ == "__main__":
    unittest.main()
