import unittest

from tools.assemble_dialogue import (
    TimelineSpec,
    allocate_word_times,
    channel_gate,
    dialogue_timeline,
    frames_for,
)

SPEC = TimelineSpec(lead_in_seconds=0.5, gap_seconds=0.4, sample_rate=24000, frame_rate_hz=12.5)


class AllocateWordTimesTests(unittest.TestCase):
    def test_duration_is_split_in_proportion_to_mora(self) -> None:
        # 東寺 is two characters but three mora; splitting by character length would give it
        # the same time as a two-mora word and shift everything after it.
        words = [("東寺", 3), ("の", 1)]
        times = allocate_word_times(words, start=0.0, end=4.0)
        self.assertAlmostEqual(times[0]["start"], 0.0)
        self.assertAlmostEqual(times[0]["end"], 3.0)
        self.assertAlmostEqual(times[1]["start"], 3.0)
        self.assertAlmostEqual(times[1]["end"], 4.0)

    def test_zero_mora_words_still_get_a_span(self) -> None:
        # Punctuation has no mora but does have characters, and tokenize_text consumes the
        # characters of every segment it is handed - a zero-length span would collide.
        times = allocate_word_times([("、", 0), ("あ", 1)], start=0.0, end=2.0)
        self.assertGreater(times[0]["end"], times[0]["start"])
        self.assertAlmostEqual(times[-1]["end"], 2.0)

    def test_spans_are_contiguous_and_fill_the_turn(self) -> None:
        times = allocate_word_times([("あ", 1), ("いう", 2), ("え", 1)], start=1.0, end=5.0)
        self.assertAlmostEqual(times[0]["start"], 1.0)
        self.assertAlmostEqual(times[-1]["end"], 5.0)
        for earlier, later in zip(times[:-1], times[1:], strict=True):
            self.assertAlmostEqual(earlier["end"], later["start"])

    def test_an_empty_word_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_word_times([], start=0.0, end=1.0)

    def test_a_non_positive_span_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_word_times([("あ", 1)], start=2.0, end=2.0)


class DialogueTimelineTests(unittest.TestCase):
    def test_turns_are_sequential_with_a_lead_in_and_fixed_gaps(self) -> None:
        placed = dialogue_timeline([("B", 2.0), ("A", 3.0), ("B", 1.0)], spec=SPEC)
        self.assertAlmostEqual(placed[0]["start"], 0.5)
        self.assertAlmostEqual(placed[0]["end"], 2.5)
        self.assertAlmostEqual(placed[1]["start"], 2.9)
        self.assertAlmostEqual(placed[1]["end"], 5.9)
        self.assertAlmostEqual(placed[2]["start"], 6.3)

    def test_no_two_turns_overlap(self) -> None:
        placed = dialogue_timeline([("B", 2.0), ("A", 3.0), ("B", 1.0)], spec=SPEC)
        for earlier, later in zip(placed[:-1], placed[1:], strict=True):
            self.assertLessEqual(earlier["end"], later["start"])

    def test_total_length_includes_the_trailing_turn(self) -> None:
        placed = dialogue_timeline([("B", 2.0), ("A", 3.0), ("B", 1.0)], spec=SPEC)
        self.assertAlmostEqual(placed[-1]["end"], 7.3)


class FramesForTests(unittest.TestCase):
    def test_frames_round_up_so_the_last_sample_is_covered(self) -> None:
        self.assertEqual(frames_for(1.04, frame_rate_hz=12.5), 13)

    def test_a_duration_landing_mid_frame_occupies_that_frame(self) -> None:
        # 1.0 s is 12.5 frames; the audio is present in the thirteenth, so it counts.
        self.assertEqual(frames_for(1.0, frame_rate_hz=12.5), 13)

    def test_a_whole_number_of_frames_gains_nothing(self) -> None:
        self.assertEqual(frames_for(1.6, frame_rate_hz=12.5), 20)


class ChannelGateTests(unittest.TestCase):
    """A left/right swap is invisible everywhere else in the pipeline."""

    def test_a_correctly_assigned_dialogue_passes(self) -> None:
        self.assertTrue(channel_gate(a_rms=1000.0, b_rms=200.0, ratio=1.5)["ok"])

    def test_a_swapped_dialogue_fails(self) -> None:
        self.assertFalse(channel_gate(a_rms=200.0, b_rms=1000.0, ratio=1.5)["ok"])

    def test_a_silent_b_channel_is_rejected_rather_than_passing_the_ratio(self) -> None:
        # A silent B satisfies any ratio, and would mean the user never speaks.
        self.assertFalse(channel_gate(a_rms=1000.0, b_rms=0.0, ratio=1.5)["ok"])

    def test_a_silent_a_channel_is_rejected(self) -> None:
        self.assertFalse(channel_gate(a_rms=0.0, b_rms=1000.0, ratio=1.5)["ok"])


if __name__ == "__main__":
    unittest.main()
