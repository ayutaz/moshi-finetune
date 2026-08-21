import unittest

from tools.turn_taking import (
    activity_mask,
    response_latencies,
    runs_of_activity,
    summarise_turn_taking,
)


class ActivityMaskTests(unittest.TestCase):
    def test_frames_above_the_threshold_are_active(self) -> None:
        samples = [0, 0, 0, 0, 900, 900, 900, 900]
        self.assertEqual(activity_mask(samples, frame=4, threshold=500), [False, True])

    def test_a_partial_final_frame_is_still_scored(self) -> None:
        # Dropping it would silently shorten every clip whose length is not a multiple.
        samples = [900, 900, 900, 900, 900, 900]
        self.assertEqual(activity_mask(samples, frame=4, threshold=500), [True, True])

    def test_an_empty_signal_has_no_frames(self) -> None:
        self.assertEqual(activity_mask([], frame=4, threshold=500), [])

    def test_a_frame_size_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            activity_mask([1, 2], frame=0, threshold=500)


class RunsOfActivityTests(unittest.TestCase):
    def test_it_returns_start_and_end_of_each_run(self) -> None:
        mask = [False, True, True, False, False, True]
        self.assertEqual(runs_of_activity(mask), [(1, 3), (5, 6)])

    def test_silence_has_no_runs(self) -> None:
        self.assertEqual(runs_of_activity([False, False]), [])

    def test_a_fully_active_mask_is_one_run(self) -> None:
        self.assertEqual(runs_of_activity([True, True, True]), [(0, 3)])


class ResponseLatencyTests(unittest.TestCase):
    def test_latency_is_measured_from_partner_end_to_next_self_onset(self) -> None:
        partner = [(0, 4)]  # partner speaks frames 0..3
        own = [(7, 10)]  # model starts at frame 7
        self.assertEqual(response_latencies(partner_runs=partner, own_runs=own), [3])

    def test_a_partner_turn_that_is_never_answered_yields_no_latency(self) -> None:
        self.assertEqual(response_latencies(partner_runs=[(0, 4)], own_runs=[]), [])

    def test_speech_already_underway_is_not_counted_as_a_response(self) -> None:
        # The model was talking before the partner finished; that is not a response time.
        self.assertEqual(response_latencies(partner_runs=[(5, 9)], own_runs=[(0, 3)]), [])


class SummariseTurnTakingTests(unittest.TestCase):
    """The detector must recognise the monologue it exists to rule out."""

    def test_a_silent_partner_is_reported_as_a_monologue(self) -> None:
        moshi = [True] * 20
        user = [False] * 20
        summary = summarise_turn_taking(moshi_mask=moshi, user_mask=user)
        self.assertEqual(summary["speaker_switches"], 0)
        self.assertEqual(summary["user_speech_ratio"], 0.0)
        self.assertTrue(summary["user_never_active"])

    def test_an_alternating_conversation_has_switches(self) -> None:
        moshi = [True] * 5 + [False] * 5 + [True] * 5
        user = [False] * 5 + [True] * 5 + [False] * 5
        summary = summarise_turn_taking(moshi_mask=moshi, user_mask=user)
        self.assertEqual(summary["speaker_switches"], 2)
        self.assertFalse(summary["user_never_active"])
        self.assertEqual(summary["moshi_speech_ratio"], 10 / 15)

    def test_overlap_is_measured_not_ignored(self) -> None:
        moshi = [True, True, True, False]
        user = [False, True, True, True]
        summary = summarise_turn_taking(moshi_mask=moshi, user_mask=user)
        self.assertEqual(summary["overlap_frames"], 2)

    def test_masks_of_different_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarise_turn_taking(moshi_mask=[True], user_mask=[True, False])

    def test_empty_masks_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarise_turn_taking(moshi_mask=[], user_mask=[])


if __name__ == "__main__":
    unittest.main()
