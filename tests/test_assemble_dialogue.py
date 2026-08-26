import unittest
from pathlib import Path

from tools.assemble_dialogue import (
    Clip,
    Join,
    OverlapSpec,
    TimelineSpec,
    allocate_word_times,
    best_lag_ncc,
    channel_gate,
    choose_pause,
    dialogue_joins,
    dialogue_timeline,
    draw_offsets,
    frames_for,
    group_dialogues,
    grouping_options,
    merge_intervals,
    overlap_frames,
    place_turns,
    quiet_runs,
    same_channel_collisions,
    silence_share,
    speaker_spans,
    speaking_seconds,
    speech_extent,
    stable_seed,
    steps_for_grouping,
)
from tools.synthesize_turns import DurationModel, turn_seed, turns_to_render

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


class ClipTests(unittest.TestCase):
    def test_speech_has_to_sit_inside_the_clip(self) -> None:
        with self.assertRaises(ValueError):
            Clip(speaker="A", duration=1.0, speech_start=0.1, speech_end=1.5)

    def test_a_clip_needs_a_positive_duration(self) -> None:
        with self.assertRaises(ValueError):
            Clip(speaker="A", duration=0.0, speech_start=0.0, speech_end=0.0)

    def test_speech_cannot_run_backwards(self) -> None:
        with self.assertRaises(ValueError):
            Clip(speaker="A", duration=2.0, speech_start=1.0, speech_end=0.5)


class JoinTests(unittest.TestCase):
    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Join(0, "middle", 0.1)


class PlaceTurnsTests(unittest.TestCase):
    """The M3-R placement: joins are measured on speech, and may be negative."""

    def _clips(self):
        return [
            Clip(speaker="B", role="open", duration=2.0, speech_start=0.1, speech_end=1.9),
            Clip(speaker="A", role="body", duration=3.0, speech_start=0.2, speech_end=2.6),
            Clip(speaker="B", role="close", duration=2.0, speech_start=0.1, speech_end=1.8),
        ]

    def test_the_first_turn_starts_at_the_lead_in(self) -> None:
        placed = place_turns(self._clips(), [None, Join(0), Join(1)], spec=SPEC)
        self.assertAlmostEqual(placed[0]["clip_start"], 0.5)

    def test_a_zero_join_makes_speech_meet_speech(self) -> None:
        # Clip edges differ from speech edges, so a zero join is not a zero gap between
        # files - it is a zero gap between voices, which is what a spec can state.
        placed = place_turns(self._clips(), [None, Join(0, "speech", 0.0), Join(1)], spec=SPEC)
        self.assertAlmostEqual(placed[1]["speech_start"], placed[0]["speech_end"])

    def test_a_negative_join_puts_two_voices_in_the_same_instant(self) -> None:
        placed = place_turns(self._clips(), [None, Join(0, "speech", -0.4), Join(1)], spec=SPEC)
        self.assertAlmostEqual(placed[1]["speech_start"], placed[0]["speech_end"] - 0.4)
        self.assertLess(placed[1]["speech_start"], placed[0]["speech_end"])

    def test_a_clip_join_of_zero_loses_no_sample(self) -> None:
        clips = [
            Clip(speaker="B", duration=2.0, speech_start=0.1, speech_end=1.9),
            Clip(speaker="A", duration=3.0, speech_start=0.2, speech_end=2.6),
            Clip(speaker="A", duration=2.0, speech_start=0.3, speech_end=1.7),
        ]
        placed = place_turns(clips, [None, Join(0), Join(1, "clip", 0.0)], spec=SPEC)
        self.assertAlmostEqual(placed[2]["clip_start"], placed[1]["clip_end"])

    def test_an_anchor_must_point_backwards(self) -> None:
        with self.assertRaises(ValueError):
            place_turns(self._clips(), [None, Join(2), Join(1)], spec=SPEC)

    def test_a_turn_pushed_before_time_zero_is_refused(self) -> None:
        # Shifting instead would move every other turn and silently change the overlaps.
        with self.assertRaises(ValueError):
            place_turns(self._clips(), [None, Join(0, "speech", -5.0), Join(1)], spec=SPEC)

    def test_a_missing_join_is_an_error_not_a_default(self) -> None:
        with self.assertRaises(ValueError):
            place_turns(self._clips(), [None, None, Join(1)], spec=SPEC)

    def test_the_join_count_has_to_match_the_clip_count(self) -> None:
        with self.assertRaises(ValueError):
            place_turns(self._clips(), [None, Join(0)], spec=SPEC)

    def test_two_clips_of_one_speaker_are_never_placed_on_top_of_each_other(self) -> None:
        # One channel carrying two clips at once is a sum of two voices, not an overlap.
        # The case that produces it: a backchannel still sounding when a very short second
        # half of speaker A's sentence has already ended, so B's closing turn would start
        # on top of B's own aizuchi.
        clips = [
            Clip(speaker="B", role="open", duration=2.0, speech_start=0.1, speech_end=1.9),
            Clip(speaker="A", role="body", duration=2.0, speech_start=0.1, speech_end=1.9),
            Clip(speaker="B", role="backchannel", duration=2.0, speech_start=0.2, speech_end=1.4),
            Clip(speaker="A", role="body", duration=0.6, speech_start=0.05, speech_end=0.5),
            Clip(speaker="B", role="close", duration=2.0, speech_start=0.1, speech_end=1.9),
        ]
        joins = [
            None,
            Join(0, "speech", -0.2),
            Join(1, "speech", -0.4),
            Join(1, "clip", 0.0),
            Join(3, "speech", -0.2),
        ]
        repaired = place_turns(clips, joins, spec=SPEC, min_same_speaker_gap=0.05)
        self.assertEqual(same_channel_collisions(repaired), [])
        # The join asked for B's close to begin 0.2 s before A stopped, at 4.4 s. B's own
        # aizuchi is still on the channel until 5.4 s, so the close waits.
        self.assertGreater(repaired[4]["deferred_seconds"], 0.0)
        self.assertGreaterEqual(repaired[4]["clip_start"], repaired[2]["clip_end"])

    def test_a_placement_that_does_not_collide_is_not_deferred(self) -> None:
        placed = place_turns(
            self._clips(),
            [None, Join(0, "speech", -0.3), Join(1, "speech", -0.3)],
            spec=SPEC,
            min_same_speaker_gap=0.05,
        )
        self.assertEqual([row["deferred_seconds"] for row in placed], [0.0, 0.0, 0.0])


class MergeIntervalsTests(unittest.TestCase):
    def test_overlapping_spans_are_counted_once(self) -> None:
        self.assertEqual(merge_intervals([(0.0, 2.0), (1.0, 3.0)]), [(0.0, 3.0)])

    def test_touching_spans_join(self) -> None:
        self.assertEqual(merge_intervals([(0.0, 1.0), (1.0, 2.0)]), [(0.0, 2.0)])

    def test_separate_spans_stay_separate(self) -> None:
        self.assertEqual(merge_intervals([(2.0, 3.0), (0.0, 1.0)]), [(0.0, 1.0), (2.0, 3.0)])

    def test_an_empty_span_contributes_nothing(self) -> None:
        self.assertEqual(merge_intervals([(1.0, 1.0)]), [])


class SilenceShareTests(unittest.TestCase):
    def test_the_clip_convention_counts_a_turn_s_own_silence_as_speaking_time(self) -> None:
        placed = [
            {
                "speaker": "A",
                "clip_start": 0.0,
                "clip_end": 4.0,
                "speech_start": 1.0,
                "speech_end": 3.0,
            },
            {
                "speaker": "B",
                "clip_start": 4.0,
                "clip_end": 8.0,
                "speech_start": 4.0,
                "speech_end": 8.0,
            },
        ]
        self.assertAlmostEqual(silence_share(placed, "A", extent="clip"), 0.5)
        self.assertAlmostEqual(silence_share(placed, "A", extent="speech"), 0.75)

    def test_a_speaker_who_never_speaks_is_silent_throughout(self) -> None:
        placed = [
            {
                "speaker": "B",
                "clip_start": 0.0,
                "clip_end": 4.0,
                "speech_start": 0.0,
                "speech_end": 4.0,
            }
        ]
        self.assertAlmostEqual(silence_share(placed, "A"), 1.0)

    def test_overlapping_turns_of_one_speaker_are_not_double_counted(self) -> None:
        placed = [
            {
                "speaker": "A",
                "clip_start": 0.0,
                "clip_end": 3.0,
                "speech_start": 0.0,
                "speech_end": 3.0,
            },
            {
                "speaker": "A",
                "clip_start": 2.0,
                "clip_end": 4.0,
                "speech_start": 2.0,
                "speech_end": 4.0,
            },
        ]
        self.assertAlmostEqual(speaking_seconds(placed, "A"), 4.0)


class OverlapFramesTests(unittest.TestCase):
    def test_a_sequential_dialogue_has_no_simultaneous_frame(self) -> None:
        placed = [
            {
                "speaker": "B",
                "clip_start": 0.0,
                "clip_end": 2.0,
                "speech_start": 0.0,
                "speech_end": 2.0,
            },
            {
                "speaker": "A",
                "clip_start": 2.0,
                "clip_end": 4.0,
                "speech_start": 2.0,
                "speech_end": 4.0,
            },
        ]
        self.assertEqual(overlap_frames(placed)["simultaneous_frames"], 0)

    def test_an_overlap_shows_up_as_frames_and_not_only_as_seconds(self) -> None:
        placed = [
            {
                "speaker": "B",
                "clip_start": 0.0,
                "clip_end": 2.0,
                "speech_start": 0.0,
                "speech_end": 2.0,
            },
            {
                "speaker": "A",
                "clip_start": 1.6,
                "clip_end": 4.0,
                "speech_start": 1.6,
                "speech_end": 4.0,
            },
        ]
        # 0.4 s at 12.5 Hz is five frames.
        self.assertEqual(overlap_frames(placed)["simultaneous_frames"], 5)

    def test_padding_overlap_alone_does_not_count(self) -> None:
        # Two clips can overlap entirely in their silent tails and heads; the gate has to
        # be about voices, not about files.
        placed = [
            {
                "speaker": "B",
                "clip_start": 0.0,
                "clip_end": 2.0,
                "speech_start": 0.0,
                "speech_end": 1.5,
            },
            {
                "speaker": "A",
                "clip_start": 1.6,
                "clip_end": 4.0,
                "speech_start": 2.1,
                "speech_end": 4.0,
            },
        ]
        self.assertEqual(overlap_frames(placed)["simultaneous_frames"], 0)


class QuietRunsTests(unittest.TestCase):
    def test_runs_are_half_open_and_in_order(self) -> None:
        self.assertEqual(quiet_runs([1.0, 0.0, 0.0, 1.0, 0.0], threshold=0.1), [(1, 3), (4, 5)])

    def test_a_fully_quiet_series_is_one_run(self) -> None:
        self.assertEqual(quiet_runs([0.0, 0.0], threshold=0.1), [(0, 2)])

    def test_a_series_with_no_quiet_frame_has_no_run(self) -> None:
        self.assertEqual(quiet_runs([1.0, 1.0], threshold=0.1), [])


class SpeechExtentTests(unittest.TestCase):
    def test_the_extent_skips_the_clip_s_silent_head_and_tail(self) -> None:
        levels = [0.0, 0.0, 1.0, 1.0, 0.0]
        self.assertEqual(speech_extent(levels, threshold=0.1, hop_seconds=0.01), (0.02, 0.04))

    def test_a_clip_with_nothing_audible_returns_the_whole_clip(self) -> None:
        self.assertEqual(speech_extent([0.0, 0.0], threshold=0.1, hop_seconds=0.01), (0.0, 0.02))


class ChoosePauseTests(unittest.TestCase):
    def test_the_pause_nearest_the_mora_target_wins(self) -> None:
        runs = [(10, 25), (60, 80)]
        chosen = choose_pause(
            runs, target_seconds=0.70, hop_seconds=0.01, min_frames=6, edge_frames=5, frames=100
        )
        self.assertEqual(chosen, (60, 80))

    def test_a_consonant_closure_is_too_short_to_be_a_pause(self) -> None:
        runs = [(50, 52), (60, 80)]
        chosen = choose_pause(
            runs, target_seconds=0.51, hop_seconds=0.01, min_frames=6, edge_frames=5, frames=100
        )
        self.assertEqual(chosen, (60, 80))

    def test_the_recording_s_own_head_and_tail_are_not_cutting_points(self) -> None:
        runs = [(0, 8), (92, 100)]
        chosen = choose_pause(
            runs, target_seconds=0.50, hop_seconds=0.01, min_frames=6, edge_frames=10, frames=100
        )
        self.assertIsNone(chosen)

    def test_no_usable_pause_returns_none_rather_than_a_guess(self) -> None:
        self.assertIsNone(
            choose_pause(
                [], target_seconds=0.5, hop_seconds=0.01, min_frames=6, edge_frames=5, frames=100
            )
        )


class DialogueJoinsTests(unittest.TestCase):
    OFFSETS = {"open_to_body": -0.2, "body_to_backchannel": -0.4, "body_to_close": -0.3}

    def test_the_backchannel_is_anchored_to_a_s_first_half(self) -> None:
        joins = dialogue_joins(["open", "body", "backchannel", "body", "close"], self.OFFSETS)
        self.assertEqual(joins[2].anchor, 1)
        self.assertLess(joins[2].seconds, 0.0)

    def test_a_s_two_halves_are_butt_joined(self) -> None:
        joins = dialogue_joins(["open", "body", "backchannel", "body", "close"], self.OFFSETS)
        self.assertEqual((joins[3].anchor, joins[3].mode, joins[3].seconds), (1, "clip", 0.0))

    def test_the_unsplit_shape_still_has_a_plan(self) -> None:
        joins = dialogue_joins(["open", "body", "close"], self.OFFSETS)
        self.assertEqual(len(joins), 3)
        self.assertEqual(joins[2].anchor, 1)

    def test_an_unknown_shape_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            dialogue_joins(["open", "close"], self.OFFSETS)


class DrawOffsetsTests(unittest.TestCase):
    def test_the_backchannel_offset_is_always_an_overlap(self) -> None:
        import random

        for seed in range(50):
            drawn = draw_offsets(random.Random(seed), OverlapSpec(), has_backchannel=True)
            self.assertLess(drawn["body_to_backchannel"], 0.0)

    def test_a_dialogue_without_a_backchannel_draws_two_offsets(self) -> None:
        import random

        drawn = draw_offsets(random.Random(0), OverlapSpec(), has_backchannel=False)
        self.assertEqual(set(drawn), {"open_to_body", "body_to_close"})

    def test_the_same_seed_draws_the_same_timeline(self) -> None:
        import random

        first = draw_offsets(random.Random(7), OverlapSpec(), has_backchannel=True)
        second = draw_offsets(random.Random(7), OverlapSpec(), has_backchannel=True)
        self.assertEqual(first, second)


class StableSeedTests(unittest.TestCase):
    def test_the_seed_depends_on_the_identity(self) -> None:
        self.assertNotEqual(stable_seed(1, "v-001"), stable_seed(1, "v-002"))

    def test_the_seed_does_not_depend_on_this_process(self) -> None:
        # Recorded rather than recomputed: a salted hash would pass a same-process check.
        self.assertEqual(stable_seed(20260825, "v-001"), stable_seed(20260825, "v-001"))
        self.assertEqual(stable_seed(20260825, "v-001", "tone"), 491519970)


class GroupDialoguesTests(unittest.TestCase):
    def test_groups_are_non_overlapping_so_no_dialogue_is_trained_on_twice(self) -> None:
        groups = group_dialogues([f"v-{i:03d}" for i in range(8)], group_size=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len({d for g in groups for d in g}), 8)

    def test_a_short_final_group_is_kept(self) -> None:
        groups = group_dialogues(["a", "b", "c"], group_size=2)
        self.assertEqual(groups, [["a", "b"], ["c"]])

    def test_a_group_size_of_one_changes_nothing(self) -> None:
        self.assertEqual(group_dialogues(["a", "b"], group_size=1), [["a"], ["b"]])

    def test_a_group_size_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            group_dialogues(["a"], group_size=0)


class StepsForGroupingTests(unittest.TestCase):
    def test_grouping_by_four_at_batch_two_keeps_m3_s_forty_five_steps(self) -> None:
        # 72 dialogues -> 18 rows; ceil(18/2) = 9 steps an epoch, the same 9 M3 ran.
        shape = steps_for_grouping(dialogues=72, group_size=4, global_batch=2, epochs=5)
        self.assertEqual(shape["rows"], 18)
        self.assertEqual(shape["steps_per_epoch"], 9)
        self.assertEqual(shape["total_steps"], 45)

    def test_grouping_without_lowering_the_batch_loses_steps(self) -> None:
        shape = steps_for_grouping(dialogues=72, group_size=4, global_batch=8, epochs=5)
        self.assertEqual(shape["total_steps"], 15)

    def test_the_ungrouped_run_is_the_m3_shape(self) -> None:
        shape = steps_for_grouping(dialogues=72, group_size=1, global_batch=8, epochs=5)
        self.assertEqual((shape["rows"], shape["total_steps"]), (72, 45))

    def test_the_dialogues_a_step_averages_over_is_reported(self) -> None:
        # 4 dialogues x batch 2 is the same 8 dialogues a step M3 saw at batch 8.
        shape = steps_for_grouping(dialogues=72, group_size=4, global_batch=2, epochs=5)
        self.assertEqual(shape["dialogues_per_step"], 8)


class BestLagNccTests(unittest.TestCase):
    """The check that would have caught the stereo built before its own backchannels."""

    def setUp(self) -> None:
        try:
            import numpy
        except ImportError as error:  # numpy is not in the test env
            self.skipTest(f"best_lag_ncc needs numpy: {error}")
        self.np = numpy
        rng = self.np.random.default_rng(0)
        self.template = rng.standard_normal(400)
        self.other = rng.standard_normal(400)
        self.signal = self.np.zeros(5000)
        self.signal[1234:1634] = self.template

    def test_an_embedded_clip_scores_one_at_its_offset(self) -> None:
        found = best_lag_ncc(self.signal, self.template)
        self.assertAlmostEqual(found["ncc"], 1.0, places=9)
        self.assertEqual(found["lag_samples"], 1234)

    def test_a_clip_that_is_not_there_scores_low(self) -> None:
        self.assertLess(best_lag_ncc(self.signal, self.other)["ncc"], 0.5)

    def test_a_clip_shifted_by_twenty_milliseconds_is_still_found(self) -> None:
        # The whole point of maximising over lag: the placed span starts at the first
        # audible sample, the wav starts at its silent head, and a fixed-offset comparison
        # reads ~0 on a file that provably contains the clip.
        found = best_lag_ncc(self.signal, self.np.concatenate([self.np.zeros(480), self.template]))
        self.assertAlmostEqual(found["ncc"], 1.0, places=6)
        self.assertEqual(found["lag_samples"], 1234 - 480)

    def test_a_room_tone_floor_under_the_clip_does_not_inflate_the_score(self) -> None:
        # Both series are mean-removed, so a shared DC offset cannot pass for a match.
        self.assertLess(
            best_lag_ncc(self.signal + 0.5, self.other + 0.5)["ncc"],
            0.5,
        )

    def test_a_silent_template_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            best_lag_ncc(self.signal, self.np.zeros(400))

    def test_a_template_longer_than_the_file_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            best_lag_ncc(self.np.zeros(100), self.template)


class SpeakerSpansTests(unittest.TestCase):
    TRANSCRIPT = [
        {"speaker": "B", "word": "あ", "start": 0.5, "end": 1.0},
        {"speaker": "B", "word": "い", "start": 1.0, "end": 1.5},
        {"speaker": "A", "word": "う", "start": 1.9, "end": 2.5},
        {"speaker": "B", "word": "え", "start": 2.9, "end": 3.4},
    ]

    def test_a_gap_between_words_starts_a_new_turn(self) -> None:
        self.assertEqual(speaker_spans(self.TRANSCRIPT, "B"), [(0.5, 1.5), (2.9, 3.4)])

    def test_one_turn_comes_back_as_one_span(self) -> None:
        self.assertEqual(speaker_spans(self.TRANSCRIPT, "A"), [(1.9, 2.5)])

    def test_a_speaker_with_no_words_has_no_span(self) -> None:
        self.assertEqual(speaker_spans(self.TRANSCRIPT, "C"), [])


class SynthesisDurationTests(unittest.TestCase):
    """Lives here rather than in test_synthesize_turns.py, which another agent owns."""

    def test_a_two_mora_aizuchi_gets_about_half_a_second(self) -> None:
        # Measured: at 0.55 s both 「ええ。」 and 「はい。」 read back correctly on every seed
        # tried; at 1.1 s all seven rendered 「ええ。」 came back as invented speech.
        model = DurationModel()
        self.assertAlmostEqual(model.seconds(mora=2, commas=0), 0.57)

    def test_the_floor_never_asks_for_more_audio_than_the_words_need(self) -> None:
        # Asking for more than the words take is how the first render broke: the model fills
        # the excess rather than leaving it silent.
        model = DurationModel()
        self.assertLessEqual(model.seconds(mora=2, commas=0), 0.7)

    def test_a_longer_aizuchi_grows_past_the_floor(self) -> None:
        model = DurationModel()
        self.assertGreater(model.seconds(mora=11, commas=1), model.floor)

    def test_a_reading_comma_buys_a_pause(self) -> None:
        model = DurationModel()
        self.assertGreater(model.seconds(mora=11, commas=1), model.seconds(mora=11, commas=0))

    def test_the_rate_matches_the_speaker_it_was_fitted_on(self) -> None:
        # The 160 shipped B turns fit 0.151 s per mora; 0.135 plus the fixed head is the
        # same line inside the range that matters here.
        model = DurationModel()
        self.assertAlmostEqual(model.seconds(mora=30, commas=0), 0.30 + 30 * 0.135)


class TurnSeedTests(unittest.TestCase):
    """Also here for the same reason; the render has to be resumable."""

    def test_two_dialogues_using_the_same_text_get_different_waveforms(self) -> None:
        self.assertNotEqual(turn_seed(1, "v-001", 2), turn_seed(1, "v-002", 2))

    def test_the_seed_survives_a_restart(self) -> None:
        self.assertEqual(turn_seed(20260825, "v-001", 2), 2833438582)


class RolesFilterTests(unittest.TestCase):
    DIALOGUES = [
        {
            "dialogue_id": "v-001",
            "turns": [
                {"speaker": "B", "text": "問", "role": "open"},
                {"speaker": "A", "text": "答", "role": "body"},
                {"speaker": "B", "text": "ええ", "role": "backchannel"},
                {"speaker": "A", "text": "続", "role": "body"},
                {"speaker": "B", "text": "受", "role": "close"},
            ],
        }
    ]

    def test_only_the_named_role_is_rendered(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(
                self.DIALOGUES,
                speaker="B",
                out_dir=Path(tmp),
                recorded=set(),
                roles=["backchannel"],
            )
        self.assertEqual([turn["turn_index"] for turn in pending], [2])

    def test_no_roles_asked_for_means_every_turn(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(
                self.DIALOGUES, speaker="B", out_dir=Path(tmp), recorded=set()
            )
        self.assertEqual(len(pending), 3)

    def test_a_script_without_roles_renders_nothing_rather_than_everything(self) -> None:
        import tempfile

        plain = [{"dialogue_id": "v-001", "turns": [{"speaker": "B", "text": "問"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            pending = turns_to_render(
                plain, speaker="B", out_dir=Path(tmp), recorded=set(), roles=["backchannel"]
            )
        self.assertEqual(pending, [])


class SameChannelCollisionsTests(unittest.TestCase):
    def test_two_turns_of_one_speaker_sharing_an_instant_are_reported(self) -> None:
        placed = [
            {
                "speaker": "B",
                "clip_start": 0.0,
                "clip_end": 2.0,
                "speech_start": 0.0,
                "speech_end": 2.0,
            },
            {
                "speaker": "B",
                "clip_start": 1.5,
                "clip_end": 3.0,
                "speech_start": 1.5,
                "speech_end": 3.0,
            },
        ]
        self.assertEqual(same_channel_collisions(placed), [(0, 1)])

    def test_two_speakers_sharing_an_instant_are_not_a_collision(self) -> None:
        placed = [
            {
                "speaker": "A",
                "clip_start": 0.0,
                "clip_end": 2.0,
                "speech_start": 0.0,
                "speech_end": 2.0,
            },
            {
                "speaker": "B",
                "clip_start": 1.5,
                "clip_end": 3.0,
                "speech_start": 1.5,
                "speech_end": 3.0,
            },
        ]
        self.assertEqual(same_channel_collisions(placed), [])


class GroupingOptionsTests(unittest.TestCase):
    """Sequence length and step count pull against each other; the options show the trade."""

    def _options(self):
        return grouping_options(
            dialogues=72,
            epochs=5,
            target_steps=45,
            seconds_per_dialogue=19.4,
            gap_seconds=0.4,
        )

    def test_every_option_lands_on_the_target_step_count(self) -> None:
        from tools.training_shape import total_steps

        for option in self._options():
            self.assertEqual(
                total_steps(examples=option["rows"], batch=option["global_batch"], epochs=5), 45
            )

    def test_the_ungrouped_m3_shape_is_among_them(self) -> None:
        shapes = {(o["group_size"], o["global_batch"]) for o in self._options()}
        self.assertIn((1, 8), shapes)

    def test_only_the_larger_groups_reach_sixty_seconds(self) -> None:
        reaching = [o for o in self._options() if o["meets_60s"]]
        self.assertTrue(reaching)
        self.assertTrue(all(o["group_size"] >= 4 for o in reaching))

    def test_grouping_by_four_at_batch_two_is_offered(self) -> None:
        shapes = {(o["group_size"], o["global_batch"]) for o in self._options()}
        self.assertIn((4, 2), shapes)

    def test_an_option_that_misses_the_target_is_not_offered(self) -> None:
        # 72 dialogues in threes is 24 rows, and no integer batch turns 24 rows into 9
        # steps an epoch.
        self.assertEqual([o for o in self._options() if o["group_size"] == 3], [])


class ButtJointSurvivesTheGapTests(unittest.TestCase):
    """The repair must not fire on the joint it was not written for."""

    def test_a_butt_joint_is_not_pushed_apart_by_the_collision_gap(self) -> None:
        # Speaker A's two halves are one recording cut in two; 0.05 s of room tone inserted
        # between them is 0.05 s that the speaker did not leave.
        clips = [
            Clip(speaker="B", duration=2.0, speech_start=0.1, speech_end=1.9),
            Clip(speaker="A", duration=2.0, speech_start=0.1, speech_end=1.6),
            Clip(speaker="A", duration=2.0, speech_start=0.3, speech_end=1.9),
        ]
        placed = place_turns(
            clips,
            [None, Join(0, "speech", -0.2), Join(1, "clip", 0.0)],
            spec=SPEC,
            min_same_speaker_gap=0.05,
        )
        self.assertAlmostEqual(placed[2]["clip_start"], placed[1]["clip_end"])
        self.assertEqual(placed[2]["deferred_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
