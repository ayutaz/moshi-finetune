import unittest

from tools.dataset_agreement import (
    AgreementError,
    assemble_report,
    channel_problems,
    counts_from_problems,
    dominance_margins_db,
    excluded_id_sightings,
    exclusive_windows,
    frames_inside,
    longest_true_run,
    normalise_text,
    observed_counts,
    text_tokens_lost,
    timestamp_problems,
    turn_intervals,
)


def word(speaker: str, text: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "word": text, "start": start, "end": end}


class TurnIntervalTests(unittest.TestCase):
    """The transcript is sorted by start time, so overlapping turns interleave in the list.

    Grouping by runs of the same speaker recovers eleven fragments from the five turns of
    a real M3-R dialogue. What separates turns is the time discontinuity inside one
    speaker's own words.
    """

    def test_contiguous_words_of_one_speaker_are_one_turn(self) -> None:
        segments = [word("A", "あ", 0.0, 0.5), word("A", "い", 0.5, 1.0)]

        turns = turn_intervals(segments)

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["text"], "あい")
        self.assertEqual((turns[0]["start"], turns[0]["end"]), (0.0, 1.0))

    def test_a_gap_inside_one_speaker_starts_a_new_turn(self) -> None:
        segments = [
            word("A", "あ", 0.0, 0.5),
            word("A", "い", 0.9, 1.4),
        ]

        turns = turn_intervals(segments)

        self.assertEqual([turn["text"] for turn in turns], ["あ", "い"])

    def test_interleaved_speakers_do_not_split_a_turn(self) -> None:
        """The backchannel lands inside A's turn; A's turn is still one turn."""
        segments = [
            word("A", "あ", 0.0, 1.0),
            word("B", "ええ", 0.8, 1.1),
            word("A", "い", 1.0, 2.0),
        ]

        turns = turn_intervals(segments)

        self.assertEqual(len(turns), 2)
        by_speaker = {turn["speaker"]: turn for turn in turns}
        self.assertEqual(by_speaker["A"]["text"], "あい")
        self.assertEqual((by_speaker["A"]["start"], by_speaker["A"]["end"]), (0.0, 2.0))

    def test_turns_are_ordered_by_start_and_indexed(self) -> None:
        segments = [
            word("B", "ど", 0.0, 1.0),
            word("A", "あ", 2.0, 3.0),
            word("B", "は", 4.0, 5.0),
        ]

        turns = turn_intervals(segments)

        self.assertEqual([turn["index"] for turn in turns], [0, 1, 2])
        self.assertEqual([turn["speaker"] for turn in turns], ["B", "A", "B"])


class TimestampTests(unittest.TestCase):
    def test_clean_transcript_has_no_problems(self) -> None:
        segments = [word("A", "あ", 0.5, 1.0), word("B", "い", 1.0, 1.5)]

        self.assertEqual(timestamp_problems(segments, duration_seconds=2.0), [])

    def test_a_word_past_the_end_of_the_audio_is_caught(self) -> None:
        segments = [word("A", "あ", 0.5, 3.0)]

        kinds = [row["kind"] for row in timestamp_problems(segments, duration_seconds=2.0)]

        self.assertEqual(kinds, ["end_past_audio"])

    def test_a_negative_start_is_caught(self) -> None:
        segments = [word("A", "あ", -0.2, 0.5)]

        kinds = [row["kind"] for row in timestamp_problems(segments, duration_seconds=2.0)]

        self.assertEqual(kinds, ["start_before_zero"])

    def test_an_inverted_word_is_caught(self) -> None:
        segments = [word("A", "あ", 1.0, 0.5)]

        kinds = [row["kind"] for row in timestamp_problems(segments, duration_seconds=2.0)]

        self.assertEqual(kinds, ["end_before_start"])

    def test_words_running_backwards_inside_one_speaker_are_caught(self) -> None:
        segments = [word("A", "あ", 1.0, 1.5), word("A", "い", 0.2, 0.4)]

        kinds = [row["kind"] for row in timestamp_problems(segments, duration_seconds=2.0)]

        self.assertIn("not_monotonic", kinds)


class ExclusiveWindowTests(unittest.TestCase):
    def test_a_turn_nobody_overlaps_is_wholly_exclusive(self) -> None:
        turn = {"speaker": "A", "start": 1.0, "end": 3.0}
        others = [{"speaker": "B", "start": 4.0, "end": 5.0}]

        self.assertEqual(exclusive_windows(turn, others), [(1.0, 3.0)])

    def test_an_overlap_is_subtracted(self) -> None:
        turn = {"speaker": "A", "start": 1.0, "end": 3.0}
        others = [{"speaker": "B", "start": 1.5, "end": 2.0}]

        self.assertEqual(exclusive_windows(turn, others), [(1.0, 1.5), (2.0, 3.0)])

    def test_the_speakers_own_other_turns_do_not_subtract(self) -> None:
        turn = {"speaker": "A", "start": 1.0, "end": 3.0}
        others = [turn, {"speaker": "A", "start": 1.5, "end": 2.0}]

        self.assertEqual(exclusive_windows(turn, others), [(1.0, 3.0)])

    def test_a_turn_covered_end_to_end_has_no_exclusive_part(self) -> None:
        turn = {"speaker": "A", "start": 1.0, "end": 3.0}
        others = [{"speaker": "B", "start": 0.0, "end": 4.0}]

        self.assertEqual(exclusive_windows(turn, others), [])


class FramesInsideTests(unittest.TestCase):
    def test_only_frames_wholly_inside_the_window_are_taken(self) -> None:
        """A frame straddling a boundary carries both sides and testifies about neither."""
        frames = frames_inside([(0.08, 0.32)], frame_count=10, frame_seconds=0.08)

        self.assertEqual(frames, [1, 2, 3])

    def test_a_partial_frame_at_each_edge_is_dropped(self) -> None:
        frames = frames_inside([(0.10, 0.30)], frame_count=10, frame_seconds=0.08)

        self.assertEqual(frames, [2])

    def test_frames_past_the_end_of_the_clip_are_not_invented(self) -> None:
        frames = frames_inside([(0.0, 10.0)], frame_count=3, frame_seconds=0.08)

        self.assertEqual(frames, [0, 1, 2])

    def test_two_windows_are_merged_without_duplicates(self) -> None:
        frames = frames_inside([(0.0, 0.16), (0.08, 0.24)], frame_count=10, frame_seconds=0.08)

        self.assertEqual(frames, [0, 1, 2])

    def test_an_edge_that_divides_inexactly_still_keeps_its_frame(self) -> None:
        """0.16 / 0.08 is 1.9999999999999998 in IEEE 754; frame 1 lies exactly inside."""
        self.assertEqual(frames_inside([(0.0, 0.16)], frame_count=10, frame_seconds=0.08), [0, 1])


class LongestRunTests(unittest.TestCase):
    def test_no_flags_is_zero(self) -> None:
        self.assertEqual(longest_true_run([False, False]), 0)

    def test_the_longest_run_is_measured_not_the_last(self) -> None:
        self.assertEqual(longest_true_run([True, True, True, False, True]), 3)

    def test_an_empty_sequence_is_zero(self) -> None:
        self.assertEqual(longest_true_run([]), 0)


class ChannelVerdictTests(unittest.TestCase):
    """M3's wording - "the other channel carries none" - is false in M3-R by construction.

    Room tone is laid under the non-speaking channel and the backchannel overlaps the body
    turn on purpose. The verdict therefore asks that the speaker dominate the other channel
    on the frames the turn owns exclusively, and that the other channel stay under the
    speech threshold on those same frames.

    Dominance is a comparison, not a level: a left/right swap moves energy between channels
    and changes no absolute level, and speaker A in the shipped dataset is 19 dB quieter
    than speaker B, so any absolute floor either passes a swap or fails a real turn.
    """

    def measurement(self, **overrides) -> dict:
        row = {
            "dialogue": "v-001",
            "turn": 1,
            "speaker": "A",
            "own_exclusive_median_rms": 0.014,
            "other_exclusive_median_rms": 0.0002,
            "other_max_rms": 0.002,
            "exclusive_frames": 30,
        }
        row.update(overrides)
        return row

    def test_a_clean_turn_has_no_problem(self) -> None:
        self.assertEqual(channel_problems(self.measurement()), [])

    def test_room_tone_under_the_threshold_is_not_a_mismatch(self) -> None:
        self.assertEqual(channel_problems(self.measurement(other_max_rms=0.009)), [])

    def test_a_quiet_speaker_still_passes_while_they_dominate(self) -> None:
        """Speaker A's turns sit under the assembler's speech threshold and are still A."""
        problems = channel_problems(
            self.measurement(own_exclusive_median_rms=0.0075, other_exclusive_median_rms=0.0002)
        )

        self.assertEqual(problems, [])

    def test_a_swapped_channel_fails_even_though_both_levels_are_unchanged(self) -> None:
        problems = channel_problems(
            self.measurement(own_exclusive_median_rms=0.0002, other_exclusive_median_rms=0.014)
        )

        self.assertEqual([row["kind"] for row in problems], ["speaking_channel_not_dominant"])

    def test_speech_on_the_other_channel_outside_any_overlap_fails(self) -> None:
        problems = channel_problems(self.measurement(other_max_rms=0.04))

        self.assertEqual([row["kind"] for row in problems], ["other_channel_loud"])

    def test_a_turn_with_no_exclusive_part_is_reported_and_judged_no_further(self) -> None:
        """Clause (b) cannot be evaluated there, so passing it silently would be a lie."""
        problems = channel_problems(
            self.measurement(
                exclusive_frames=0,
                other_max_rms=9.0,
                own_exclusive_median_rms=0.0,
                other_exclusive_median_rms=0.0,
            )
        )

        self.assertEqual([row["kind"] for row in problems], ["no_exclusive_frames"])

    def test_the_problem_carries_the_dialogue_it_came_from(self) -> None:
        problems = channel_problems(self.measurement(own_exclusive_median_rms=0.0))

        self.assertEqual(problems[0]["dialogue"], "v-001")
        self.assertEqual(problems[0]["speaker"], "A")


class TextTruncationTests(unittest.TestCase):
    """tokenize_text pads to (last_token_end + 1s) * frame_rate, so the raw stream is longer
    than the audio in every dataset, M3's included. merge_text_audio cuts that padding off.
    A cut *token* is the defect the count was named for."""

    def test_trailing_padding_being_cut_is_not_a_defect(self) -> None:
        row = text_tokens_lost([7, 3, 3, 3, 3], audio_frames=3, padding_id=3)

        self.assertEqual(row["length_difference"], 2)
        self.assertEqual(row["non_padding_tokens_truncated"], 0)

    def test_a_token_past_the_audio_is_counted(self) -> None:
        row = text_tokens_lost([7, 3, 3, 9], audio_frames=3, padding_id=3)

        self.assertEqual(row["non_padding_tokens_truncated"], 1)
        self.assertEqual(row["last_non_padding_index"], 3)

    def test_a_stream_shorter_than_the_audio_loses_nothing(self) -> None:
        row = text_tokens_lost([7, 9], audio_frames=5, padding_id=3)

        self.assertEqual(row["length_difference"], -3)
        self.assertEqual(row["non_padding_tokens_truncated"], 0)


class ExcludedIdTests(unittest.TestCase):
    def test_a_dropped_dialogue_that_is_gone_everywhere_reports_nothing(self) -> None:
        sightings = excluded_id_sightings(
            ["v-047"], {"manifest": ["v-046", "v-048"], "parquet": ["train/v-046"]}
        )

        self.assertEqual(sightings, [])

    def test_a_dropped_dialogue_is_found_behind_a_split_prefix(self) -> None:
        sightings = excluded_id_sightings(["v-047"], {"parquet": ["train/v-047"]})

        self.assertEqual(
            sightings, [{"dialogue": "v-047", "place": "parquet", "found": ["train/v-047"]}]
        )

    def test_a_longer_name_that_merely_contains_the_id_is_not_a_sighting(self) -> None:
        sightings = excluded_id_sightings(["v-047"], {"manifest": ["v-0470", "xv-047"]})

        self.assertEqual(sightings, [])

    def test_each_place_is_reported_separately(self) -> None:
        sightings = excluded_id_sightings(
            ["v-047"], {"manifest": ["v-047"], "split-map": ["v-047"]}
        )

        self.assertEqual([row["place"] for row in sightings], ["manifest", "split-map"])


class ObservedCountTests(unittest.TestCase):
    """`turns_never_alone` is a shape the assembler makes on purpose - eight of the nine are
    backchannels buried inside speaker A's turn. It is recorded, not gated, and it must not
    leak into the nine counts M3 pre-registered."""

    def test_an_observed_count_does_not_enter_the_nine(self) -> None:
        problems = [{"count": "turns_never_alone"}, {"count": "channel_mismatches"}]

        counts = counts_from_problems(problems)

        self.assertEqual(counts["channel_mismatches"], 1)
        self.assertNotIn("turns_never_alone", counts)

    def test_an_observed_count_is_tallied_separately(self) -> None:
        problems = [{"count": "turns_never_alone"}, {"count": "channel_mismatches"}]

        self.assertEqual(observed_counts(problems), {"turns_never_alone": 1})


class CountTests(unittest.TestCase):
    def test_a_clean_dataset_is_nine_zeroes(self) -> None:
        counts = counts_from_problems([])

        self.assertEqual(len(counts), 9)
        self.assertEqual(set(counts.values()), {0})

    def test_problems_are_tallied_by_their_count_name(self) -> None:
        counts = counts_from_problems(
            [{"count": "channel_mismatches"}, {"count": "channel_mismatches"}]
        )

        self.assertEqual(counts["channel_mismatches"], 2)
        self.assertEqual(counts["text_mismatches"], 0)

    def test_an_unknown_count_is_refused_rather_than_dropped(self) -> None:
        with self.assertRaises(AgreementError):
            counts_from_problems([{"count": "something_else"}])


class NormaliseTests(unittest.TestCase):
    def test_punctuation_and_spacing_do_not_make_a_mismatch(self) -> None:
        self.assertEqual(normalise_text("はい、はい。"), normalise_text("はい はい"))

    def test_width_is_normalised(self) -> None:
        self.assertEqual(normalise_text("ＡＢ"), "AB")


if __name__ == "__main__":
    unittest.main()


class DominanceMarginTests(unittest.TestCase):
    def measurement(self, turns) -> dict:
        return {"per_dialogue": [{"dialogue_id": "v-001", "turns": turns}]}

    def turn(self, **overrides) -> dict:
        row = {
            "dialogue": "v-001",
            "turn": 1,
            "speaker": "A",
            "role": "body",
            "exclusive_frames": 10,
            "own_exclusive_median_rms": 0.1,
            "other_exclusive_median_rms": 0.01,
        }
        row.update(overrides)
        return row

    def test_a_ten_to_one_ratio_is_twenty_decibels(self) -> None:
        margins = dominance_margins_db(self.measurement([self.turn()]))

        self.assertEqual(len(margins), 1)
        self.assertAlmostEqual(margins[0]["margin_db"], 20.0, places=6)

    def test_a_turn_with_no_exclusive_frames_has_no_margin(self) -> None:
        margins = dominance_margins_db(self.measurement([self.turn(exclusive_frames=0)]))

        self.assertEqual(margins, [])

    def test_a_digitally_silent_channel_is_skipped_rather_than_infinite(self) -> None:
        margins = dominance_margins_db(
            self.measurement([self.turn(other_exclusive_median_rms=0.0)])
        )

        self.assertEqual(margins, [])

    def test_the_margin_carries_the_turn_it_came_from(self) -> None:
        margins = dominance_margins_db(self.measurement([self.turn(role="backchannel")]))

        self.assertEqual(margins[0]["role"], "backchannel")
        self.assertEqual(margins[0]["dialogue"], "v-001")


class AssembleReportTests(unittest.TestCase):
    """The six scripts that built v-real-v2 lived under gitignored `data/`, so the procedure
    that made the dataset was not in the repository. Assembling the report from the
    measurement files rather than from a person reading them keeps that from recurring."""

    def measured(self, **overrides) -> dict:
        row = {
            "dataset_id": "v-real-v2",
            "passed": True,
            "counts": dict.fromkeys(
                (
                    "channel_mismatches",
                    "timestamp_violations",
                    "text_mismatches",
                    "non_stereo",
                    "wrong_sample_rate",
                    "zero_length",
                    "below_min_frames",
                    "text_frames_exceeding_audio",
                    "saturated_files",
                ),
                0,
            ),
            "observed_counts": {"turns_never_alone": 1},
            "speaker_levels": {"a_minus_b_db": -19.0},
            "problems": [{"count": "turns_never_alone", "dialogue": "v-015"}],
            "excluded": {"blocking": []},
            "backchannel": {"checked": 2, "failures": [], "ncc": {"n": 2}},
            "room_tone": {"failures": []},
            "per_dialogue": [
                {
                    "dialogue_id": "v-001",
                    "split": "train",
                    "frames": 240,
                    "duration_seconds": 19.2,
                    "text": [{"turn": 0}],
                    "turns": [
                        {
                            "dialogue": "v-001",
                            "turn": 0,
                            "speaker": "A",
                            "role": "body",
                            "exclusive_frames": 5,
                            "own_exclusive_median_rms": 0.1,
                            "other_exclusive_median_rms": 0.01,
                        }
                    ],
                    "text_streams": [{"length_difference": -17, "non_padding_tokens_truncated": 0}],
                }
            ],
        }
        row.update(overrides)
        return row

    def swapped(self) -> dict:
        row = self.measured(passed=False)
        row["counts"]["channel_mismatches"] = 1
        row["problems"] = [{"count": "channel_mismatches", "dialogue": "v-001"}]
        turn = row["per_dialogue"][0]["turns"][0]
        turn["own_exclusive_median_rms"], turn["other_exclusive_median_rms"] = (0.01, 0.1)
        row["backchannel"]["ncc"] = {"n": 2, "median": 0.18}
        return row

    def test_nine_zeroes_pass_the_nine_even_when_an_addition_fails(self) -> None:
        report = assemble_report(
            measured=self.measured(passed=False),
            swapped=self.swapped(),
            parquet={"train": {"rows": 70}},
        )

        self.assertEqual(report["nine_counts_status"], "pass")
        self.assertEqual(report["status"], "fail")

    def test_the_negative_control_is_carried_beside_the_measurement(self) -> None:
        report = assemble_report(measured=self.measured(), swapped=self.swapped(), parquet={})

        self.assertEqual(report["negative_control"]["counts"]["channel_mismatches"], 1)
        self.assertEqual(report["negative_control"]["dialogues_with_a_channel_mismatch"], 1)

    def test_the_margins_of_both_runs_are_reported_so_the_gate_has_a_band(self) -> None:
        report = assemble_report(measured=self.measured(), swapped=self.swapped(), parquet={})

        margin = report["channel_margin_db"]
        self.assertAlmostEqual(margin["shipped"]["min"], 20.0, places=6)
        self.assertAlmostEqual(margin["channels_swapped"]["max"], -20.0, places=6)
        self.assertAlmostEqual(margin["separation_db"], 40.0, places=6)
        self.assertEqual(margin["shipped_turns_at_or_below_zero"], 0)

    def test_annotations_are_merged_over_the_numbers_without_replacing_a_branch(self) -> None:
        report = assemble_report(
            measured=self.measured(),
            swapped=self.swapped(),
            parquet={},
            annotations={"condition": "1", "scope": {"note": "authored"}},
        )

        self.assertEqual(report["condition"], "1")
        self.assertEqual(report["scope"]["note"], "authored")
        self.assertEqual(report["scope"]["dialogues"], 1)

    def test_splits_are_summarised_from_the_dialogues_not_declared(self) -> None:
        report = assemble_report(measured=self.measured(), swapped=self.swapped(), parquet={})

        self.assertEqual(report["scope"]["splits"]["train"]["dialogues"], 1)
        self.assertEqual(report["scope"]["splits"]["train"]["frames"]["min"], 240)
