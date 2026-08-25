import random
import unittest

from tools.room_tone import (
    FLOOR_WITH_EVENTS,
    QUIET_FLOOR,
    RenderSpec,
    deck_order,
    equal_power_ramp,
    gate_verdict,
    level_in_db,
    plan_units,
    runs_in_band,
    sources_excluding,
    token_stats,
    trim_runs,
)


class RunsInBandTests(unittest.TestCase):
    def test_a_contiguous_stretch_inside_the_band_is_one_run(self) -> None:
        self.assertEqual(runs_in_band([0.0, 3.0, 3.0, 3.0, 0.0], low=1.0, high=5.0), [(1, 4)])

    def test_the_band_is_half_open_so_the_upper_edge_is_excluded(self) -> None:
        # Speech starts somewhere; a value sitting exactly on the ceiling belongs to it.
        self.assertEqual(runs_in_band([2.0, 5.0, 2.0], low=1.0, high=5.0), [(0, 1), (2, 3)])

    def test_a_run_reaching_the_end_is_closed_at_the_end(self) -> None:
        self.assertEqual(runs_in_band([0.0, 3.0, 3.0], low=1.0, high=5.0), [(1, 3)])

    def test_nothing_in_band_gives_no_runs(self) -> None:
        self.assertEqual(runs_in_band([0.0, 0.0], low=1.0, high=5.0), [])


class TrimRunsTests(unittest.TestCase):
    def test_the_guard_is_taken_off_both_ends(self) -> None:
        self.assertEqual(trim_runs([(10, 30)], guard_frames=3, min_frames=1), [(13, 27)])

    def test_a_run_shorter_than_the_minimum_after_trimming_is_dropped(self) -> None:
        self.assertEqual(trim_runs([(10, 16)], guard_frames=2, min_frames=4), [])

    def test_a_run_exactly_the_minimum_is_kept(self) -> None:
        self.assertEqual(trim_runs([(10, 16)], guard_frames=1, min_frames=4), [(11, 15)])

    def test_a_zero_guard_leaves_the_run_alone(self) -> None:
        # The collected band *is* the decay tail, so the guard has to be able to be nothing.
        self.assertEqual(trim_runs([(4, 9)], guard_frames=0, min_frames=1), [(4, 9)])

    def test_a_negative_guard_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            trim_runs([(0, 10)], guard_frames=-1, min_frames=1)


class EqualPowerRampTests(unittest.TestCase):
    def test_the_ramp_runs_from_silence_to_full(self) -> None:
        ramp = equal_power_ramp(9)
        self.assertAlmostEqual(ramp[0], 0.0)
        self.assertAlmostEqual(ramp[-1], 1.0)

    def test_power_is_conserved_across_the_crossfade(self) -> None:
        # Two uncorrelated recordings add in power, not amplitude. A linear fade would dip.
        ramp = equal_power_ramp(16)
        for rising, falling in zip(ramp, reversed(ramp), strict=True):
            self.assertAlmostEqual(rising**2 + falling**2, 1.0)

    def test_a_single_sample_ramp_is_full_scale(self) -> None:
        self.assertEqual(equal_power_ramp(1), [1.0])

    def test_a_zero_length_ramp_is_empty(self) -> None:
        self.assertEqual(equal_power_ramp(0), [])


class DeckOrderTests(unittest.TestCase):
    def test_no_segment_repeats_before_every_segment_has_been_used(self) -> None:
        # A pool that repeats early is a loop, and a loop teaches its own period.
        order = deck_order(6, 18, random.Random(0))
        for start in (0, 6, 12):
            self.assertEqual(sorted(order[start : start + 6]), list(range(6)))

    def test_the_same_seed_gives_the_same_order(self) -> None:
        self.assertEqual(deck_order(8, 20, random.Random(4)), deck_order(8, 20, random.Random(4)))

    def test_different_seeds_give_different_orders(self) -> None:
        self.assertNotEqual(
            deck_order(20, 20, random.Random(1)), deck_order(20, 20, random.Random(2))
        )

    def test_an_empty_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            deck_order(0, 3, random.Random(0))


SPEC = RenderSpec(
    levels=(1e-4, 2e-4, 4e-4),
    level_weights=(1.0, 1.0, 1.0),
    eq_db=28.0,
    speed_jitter=0.15,
    crossfade_seconds=0.02,
)


class PlanUnitsTests(unittest.TestCase):
    durations = [0.10, 0.14, 0.22, 0.11, 0.30, 0.18]

    def test_the_plan_covers_the_length_asked_for(self) -> None:
        plan = plan_units(self.durations, 12.0, spec=SPEC, seed=3)
        covered = sum(
            max(
                self.durations[d.segment] / d.speed - SPEC.crossfade_seconds, SPEC.crossfade_seconds
            )
            for d in plan
        )
        self.assertGreaterEqual(covered, 12.0)

    def test_the_plan_does_not_overshoot_by_more_than_one_unit(self) -> None:
        plan = plan_units(self.durations, 12.0, spec=SPEC, seed=3)
        covered = sum(
            max(
                self.durations[d.segment] / d.speed - SPEC.crossfade_seconds, SPEC.crossfade_seconds
            )
            for d in plan[:-1]
        )
        self.assertLess(covered, 12.0)

    def test_the_same_seed_reproduces_every_decision(self) -> None:
        self.assertEqual(
            plan_units(self.durations, 5.0, spec=SPEC, seed=11),
            plan_units(self.durations, 5.0, spec=SPEC, seed=11),
        )

    def test_a_different_seed_changes_the_plan(self) -> None:
        self.assertNotEqual(
            plan_units(self.durations, 5.0, spec=SPEC, seed=11),
            plan_units(self.durations, 5.0, spec=SPEC, seed=12),
        )

    def test_every_level_in_the_spec_gets_used(self) -> None:
        # The level set is the whole reason the token histogram spreads; a plan that only
        # ever picks one level would pass every other check and still be a single-code bed.
        plan = plan_units(self.durations, 60.0, spec=SPEC, seed=7)
        self.assertEqual({d.level for d in plan}, set(SPEC.levels))

    def test_speed_stays_inside_the_jitter(self) -> None:
        for draw in plan_units(self.durations, 20.0, spec=SPEC, seed=5):
            self.assertGreaterEqual(draw.speed, 1.0 - SPEC.speed_jitter)
            self.assertLessEqual(draw.speed, 1.0 + SPEC.speed_jitter)

    def test_eq_gains_cover_every_band_and_stay_inside_the_range(self) -> None:
        for draw in plan_units(self.durations, 8.0, spec=SPEC, seed=5):
            self.assertEqual(len(draw.eq_gains), len(SPEC.eq_bands_hz))
            for gain in draw.eq_gains:
                self.assertLessEqual(abs(gain), SPEC.eq_db)

    def test_an_empty_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_units([], 1.0, spec=SPEC, seed=0)

    def test_a_non_positive_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_units(self.durations, 0.0, spec=SPEC, seed=0)

    def test_a_spec_with_no_levels_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_units(self.durations, 1.0, spec=RenderSpec(levels=(), level_weights=()), seed=0)

    def test_weights_that_do_not_match_the_levels_are_rejected(self) -> None:
        # Silently recycling or truncating would change the level distribution, which is the
        # one parameter the token histogram actually depends on.
        with self.assertRaises(ValueError):
            plan_units(
                self.durations,
                1.0,
                spec=RenderSpec(levels=(1e-4, 2e-4), level_weights=(1.0,)),
                seed=0,
            )

    def test_rare_levels_stay_rare_and_still_appear(self) -> None:
        # The loud events are what stops a long gap settling on one code, and the quiet
        # floor is what keeps them inaudible. Both halves of that have to hold.
        spec = RenderSpec(levels=(1e-4, 2e-4, 4e-4, 3.2e-3, 6.4e-3), level_weights=(9, 9, 9, 1, 1))
        plan = plan_units(self.durations, 400.0, spec=spec, seed=2)
        loud = [d for d in plan if d.level > 1e-3]
        self.assertGreater(len(loud), 0)
        self.assertLess(len(loud) / len(plan), 0.15)
        self.assertEqual({d.level for d in plan}, set(spec.levels))

    def test_a_zero_weight_level_is_never_drawn(self) -> None:
        spec = RenderSpec(levels=(1e-4, 9e-3), level_weights=(1.0, 0.0))
        self.assertEqual(
            {d.level for d in plan_units(self.durations, 60.0, spec=spec, seed=1)}, {1e-4}
        )


class PresetTests(unittest.TestCase):
    """The two presets differ only in whether rare loud events are drawn."""

    def test_the_default_spec_is_the_preset_with_events(self) -> None:
        self.assertEqual(RenderSpec(), FLOOR_WITH_EVENTS)

    def test_the_quiet_preset_has_no_loud_level(self) -> None:
        self.assertTrue(all(level < 1e-3 for level in QUIET_FLOOR.levels))

    def test_the_event_preset_keeps_the_same_quiet_floor(self) -> None:
        quiet = [level for level in FLOOR_WITH_EVENTS.levels if level < 1e-3]
        self.assertEqual(tuple(quiet), QUIET_FLOOR.levels)

    def test_an_omitted_weight_tuple_falls_back_to_uniform(self) -> None:
        spec = RenderSpec(levels=(1e-4, 2e-4), level_weights=())
        self.assertEqual(spec.weights, (1.0, 1.0))


class SourcesExcludingTests(unittest.TestCase):
    """Room tone from a held-out recording would put that recording into training."""

    def test_a_held_out_stem_is_dropped_whatever_directory_it_came_from(self) -> None:
        kept = sources_excluding(
            ["/corpus/VOICEACTRESS100_026.wav", "/corpus/VOICEACTRESS100_001.wav"],
            ["/eval/heldout/VOICEACTRESS100_026.wav"],
        )
        self.assertEqual(kept, ["/corpus/VOICEACTRESS100_001.wav"])

    def test_nothing_held_out_keeps_everything(self) -> None:
        paths = ["/corpus/a.wav", "/corpus/b.wav"]
        self.assertEqual(sources_excluding(paths, []), paths)

    def test_the_order_of_the_sources_is_preserved(self) -> None:
        paths = ["/c/c.wav", "/c/a.wav", "/c/b.wav"]
        self.assertEqual(sources_excluding(paths, ["/h/a.wav"]), ["/c/c.wav", "/c/b.wav"])


class TokenStatsTests(unittest.TestCase):
    def test_a_single_repeated_token_is_the_degenerate_case(self) -> None:
        stats = token_stats([1316] * 40)
        self.assertEqual(stats["distinct"], 1)
        self.assertAlmostEqual(stats["top_share"], 1.0)
        self.assertAlmostEqual(stats["entropy_bits"], 0.0)
        self.assertAlmostEqual(stats["silence_token_share"], 1.0)

    def test_the_silence_token_is_counted_separately_from_the_top_token(self) -> None:
        stats = token_stats([1316, 1316, 7, 7, 7])
        self.assertEqual(stats["top_token"], 7)
        self.assertAlmostEqual(stats["silence_token_share"], 0.4)

    def test_a_uniform_histogram_has_log2_entropy(self) -> None:
        self.assertAlmostEqual(token_stats([1, 2, 3, 4])["entropy_bits"], 2.0)

    def test_no_tokens_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            token_stats([])


class GateTests(unittest.TestCase):
    def test_both_conditions_must_hold(self) -> None:
        self.assertTrue(gate_verdict({"distinct": 40, "top_share": 0.30})["passed"])
        self.assertFalse(gate_verdict({"distinct": 40, "top_share": 0.44})["passed"])
        self.assertFalse(gate_verdict({"distinct": 20, "top_share": 0.10})["passed"])

    def test_the_boundary_counts_as_a_pass(self) -> None:
        self.assertTrue(gate_verdict({"distinct": 35, "top_share": 0.35})["passed"])


class LevelInDbTests(unittest.TestCase):
    def test_the_speech_reference_is_zero_db(self) -> None:
        self.assertAlmostEqual(level_in_db(0.0258), 0.0, places=6)

    def test_a_tenth_of_the_reference_is_twenty_db_down(self) -> None:
        self.assertAlmostEqual(level_in_db(0.00258), -20.0, places=6)

    def test_digital_silence_has_no_level(self) -> None:
        with self.assertRaises(ValueError):
            level_in_db(0.0)


if __name__ == "__main__":
    unittest.main()
