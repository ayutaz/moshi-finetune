"""Gates for the parquet -> Mimi -> WAV round trip.

The point of `tools/roundtrip_audio.py` is that a swapped, doubled or broken channel does
not show up in a loss curve. These tests hold the parts of that check that can be wrong
without anything raising: the row order handed to the decoder, the assignment verdict, and
the frame indices the silence gate is computed over.

Nothing here imports numpy, torch, soundfile or pandas - the whole suite has to run without
them, which is why every function under test takes plain sequences.
"""

from __future__ import annotations

import math

import pytest

from tools.roundtrip_audio import (
    RoundtripShapeError,
    band_energy_ratio,
    best_lag_correlation,
    channel_assignment,
    clipping_stats,
    decoder_token_rows,
    digitally_silent_mask,
    evenly_spaced,
    frame_peak,
    frame_rms,
    leave_one_out_against,
    mask_agreement,
    pearson,
    quiet_frame_indices,
    spectral_centroid_hz,
    split_speaker_column,
    spread,
    tokens_at,
    voiced_mask,
)


def column(fill: int, frames: int = 4, streams: int = 9) -> list[list[int]]:
    return [[fill + row for _ in range(frames)] for row in range(streams)]


def test_split_speaker_column_returns_text_row_then_codebooks():
    text, codebooks = split_speaker_column(column(100))
    assert text == [100, 100, 100, 100]
    assert len(codebooks) == 8
    assert codebooks[0] == [101, 101, 101, 101]
    assert codebooks[-1] == [108, 108, 108, 108]


def test_split_speaker_column_rejects_a_column_that_lost_its_text_row():
    with pytest.raises(RoundtripShapeError, match="8 streams"):
        split_speaker_column(column(0, streams=8))


def test_split_speaker_column_rejects_a_ragged_column():
    cell = column(0)
    cell[3] = cell[3][:-1]
    with pytest.raises(RoundtripShapeError, match="rectangle"):
        split_speaker_column(cell)


def test_split_speaker_column_rejects_an_empty_column():
    with pytest.raises(RoundtripShapeError, match="no frames"):
        split_speaker_column(column(0, frames=0))


def test_decoder_token_rows_puts_speaker_a_first():
    """decode_audio splits the block down the middle, so row order IS channel order.

    Speaker A came from source channel 0. If this ever returns B first, every decode is
    swapped relative to the recording and the swap detector agrees with itself.
    """
    block = decoder_token_rows(column(100), column(200))
    assert len(block) == 16
    assert block[0] == [101, 101, 101, 101]
    assert block[7] == [108, 108, 108, 108]
    assert block[8] == [201, 201, 201, 201]
    assert block[15] == [208, 208, 208, 208]


def test_decoder_token_rows_drops_no_codebook():
    block = decoder_token_rows(column(100), column(200))
    assert [row[0] for row in block] == [101, 102, 103, 104, 105, 106, 107, 108] + [
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
    ]


def test_decoder_token_rows_rejects_channels_of_different_length():
    with pytest.raises(RoundtripShapeError, match="cannot differ in length"):
        decoder_token_rows(column(100, frames=4), column(200, frames=5))


def test_frame_rms_measures_whole_frames_only():
    samples = [1.0] * 4 + [0.0] * 4 + [0.5]
    assert frame_rms(samples, hop=4) == [1.0, 0.0]


def test_frame_rms_rejects_a_zero_hop():
    with pytest.raises(ValueError, match="hop must be positive"):
        frame_rms([1.0, 2.0], hop=0)


def test_pearson_is_one_for_a_scaled_copy():
    left = [0.0, 1.0, 2.0, 3.0]
    right = [1.0, 3.0, 5.0, 7.0]
    assert pearson(left, right) == pytest.approx(1.0)


def test_pearson_is_minus_one_for_a_mirrored_signal():
    assert pearson([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]) == pytest.approx(-1.0)


def test_pearson_rejects_a_constant_channel():
    """A dead channel must not pass the swap check by being equally unlike both sources."""
    with pytest.raises(ValueError, match="constant signal"):
        pearson([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


def test_pearson_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="lengths differ"):
        pearson([1.0, 2.0], [1.0, 2.0, 3.0])


def test_voiced_mask_uses_the_builders_threshold():
    assert voiced_mask([0.0, 0.009, 0.01, 0.2]) == [False, False, True, True]


def test_quiet_frame_indices_is_the_complement_of_voiced_mask():
    envelope = [0.0, 0.009, 0.01, 0.2, 0.001]
    quiet = quiet_frame_indices(envelope)
    assert quiet == [0, 1, 4]
    assert [i for i, v in enumerate(voiced_mask(envelope)) if not v] == quiet


def test_mask_agreement_counts_every_frame():
    left = [True, True, False, False]
    right = [True, False, True, False]
    result = mask_agreement(left, right)
    assert (result["both"], result["left_only"], result["right_only"], result["neither"]) == (
        1,
        1,
        1,
        1,
    )
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["accuracy"] == pytest.approx(0.5)


def test_mask_agreement_reports_iou_one_for_two_silent_channels():
    """Two all-silent masks agree perfectly; a 0/0 union must not be a ZeroDivisionError."""
    assert mask_agreement([False, False], [False, False])["iou"] == 1.0


def test_mask_agreement_rejects_different_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        mask_agreement([True], [True, False])


def test_channel_assignment_calls_a_clean_match_identity():
    scores = {"A": {"A": 0.95, "B": 0.10}, "B": {"A": 0.08, "B": 0.91}}
    result = channel_assignment(scores)
    assert result["verdict"] == "identity"
    assert result["min_margin"] == pytest.approx(0.83)


def test_channel_assignment_detects_a_swap():
    """The defect that leaves both channels individually perfect."""
    scores = {"A": {"A": 0.10, "B": 0.95}, "B": {"A": 0.91, "B": 0.08}}
    result = channel_assignment(scores)
    assert result["verdict"] == "swapped"
    assert result["best_match"] == {"A": "B", "B": "A"}


def test_channel_assignment_calls_a_half_match_ambiguous():
    """One confident channel must not carry a channel that matches the wrong source."""
    scores = {"A": {"A": 0.95, "B": 0.10}, "B": {"A": 0.80, "B": 0.20}}
    assert channel_assignment(scores)["verdict"] == "ambiguous"


def test_channel_assignment_rejects_a_matrix_that_is_not_two_by_two():
    with pytest.raises(ValueError, match="exactly two channels"):
        channel_assignment({"A": {"A": 1.0}})


def test_channel_assignment_rejects_a_missing_cross_score():
    with pytest.raises(ValueError, match="no entry for"):
        channel_assignment({"A": {"A": 1.0}, "B": {"B": 1.0}})


def test_spectral_centroid_of_a_single_bin_is_that_bin():
    assert spectral_centroid_hz([0.0, 1.0, 0.0], [0.0, 1000.0, 2000.0]) == pytest.approx(1000.0)


def test_spectral_centroid_is_the_energy_weighted_mean():
    assert spectral_centroid_hz([1.0, 3.0], [0.0, 4000.0]) == pytest.approx(3000.0)


def test_spectral_centroid_rejects_a_silent_spectrum():
    with pytest.raises(ValueError, match="no energy"):
        spectral_centroid_hz([0.0, 0.0], [0.0, 1000.0])


def test_band_energy_ratio_uses_squared_magnitudes():
    """Half the magnitude is a quarter of the energy; a magnitude-weighted ratio would say half."""
    ratio = band_energy_ratio([2.0, 1.0], [0.0, 5000.0], cutoff_hz=4000.0)
    assert ratio == pytest.approx(1.0 / 5.0)


def test_band_energy_ratio_includes_the_cutoff_bin():
    assert band_energy_ratio([0.0, 1.0], [0.0, 4000.0], cutoff_hz=4000.0) == pytest.approx(1.0)


def test_band_energy_ratio_rejects_a_silent_spectrum():
    with pytest.raises(ValueError, match="no energy"):
        band_energy_ratio([0.0], [0.0], cutoff_hz=100.0)


def test_clipping_stats_counts_saturated_samples():
    result = clipping_stats([0.1, -0.99, 1.0, 0.5])
    assert result["clipped"] == 2
    assert result["clipped_share"] == pytest.approx(0.5)
    assert result["peak"] == pytest.approx(1.0)


def test_clipping_stats_rejects_an_empty_waveform():
    with pytest.raises(ValueError, match="empty waveform"):
        clipping_stats([])


def test_tokens_at_picks_the_named_frames():
    assert tokens_at([10, 11, 12, 13], [0, 3]) == [10, 13]


def test_tokens_at_raises_rather_than_shrinking_the_sample():
    """A numpy fancy-index guard would drop the stray index and quietly shrink the gate."""
    with pytest.raises(IndexError, match="outside the 4-frame row"):
        tokens_at([10, 11, 12, 13], [0, 9])


def test_evenly_spaced_covers_both_ends():
    picked = evenly_spaced(list(range(70)), 8)
    assert picked[0] == 0
    assert picked[-1] == 69
    assert len(picked) == 8


def test_evenly_spaced_is_deterministic():
    items = [f"v-{i:03d}" for i in range(70)]
    assert evenly_spaced(items, 8) == evenly_spaced(items, 8)


def test_evenly_spaced_returns_everything_when_asked_for_more_than_exists():
    assert evenly_spaced([1, 2, 3], 10) == [1, 2, 3]


def test_evenly_spaced_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="empty sequence"):
        evenly_spaced([], 3)


def test_evenly_spaced_rejects_a_non_positive_count():
    with pytest.raises(ValueError, match="count must be positive"):
        evenly_spaced([1, 2], 0)


def test_spread_reports_the_four_numbers():
    result = spread([1.0, 2.0, 3.0, 10.0])
    assert result["n"] == 4
    assert result["min"] == 1.0
    assert result["max"] == 10.0
    assert result["median"] == pytest.approx(2.5)
    assert result["mean"] == pytest.approx(4.0)


def test_spread_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        spread([])


def test_a_swapped_pipeline_is_caught_end_to_end():
    """The whole point, assembled from the pure parts: A loud early, B loud late.

    If the decode came back with the channels exchanged, every per-channel statistic would
    still look healthy and only the assignment would notice.
    """
    hop = 4
    source_a = [1.0] * hop + [0.0] * hop
    source_b = [0.0] * hop + [1.0] * hop
    decoded_a, decoded_b = source_b, source_a  # the bug

    envelopes = {
        ("source", "A"): frame_rms(source_a, hop=2),
        ("source", "B"): frame_rms(source_b, hop=2),
        ("decoded", "A"): frame_rms(decoded_a, hop=2),
        ("decoded", "B"): frame_rms(decoded_b, hop=2),
    }
    matrix = {
        src: {dec: pearson(envelopes[("source", src)], envelopes[("decoded", dec)]) for dec in "AB"}
        for src in "AB"
    }
    assert channel_assignment(matrix)["verdict"] == "swapped"
    # ... while each channel on its own is a perfect match for *something*.
    assert math.isclose(max(matrix["A"].values()), 1.0)
    assert math.isclose(max(matrix["B"].values()), 1.0)


def test_leave_one_out_against_excludes_the_probes_own_twin():
    """A round-tripped clip scored against a centroid holding its own natural twin inflates."""
    reference = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]}
    scores = leave_one_out_against(reference, {"a": [1.0, 0.0]})
    # centroid of b and c is (0.5, 0.5); with a included it would have been nearer (1, 0).
    assert scores["a"] == pytest.approx(1 / math.sqrt(2))


def test_leave_one_out_against_scores_every_probe():
    reference = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    scores = leave_one_out_against(reference, dict(reference))
    assert sorted(scores) == ["a", "b", "c"]


def test_leave_one_out_against_rejects_a_probe_with_no_reference():
    reference = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    with pytest.raises(ValueError, match="no reference recording"):
        leave_one_out_against(reference, {"d": [1.0, 0.0]})


def test_leave_one_out_against_rejects_a_reference_set_too_small_for_a_band():
    with pytest.raises(ValueError, match="at least 3 reference"):
        leave_one_out_against({"a": [1.0, 0.0], "b": [0.0, 1.0]}, {"a": [1.0, 0.0]})


def test_best_lag_is_zero_for_an_aligned_pair():
    envelope = [0.0, 1.0, 0.5, 0.0, 0.9, 0.1]
    result = best_lag_correlation(envelope, envelope, max_lag=3)
    assert result["lag_frames"] == 0
    assert result["correlation"] == pytest.approx(1.0)


def test_best_lag_finds_a_delay_baked_into_the_decode():
    """A delay written into the parquet would be applied twice at training time.

    Every statistic taken at lag zero still looks plausible when that happens - the shapes
    are merely shifted - so the shift has to be searched for explicitly.
    """
    source = [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.9, 0.2, 0.0, 0.0]
    decoded = [0.0, 0.0] + source[:-2]  # the decode arrives two frames late
    result = best_lag_correlation(source, decoded, max_lag=4)
    assert result["lag_frames"] == -2
    assert result["correlation"] == pytest.approx(1.0)


def test_best_lag_rejects_a_negative_search_width():
    with pytest.raises(ValueError, match="max_lag cannot be negative"):
        best_lag_correlation([1.0, 2.0], [1.0, 2.0], max_lag=-1)


def test_best_lag_raises_when_no_lag_leaves_a_comparable_overlap():
    with pytest.raises(ValueError, match="comparable overlap"):
        best_lag_correlation([1.0], [1.0], max_lag=2)


def test_frame_peak_takes_the_largest_absolute_sample():
    assert frame_peak([0.1, -0.9, 0.2, 0.0, 0.0, 0.3], hop=3) == [0.9, 0.3]


def test_frame_peak_drops_a_trailing_partial_frame():
    assert frame_peak([1.0, 1.0, 1.0, 0.5], hop=3) == [1.0]


def test_digitally_silent_mask_separates_exact_zero_from_merely_quiet():
    """Room tone at -60 dBFS is quiet; only an exact zero gets Mimi's silence code."""
    assert digitally_silent_mask([0.0, 1e-9, 0.0, 0.5]) == [True, False, True, False]
