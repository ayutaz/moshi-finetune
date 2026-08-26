import unittest

from tools.parquet_shape import (
    EXPECTED_STREAMS_PER_EXAMPLE,
    ParquetShapeError,
    stream_shape_problems,
    summarise_streams,
)


def example(streams: int, frames: int) -> list[list[int]]:
    return [[0] * frames for _ in range(streams)]


class SummariseStreamsTests(unittest.TestCase):
    """Counted on the trainer's own collation, not on the column shape prepare_dataset wrote.

    The two are only equal while `utils.data.main_speaker_streams` behaves the way whoever
    read the column shape assumed. A parquet that comes out at anything but seventeen
    streams trains anyway, on an alignment the delay pattern does not describe.
    """

    def test_a_uniform_split_reports_one_stream_count(self) -> None:
        summary = summarise_streams([example(17, 276), example(17, 242)])

        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["streams_per_example"], [17])

    def test_the_frame_statistics_come_from_the_streams(self) -> None:
        summary = summarise_streams([example(17, 195), example(17, 276), example(17, 299)])

        self.assertEqual(summary["frames"], {"min": 195, "max": 299, "median": 276.0})

    def test_a_mixed_split_reports_every_stream_count(self) -> None:
        summary = summarise_streams([example(17, 100), example(9, 100)])

        self.assertEqual(summary["streams_per_example"], [9, 17])

    def test_a_ragged_example_is_refused(self) -> None:
        """An example is a rectangle by construction; a ragged one means a row was padded."""
        with self.assertRaises(ParquetShapeError) as raised:
            summarise_streams([[[0] * 10, [0] * 11]])

        self.assertIn("rectangle", str(raised.exception))

    def test_an_example_with_no_streams_is_refused(self) -> None:
        with self.assertRaises(ParquetShapeError):
            summarise_streams([[]])

    def test_an_empty_split_is_refused(self) -> None:
        """Zero examples would otherwise pass every gate below it."""
        with self.assertRaises(ParquetShapeError):
            summarise_streams([])


class StreamShapeGateTests(unittest.TestCase):
    def test_seventeen_streams_passes(self) -> None:
        summary = summarise_streams([example(EXPECTED_STREAMS_PER_EXAMPLE, 276)])

        self.assertEqual(stream_shape_problems(summary), [])

    def test_nine_streams_fails(self) -> None:
        """The negative control: the speaker column's own shape is not the example's."""
        summary = summarise_streams([example(9, 276)])

        problems = stream_shape_problems(summary)

        self.assertEqual(len(problems), 1)
        self.assertIn("[9]", problems[0])

    def test_a_split_that_is_only_sometimes_right_fails(self) -> None:
        summary = summarise_streams([example(17, 276), example(16, 276)])

        self.assertTrue(stream_shape_problems(summary))


if __name__ == "__main__":
    unittest.main()
