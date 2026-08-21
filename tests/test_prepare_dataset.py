import unittest

from tools.prepare_dataset import StemMismatchError, dialogue_id_for, matched_dialogue_stems


class MatchedDialogueStemsTests(unittest.TestCase):
    def test_matching_sets_return_a_sorted_list(self) -> None:
        # Sorted, not os.listdir order: the parquet row order decides which dialogue lands
        # in which batch, and an arbitrary order makes a run unreproducible.
        stems = matched_dialogue_stems(["v-002", "v-001"], ["v-001", "v-002"])
        self.assertEqual(stems, ["v-001", "v-002"])

    def test_missing_text_raises_with_the_names(self) -> None:
        with self.assertRaises(StemMismatchError) as caught:
            matched_dialogue_stems(["v-001"], ["v-001", "v-002"])
        message = str(caught.exception)
        self.assertIn("v-002", message)
        self.assertIn("text", message)

    def test_missing_audio_raises_with_the_names(self) -> None:
        with self.assertRaises(StemMismatchError) as caught:
            matched_dialogue_stems(["v-001", "v-002"], ["v-001"])
        message = str(caught.exception)
        self.assertIn("v-002", message)
        self.assertIn("audio", message)

    def test_both_directions_are_reported_together(self) -> None:
        with self.assertRaises(StemMismatchError) as caught:
            matched_dialogue_stems(["v-001", "v-003"], ["v-001", "v-002"])
        message = str(caught.exception)
        self.assertIn("v-002", message)
        self.assertIn("v-003", message)

    def test_an_empty_pair_raises_rather_than_writing_an_empty_parquet(self) -> None:
        # Two empty directories match perfectly. Without this the script would write a
        # zero-row parquet and exit 0, and the failure would surface as a training run
        # with no data on a billing GPU.
        with self.assertRaises(StemMismatchError):
            matched_dialogue_stems([], [])

    def test_the_error_is_not_swallowed_as_a_plain_valueerror(self) -> None:
        self.assertTrue(issubclass(StemMismatchError, Exception))


class DialogueIdTests(unittest.TestCase):
    def test_the_id_is_namespaced_by_split_not_by_local_path(self) -> None:
        # Upstream joined the whole --output_prefix, so a parquet built on a laptop and
        # uploaded to a rented instance carried the laptop's absolute directories.
        self.assertEqual(dialogue_id_for("/some/where/deep/train", "v-001"), "train/v-001")

    def test_a_bare_prefix_still_namespaces(self) -> None:
        self.assertEqual(dialogue_id_for("dev", "v-008"), "dev/v-008")

    def test_train_and_dev_ids_do_not_collide(self) -> None:
        self.assertNotEqual(
            dialogue_id_for("/a/train", "v-001"), dialogue_id_for("/a/dev", "v-001")
        )


if __name__ == "__main__":
    unittest.main()
