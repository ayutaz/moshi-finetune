import unittest

from tools.training_shape import (
    checkpoint_steps,
    global_batch_size,
    steps_per_epoch,
    total_steps,
)


class GlobalBatchSizeTests(unittest.TestCase):
    def test_it_multiplies_devices_and_accumulation(self) -> None:
        # finetune.py:736 - per_device * num_processes * gradient_accumulation
        self.assertEqual(global_batch_size(per_device=1, processes=2, gradient_accumulation=4), 8)

    def test_a_single_process_run_is_smaller(self) -> None:
        self.assertEqual(global_batch_size(per_device=1, processes=1, gradient_accumulation=4), 4)

    def test_a_non_positive_factor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            global_batch_size(per_device=0, processes=2, gradient_accumulation=4)


class StepsPerEpochTests(unittest.TestCase):
    def test_it_rounds_up_so_the_tail_batch_counts(self) -> None:
        # 72 rows at batch 8 is exactly 9; 73 would need a tenth, partial, step.
        self.assertEqual(steps_per_epoch(examples=72, batch=8), 9)
        self.assertEqual(steps_per_epoch(examples=73, batch=8), 10)

    def test_an_empty_dataset_is_rejected_rather_than_returning_zero(self) -> None:
        with self.assertRaises(ValueError):
            steps_per_epoch(examples=0, batch=8)


class TotalStepsTests(unittest.TestCase):
    def test_it_is_epochs_times_steps_per_epoch(self) -> None:
        self.assertEqual(total_steps(examples=72, batch=8, epochs=5), 45)


class CheckpointStepsTests(unittest.TestCase):
    """finetune.py saves in-loop on save_steps AND unconditionally after the loop."""

    def test_saving_every_epoch_yields_one_checkpoint_per_epoch(self) -> None:
        # 72 rows: S is 9, saves land on 9/18/27/36/45, and the post-loop save reuses 45.
        self.assertEqual(
            checkpoint_steps(examples=72, batch=8, epochs=5, save_steps=9), [9, 18, 27, 36, 45]
        )

    def test_an_unaligned_save_interval_adds_a_sixth_checkpoint(self) -> None:
        # 80 rows: S is 10, so total is 50. In-loop saves still land on multiples of 9,
        # and the post-loop save writes step_50 - six directories, not five, and only one
        # of them on an epoch boundary. This is the case that overruns the disk.
        self.assertEqual(
            checkpoint_steps(examples=80, batch=8, epochs=5, save_steps=9), [9, 18, 27, 36, 45, 50]
        )

    def test_the_post_loop_save_is_never_double_counted(self) -> None:
        steps = checkpoint_steps(examples=72, batch=8, epochs=5, save_steps=9)
        self.assertEqual(len(steps), len(set(steps)))

    def test_a_save_interval_larger_than_the_run_still_saves_once(self) -> None:
        self.assertEqual(checkpoint_steps(examples=72, batch=8, epochs=5, save_steps=999), [45])


if __name__ == "__main__":
    unittest.main()
