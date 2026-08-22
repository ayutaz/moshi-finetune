import argparse
import unittest


def _postprocess(args: argparse.Namespace) -> argparse.Namespace:
    """The `with_tracking` half of finetune.py's postprocess_args, lifted verbatim.

    finetune.py imports torch at module scope, so the suite cannot import it. Copying the
    one line under test keeps the check runnable; if finetune.py changes this line and this
    file is not updated, the copy is wrong and the test is worthless - so it asserts the
    line still reads the way it does, below.
    """
    args.with_tracking = args.report_to is not None
    return args


class WithTrackingTests(unittest.TestCase):
    """Every run without --report_to used to die before the first training step.

    `args.with_tracking = True` sat inside `if args.report_to is not None`, so omitting the
    flag left the attribute unset and finetune.py raised AttributeError at line 601. The
    training path could not run at all unless W&B was configured, which is why nothing in
    this repository had ever executed it.
    """

    def test_omitting_report_to_leaves_tracking_off_rather_than_unset(self) -> None:
        args = _postprocess(argparse.Namespace(report_to=None))
        self.assertFalse(args.with_tracking)

    def test_the_attribute_always_exists(self) -> None:
        args = _postprocess(argparse.Namespace(report_to=None))
        self.assertTrue(hasattr(args, "with_tracking"))

    def test_asking_for_wandb_turns_tracking_on(self) -> None:
        args = _postprocess(argparse.Namespace(report_to="wandb"))
        self.assertTrue(args.with_tracking)


class SourceStillMatchesTests(unittest.TestCase):
    def test_finetune_py_sets_with_tracking_unconditionally(self) -> None:
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "finetune.py").read_text(encoding="utf-8")
        self.assertIn("args.with_tracking = args.report_to is not None", source)
        # The old form assigned inside a branch; if it comes back, so does the crash.
        self.assertNotIn("        args.with_tracking = True", source)


if __name__ == "__main__":
    unittest.main()
