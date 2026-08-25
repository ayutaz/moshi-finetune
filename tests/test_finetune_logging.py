"""What finetune.py prints to stdout, checked without importing finetune.py.

finetune.py imports torch, deepspeed, accelerate and datasets at module scope and none of
those are installed for the test suite, so the module cannot be imported here. The
functions under test are therefore lifted out of the real source with `ast` and executed in
an empty namespace. That is deliberately not the approach of `test_finetune_args.py`, which
copies the line under test into itself and can silently drift; loading the shipped
definitions fails loudly when they move, and what runs in the test is what runs on the GPU.

The behaviour being pinned down is M3's central loss of evidence. Every per-component loss
was computed each step and reported only through W&B, M3 ran without `--report_to`, and so
two arms times five epochs of `loss/audio_semantic` and `loss/audio_semantic_user` - the
speaker-A/speaker-B split the whole voice question turns on - went to nowhere and cannot be
recovered: the dep_q=16 weights are gone once the checkpoints are converted for inference.
"""

import ast
import math
import unittest
from pathlib import Path

FINETUNE_PY = Path(__file__).resolve().parents[1] / "finetune.py"

# The four the M3-R plan (step 1-6) names. audio_semantic_user exists only under
# --model_user_stream, which is why format_loss_breakdown skips absent keys instead of
# printing a placeholder.
REQUIRED_BREAKDOWN = (
    "loss/text_non_pad",
    "loss/text_pad",
    "loss/audio_semantic",
    "loss/audio_semantic_user",
)


def load_definitions(path: Path, names, extra_globals=None) -> dict:
    """Execute only the named top-level definitions of `path` in a fresh namespace.

    Nothing else in the file runs, so the heavy imports at the top never execute. Every
    name a loaded definition touches must therefore be a builtin, another loaded name, or
    supplied through `extra_globals`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name in wanted:
                selected.append(node)
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & wanted:
                selected.append(node)
                found |= targets & wanted
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in wanted:
                selected.append(node)
                found.add(node.target.id)
    missing = wanted - found
    if missing:
        raise AssertionError(f"{path.name} no longer defines {sorted(missing)} at module level")
    namespace = dict(extra_globals or {})
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, filename=str(path), mode="exec"), namespace)  # noqa: S102
    return namespace


SOURCE = FINETUNE_PY.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(FINETUNE_PY))
NS = load_definitions(
    FINETUNE_PY,
    [
        "LOSS_BREAKDOWN_KEYS",
        "format_loss_breakdown",
        "format_training_log_line",
        "format_evaluation_log_line",
        "reduce_partial_means",
    ],
)
format_loss_breakdown = NS["format_loss_breakdown"]
format_training_log_line = NS["format_training_log_line"]
format_evaluation_log_line = NS["format_evaluation_log_line"]
reduce_partial_means = NS["reduce_partial_means"]


def two_stream_means() -> dict[str, float]:
    """A logging window from a --model_user_stream run, already reduced."""
    return {
        "training_loss/total": 7.03000,
        "training_loss/text_total": 1.50000,
        "training_loss/audio_total": 5.53000,
        "training_loss/text_non_pad": 3.25000,
        "training_loss/text_pad": 0.12500,
        "training_loss/audio_semantic": 4.00000,
        "training_loss/audio_acoustic": 6.25000,
        "training_loss/audio_semantic_user": 2.50000,
        "training_loss/audio_acoustic_user": 5.75000,
        "training_accuracy/text_non_pad": 0.42000,
    }


class LossBreakdownTests(unittest.TestCase):
    def test_every_required_component_is_named(self) -> None:
        keys = NS["LOSS_BREAKDOWN_KEYS"]
        self.assertIsInstance(keys, tuple)
        for key in REQUIRED_BREAKDOWN:
            self.assertIn(key, keys, key)

    def test_two_stream_window_prints_all_four_required_components(self) -> None:
        line = format_loss_breakdown(two_stream_means())
        for key in REQUIRED_BREAKDOWN:
            self.assertIn(f"{key}=", line, key)

    def test_values_are_the_ones_given(self) -> None:
        line = format_loss_breakdown(two_stream_means())
        self.assertIn("loss/audio_semantic=4.00000", line)
        self.assertIn("loss/audio_semantic_user=2.50000", line)

    def test_order_is_fixed_rather_than_dict_order(self) -> None:
        means = two_stream_means()
        shuffled = dict(reversed(list(means.items())))
        self.assertEqual(format_loss_breakdown(means), format_loss_breakdown(shuffled))

    def test_single_stream_run_omits_the_user_components_rather_than_faking_them(self) -> None:
        means = {k: v for k, v in two_stream_means().items() if not k.endswith("_user")}
        line = format_loss_breakdown(means)
        self.assertNotIn("_user", line)
        self.assertIn("loss/audio_semantic=", line)

    def test_metrics_outside_the_breakdown_are_not_printed(self) -> None:
        self.assertNotIn("accuracy", format_loss_breakdown(two_stream_means()))

    def test_a_nan_component_is_shown_rather_than_dropped(self) -> None:
        means = two_stream_means()
        means["training_loss/text_pad"] = float("nan")
        self.assertIn("loss/text_pad=nan", format_loss_breakdown(means))

    def test_an_empty_window_yields_an_empty_string(self) -> None:
        self.assertEqual(format_loss_breakdown({}), "")


class TrainingLogLineTests(unittest.TestCase):
    LRS = {"tempformer": "2.000e-06", "depformer": "4.000e-06"}

    def line(self, means=None) -> str:
        return format_training_log_line(
            epoch=2, steps=45, lrs=self.LRS, means=two_stream_means() if means is None else means
        )

    def test_it_reports_the_number_it_was_given(self) -> None:
        """The old line printed one micro-batch of eight, so identical scripts disagreed."""
        self.assertIn("Loss: 7.03000", self.line())
        self.assertIn("text: 1.50000", self.line())
        self.assertIn("audio: 5.53000", self.line())

    def test_it_carries_the_epoch_step_and_learning_rates(self) -> None:
        line = self.line()
        self.assertIn("Epoch: 2", line)
        self.assertIn("Steps: 45", line)
        self.assertIn("2.000e-06", line)

    def test_the_learning_rate_column_says_it_is_the_next_step_rate(self) -> None:
        """DeepSpeed steps the scheduler before the caller reads param_groups, so the rate
        beside step N is the one step N+1 uses. M3's log did not say so, and three
        measurements of the untrained base were read as three training steps."""
        self.assertIn("LRs(next step):", self.line())

    def test_the_breakdown_is_appended(self) -> None:
        line = self.line()
        self.assertIn(" | ", line)
        for key in REQUIRED_BREAKDOWN:
            self.assertIn(f"{key}=", line, key)

    def test_a_missing_total_prints_nan_instead_of_raising(self) -> None:
        """The line it replaced indexed `log['loss/text_total']` and would KeyError."""
        means = {k: v for k, v in two_stream_means().items() if k != "training_loss/text_total"}
        self.assertIn("text: nan", self.line(means))

    def test_a_window_with_nothing_in_it_still_produces_a_line(self) -> None:
        line = self.line({})
        self.assertIn("Steps: 45", line)
        self.assertIn("Loss: nan", line)
        self.assertNotIn(" | ", line)


class EvaluationLogLineTests(unittest.TestCase):
    MEANS = {
        "evaluation_loss/total": 6.10000,
        "evaluation_loss/audio_semantic": 4.00000,
        "evaluation_loss/audio_semantic_user": 2.50000,
    }

    def test_it_starts_with_the_string_the_logs_were_grepped_for(self) -> None:
        """Grepping both M3 nohup logs for this prefix returned zero lines."""
        line = format_evaluation_log_line(45, self.MEANS)
        self.assertTrue(line.startswith("Evaluation at step 45: "), line)

    def test_the_user_side_audio_loss_is_present(self) -> None:
        self.assertIn(
            "evaluation_loss/audio_semantic_user=2.50000",
            format_evaluation_log_line(45, self.MEANS),
        )

    def test_keys_are_sorted_so_two_runs_can_be_diffed(self) -> None:
        forwards = format_evaluation_log_line(45, self.MEANS)
        backwards = format_evaluation_log_line(45, dict(reversed(list(self.MEANS.items()))))
        self.assertEqual(forwards, backwards)

    def test_no_metrics_still_names_the_step(self) -> None:
        self.assertEqual(format_evaluation_log_line(45, {}), "Evaluation at step 45: ")


class ReducePartialMeansTests(unittest.TestCase):
    def test_it_pools_observations_across_processes(self) -> None:
        self.assertAlmostEqual(reduce_partial_means([4.0, 2.0], [2.0, 1.0]), 2.0)

    def test_it_weights_by_count_rather_than_averaging_per_process_means(self) -> None:
        """Process 0 saw three values averaging 1.0, process 1 saw one worth 10.0."""
        self.assertAlmostEqual(reduce_partial_means([3.0, 10.0], [3.0, 1.0]), 3.25)
        self.assertNotAlmostEqual(reduce_partial_means([3.0, 10.0], [3.0, 1.0]), 5.5)

    def test_a_process_with_no_valid_observation_does_not_drag_the_mean(self) -> None:
        self.assertAlmostEqual(reduce_partial_means([6.0, 0.0], [3.0, 0.0]), 2.0)

    def test_nothing_observed_anywhere_is_nan_not_a_crash(self) -> None:
        """`loss/text_pad` on a window whose batches carried no padding."""
        self.assertTrue(math.isnan(reduce_partial_means([0.0, 0.0], [0.0, 0.0])))

    def test_a_single_process_run_is_just_the_mean(self) -> None:
        self.assertAlmostEqual(reduce_partial_means([7.5], [3.0]), 2.5)

    def test_mismatched_lengths_raise_rather_than_silently_truncating(self) -> None:
        """Sums and counts come from one gather of matched keys; a mismatch is a bug."""
        with self.assertRaises(ValueError):
            reduce_partial_means([1.0, 2.0], [1.0])


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def tracking_guarded_nodes(tree: ast.AST) -> set[ast.AST]:
    """Every node that runs only when `--with_tracking` is on."""
    guarded: set[ast.AST] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            child.attr for child in ast.walk(node.test) if isinstance(child, ast.Attribute)
        } | {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        if "with_tracking" not in names:
            continue
        for statement in node.body:
            guarded |= set(ast.walk(statement))
    return guarded


class ReportingIsUnconditionalTests(unittest.TestCase):
    """The regression that cost M3 ten epochs of loss breakdown, pinned structurally.

    Nothing a pure formatting test can say proves the formatter is reached without W&B, so
    this reads finetune.py itself and checks that the reporting calls are not sitting inside
    an `if args.with_tracking:` block.
    """

    def setUp(self) -> None:
        self.guarded = tracking_guarded_nodes(TREE)

    def assert_unguarded(self, name: str) -> None:
        found = calls_named(TREE, name)
        self.assertTrue(found, f"finetune.py never calls {name}")
        for call in found:
            self.assertNotIn(
                call,
                self.guarded,
                f"{name} at line {call.lineno} only runs under --with_tracking",
            )

    def test_the_training_line_is_printed_without_tracking(self) -> None:
        self.assert_unguarded("format_training_log_line")

    def test_the_evaluation_line_is_printed_without_tracking(self) -> None:
        self.assert_unguarded("format_evaluation_log_line")

    def test_the_cross_process_reduction_runs_without_tracking(self) -> None:
        self.assert_unguarded("gather_metric_means")

    def test_the_reduction_happens_before_the_line_is_built(self) -> None:
        """`logger` writes on the main process only; a gather inside it hangs the others."""
        gathers = [call.lineno for call in calls_named(TREE, "gather_metric_means")]
        for name in ("format_training_log_line", "format_evaluation_log_line"):
            for call in calls_named(TREE, name):
                self.assertTrue(
                    any(lineno < call.lineno for lineno in gathers),
                    f"{name} at line {call.lineno} has no reduction before it",
                )

    def test_the_per_microbatch_loss_is_no_longer_what_gets_printed(self) -> None:
        self.assertNotIn("Loss: {total_loss.item():.5f}", SOURCE)

    def test_the_raw_buffer_is_no_longer_gathered_whole(self) -> None:
        """Gathering per-micro-batch lists deadlocks when two processes buffer different
        numbers of them; only a (sum, count) pair per key should cross the wire."""
        self.assertNotIn("torch.tensor(values, device=accelerator.device)", SOURCE)


if __name__ == "__main__":
    unittest.main()
