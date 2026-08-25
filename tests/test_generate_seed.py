"""generate.py must refuse to run without a seed, and must record the one it used.

M3's generations passed no `--seed`, so `set_seed` was never called: the collapse counts of
condition 3, the win counts of condition 4 and the voiced-frame counts of condition 5 each
rest on a single draw that cannot be drawn again, and nothing in the output said so. The
failure was silence, so a default seed would reproduce it exactly - the argument is
required instead.

generate.py imports torch, accelerate and datasets at module scope and none of those are
installed for the test suite, so the definitions under test are lifted out of the real
source with `ast` and executed in an empty namespace. See tests/test_finetune_logging.py
for the loader; it is imported from there rather than duplicated.
"""

import argparse
import ast
import contextlib
import io
import unittest
from pathlib import Path

from test_finetune_logging import load_definitions

GENERATE_PY = Path(__file__).resolve().parents[1] / "generate.py"

SOURCE = GENERATE_PY.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(GENERATE_PY))
NS = load_definitions(
    GENERATE_PY,
    ["setup_argparser", "build_run_config"],
    extra_globals={"argparse": argparse},
)
setup_argparser = NS["setup_argparser"]
build_run_config = NS["build_run_config"]

# Everything generate.py already required before this change, so that a parse failure in
# these tests can only be about the seed.
MINIMUM_ARGS = [
    "--output_dir",
    "out",
    "--eval_data_files",
    "prompts-*.parquet",
    "--model_dir",
    "checkpoint",
]


def parse(argv: list[str]) -> argparse.Namespace:
    parser = setup_argparser(argparse.ArgumentParser())
    return parser.parse_args(argv)


def parse_expecting_failure(argv: list[str]) -> str:
    """argparse exits and prints usage; the message is returned rather than spilled."""
    parser = setup_argparser(argparse.ArgumentParser(prog="generate.py"))
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.suppress(SystemExit):
        parser.parse_args(argv)
        raise AssertionError(f"parsing {argv} was expected to fail but did not")
    return stderr.getvalue()


class SeedIsRequiredTests(unittest.TestCase):
    def test_a_command_line_without_a_seed_does_not_parse(self) -> None:
        message = parse_expecting_failure(MINIMUM_ARGS)
        self.assertIn("--seed", message)

    def test_the_same_command_line_with_a_seed_parses(self) -> None:
        args = parse([*MINIMUM_ARGS, "--seed", "20260824"])
        self.assertEqual(args.seed, 20260824)

    def test_the_seed_is_an_int_not_a_string(self) -> None:
        self.assertIsInstance(parse([*MINIMUM_ARGS, "--seed", "7"]).seed, int)

    def test_seed_zero_is_a_seed(self) -> None:
        """`if args.seed:` would treat 0 as unset; the old code used `is not None`."""
        self.assertEqual(parse([*MINIMUM_ARGS, "--seed", "0"]).seed, 0)

    def test_the_other_required_arguments_are_still_required(self) -> None:
        self.assertIn(
            "--model_dir",
            parse_expecting_failure(
                ["--output_dir", "out", "--eval_data_files", "p.parquet", "--seed", "1"]
            ),
        )


class RunConfigTests(unittest.TestCase):
    def test_the_seed_reaches_the_config_that_gets_written(self) -> None:
        config = build_run_config(parse([*MINIMUM_ARGS, "--seed", "20260824"]))
        self.assertEqual(config["seed"], 20260824)

    def test_the_sampling_settings_are_recorded_alongside_it(self) -> None:
        """A seed reproduces nothing unless the sampler it seeded is recorded too."""
        config = build_run_config(parse([*MINIMUM_ARGS, "--seed", "1", "--temperature", "0.8"]))
        for key in ("temperature", "top_k", "top_p", "use_sampling", "prompt_length"):
            self.assertIn(key, config)

    def test_the_config_is_a_plain_dict_that_json_can_take(self) -> None:
        import json

        json.dumps(build_run_config(parse([*MINIMUM_ARGS, "--seed", "1"])))

    def test_editing_the_config_does_not_reach_back_into_the_args(self) -> None:
        args = parse([*MINIMUM_ARGS, "--seed", "1"])
        build_run_config(args)["seed"] = 999
        self.assertEqual(args.seed, 1)

    def test_a_caller_that_bypasses_argparse_is_still_refused(self) -> None:
        args = parse([*MINIMUM_ARGS, "--seed", "1"])
        args.seed = None
        with self.assertRaises(ValueError) as caught:
            build_run_config(args)
        self.assertIn("seed", str(caught.exception))


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


class SourceStillWiresItUpTests(unittest.TestCase):
    """The parser and the config helper are only worth anything if generate.py uses them."""

    def test_seeding_is_no_longer_conditional(self) -> None:
        self.assertNotIn("if args.seed is not None:", SOURCE)
        self.assertTrue(calls_named(TREE, "set_seed"), "generate.py never calls set_seed")

    def test_set_seed_is_not_nested_in_a_branch(self) -> None:
        guarded = set()
        for node in ast.walk(TREE):
            if isinstance(node, ast.If):
                for statement in node.body + node.orelse:
                    guarded |= set(ast.walk(statement))
        for call in calls_named(TREE, "set_seed"):
            self.assertNotIn(call, guarded, f"set_seed at line {call.lineno} is conditional")

    def test_the_written_config_goes_through_the_guarded_builder(self) -> None:
        self.assertTrue(calls_named(TREE, "build_run_config"))
        self.assertNotIn("json.dump(vars(args), f", SOURCE)

    def test_parse_args_uses_the_parser_the_tests_build(self) -> None:
        self.assertTrue(calls_named(TREE, "setup_argparser"))


if __name__ == "__main__":
    unittest.main()
