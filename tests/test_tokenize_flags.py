import ast
import json
import unittest
from pathlib import Path

from tools.tokenize_flags import (
    PIPELINE_ORDER,
    PREPARE_DATASET,
    TOKENIZE_AUDIO,
    TOKENIZE_TEXT,
    TOOLS,
    UnknownToolError,
    check_invocation,
    check_record,
    invocation_tool,
    known_defects,
    render_argv,
    render_command,
    resolve_invocation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPOSITORY_ROOT / "experiments" / "tsukuyomi_ojousama" / "manifests"

# Sidecars written before the record covered three tools. Each entry says why, and a
# dataset that is not on this list has to record all three. The list is meant to shrink:
# `v-real-v2` comes off it as soon as its sidecar is rewritten with the audio and parquet
# invocations, which is the last thing standing between `--device cpu` and being reachable
# from the manifest rather than from a report.
PRE_MECHANISM_SIDECARS = {
    "v-real-v1-tokenize.json": (
        "M3 recorded no command line anywhere. This sidecar is a reconstruction of the "
        "tokenize_text run alone; M3's audio and parquet flags were never written down and "
        "cannot be recovered. Superseded by v-real-v2."
    ),
    "v-tts-v1-tokenize.json": (
        "Same M3 run as v-real-v1, same reconstruction, same missing audio and parquet "
        "flags. The V-tts arm is out of M3-R."
    ),
    "v-real-v2-tokenize.json": (
        "Recorded on 2026-08-26, before tools/tokenize_flags.py existed, so only the "
        "tokenize_text invocations are in it. The audio and parquet flags - --device cpu "
        "among them - are in experiments/tsukuyomi_ojousama/reports/m3r-tokenize.json. "
        "Rerun the three record-tokenize commands in m3r/TOKENIZE_COMMANDS.md section 4 and "
        "remove this entry."
    ),
}


def add_argument_calls(source_path: Path) -> list[ast.Call]:
    """Every `parser.add_argument("--name", ...)` call in a tool's source."""
    source = source_path.read_text(encoding="utf-8")
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(getattr(node.args[0], "value", None), str)
    ]


class FlagTablesMatchTheirSourceTests(unittest.TestCase):
    """Each table is a copy of a parser that cannot be imported.

    All three tools build their `argparse` block inside `if __name__ == "__main__"`, so the
    flags have to be duplicated here. A flag added to one of them and forgotten here would
    be missing from every future record while the record still claimed to be complete -
    which is the exact failure the record exists to prevent. This is the only thing that
    keeps the copies honest.
    """

    def test_every_tool_declares_the_same_flags_as_its_source(self) -> None:
        for name, tool in sorted(TOOLS.items()):
            with self.subTest(tool=name):
                declared = {
                    call.args[0].value.removeprefix("--")
                    for call in add_argument_calls(REPOSITORY_ROOT / tool.source)
                }
                self.assertEqual(declared, set(tool.defaults))

    def test_every_recorded_default_matches_its_source(self) -> None:
        for name, tool in sorted(TOOLS.items()):
            for call in add_argument_calls(REPOSITORY_ROOT / tool.source):
                flag = call.args[0].value.removeprefix("--")
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                if "default" not in keywords:
                    continue
                with self.subTest(tool=name, flag=flag):
                    self.assertEqual(keywords["default"].value, tool.defaults[flag])

    def test_the_required_and_store_true_flags_match_their_source(self) -> None:
        """`resolve_invocation` rebuilds each parser from these two sets.

        A flag this table calls `required` is given `type=str` (its entry in `defaults` is
        None, so the type of the default cannot stand in for the parser's type), and one it
        calls `store_true` is given no value at all. Get either wrong and the record either
        refuses a command line the tool accepts, or accepts one it does not.
        """
        for name, tool in sorted(TOOLS.items()):
            calls = add_argument_calls(REPOSITORY_ROOT / tool.source)
            required = {
                call.args[0].value.removeprefix("--")
                for call in calls
                if any(
                    keyword.arg == "required" and getattr(keyword.value, "value", None) is True
                    for keyword in call.keywords
                )
            }
            store_true = {
                call.args[0].value.removeprefix("--")
                for call in calls
                if any(
                    keyword.arg == "action"
                    and getattr(keyword.value, "value", None) == "store_true"
                    for keyword in call.keywords
                )
            }
            with self.subTest(tool=name):
                self.assertEqual(required, set(tool.required))
                self.assertEqual(store_true, set(tool.store_true))

    def test_tokenize_audio_still_defaults_to_cuda(self) -> None:
        """The reason `--device` must be recorded rather than inferred.

        A local run that does not name a device does not quietly fall back to cpu, it
        raises; and a run that names mps produces a different parquet from the same wav.
        Either way the value is unrecoverable from the artifacts, so it has to be written
        down. If this default ever changes, the sentence above stops being true.
        """
        self.assertEqual(TOKENIZE_AUDIO.defaults["device"], "cuda")


class InvocationRecordTests(unittest.TestCase):
    def test_an_omitted_store_true_flag_is_recorded_as_false(self) -> None:
        record = resolve_invocation(
            ["--audio_dir", "in", "--output_dir", "out"], tool="tokenize_audio"
        )

        self.assertIs(record["flags"]["resume"], False)
        self.assertIn("resume", record["defaults_used"])

    def test_the_device_is_recorded_even_when_it_is_the_default(self) -> None:
        record = resolve_invocation(
            ["--audio_dir", "in", "--output_dir", "out"], tool="tokenize_audio"
        )

        self.assertEqual(record["flags"]["device"], "cuda")
        self.assertIn("device", record["defaults_used"])

    def test_the_record_names_its_tool(self) -> None:
        record = resolve_invocation(
            ["--tokenized_text_dir", "t", "--tokenized_audio_dir", "a", "--output_prefix", "p"],
            tool="prepare_dataset",
        )

        self.assertEqual(record["tool"], "prepare_dataset")
        self.assertEqual(set(record["flags"]), set(PREPARE_DATASET.defaults))

    def test_an_unknown_tool_raises(self) -> None:
        with self.assertRaises(UnknownToolError):
            resolve_invocation([], tool="tokenize_video")

    def test_a_record_without_a_tool_is_read_as_tokenize_text(self) -> None:
        """The three sidecars written before `--tool` existed describe tokenize_text alone.

        Reading them as anything else would attribute flags to a tool that never ran.
        """
        self.assertEqual(invocation_tool({"flags": {}}), TOKENIZE_TEXT.name)


class CommandRenderingTests(unittest.TestCase):
    """The procedure document and the report render from the record, not from memory."""

    def _audio_flags(self, **overrides):
        return {**TOKENIZE_AUDIO.defaults, "audio_dir": "in", "output_dir": "out", **overrides}

    def test_the_rendered_argv_round_trips(self) -> None:
        flags = self._audio_flags(device="cpu")

        resolved = resolve_invocation(
            render_argv(flags, tool="tokenize_audio"), tool="tokenize_audio"
        )

        self.assertEqual(resolved["flags"], flags)

    def test_a_false_store_true_flag_renders_as_nothing(self) -> None:
        argv = render_argv(self._audio_flags(device="cpu"), tool="tokenize_audio")

        self.assertNotIn("--resume", argv)

    def test_a_true_store_true_flag_renders_as_the_bare_flag(self) -> None:
        argv = render_argv(self._audio_flags(device="cpu", resume=True), tool="tokenize_audio")

        self.assertIn("--resume", argv)
        self.assertNotIn("True", argv)

    def test_the_command_names_the_module(self) -> None:
        command = render_command(
            self._audio_flags(device="cpu"), tool="tokenize_audio", prefix=["nice", "-n", "19"]
        )

        self.assertTrue(command.startswith("nice -n 19 python -m tools.tokenize_audio "))
        self.assertIn("--device cpu", command)

    def test_a_missing_value_refuses_to_render(self) -> None:
        with self.assertRaises(ValueError):
            render_argv({**TOKENIZE_AUDIO.defaults, "output_dir": "out"}, tool="tokenize_audio")


def audio(split="train", **overrides):
    flags = {
        **TOKENIZE_AUDIO.defaults,
        "audio_dir": f"d/{split}/audio",
        "output_dir": f"d/{split}/tok-audio",
        "device": "cpu",
        **overrides,
    }
    return {
        "split": split,
        **resolve_invocation(render_argv(flags, tool="tokenize_audio"), tool="tokenize_audio"),
    }


def text(split="train", **overrides):
    flags = {
        **TOKENIZE_TEXT.defaults,
        "word_transcript_dir": f"d/{split}/text",
        "output_dir": f"d/{split}/tok-text",
        "no_whitespace_before_word": True,
        **overrides,
    }
    return {
        "split": split,
        **resolve_invocation(render_argv(flags, tool="tokenize_text"), tool="tokenize_text"),
    }


def parquet(split="train", **overrides):
    flags = {
        **PREPARE_DATASET.defaults,
        "tokenized_text_dir": f"d/{split}/tok-text",
        "tokenized_audio_dir": f"d/{split}/tok-audio",
        "output_prefix": f"d/parquet/{split}",
        **overrides,
    }
    return {
        "split": split,
        **resolve_invocation(render_argv(flags, tool="prepare_dataset"), tool="prepare_dataset"),
    }


def record(*invocations, **extra):
    return {"dataset_id": "d-v1", "invocations": list(invocations), **extra}


class RecordCompletenessTests(unittest.TestCase):
    """The negative controls. A gate never seen to fail is a claim, not a measurement."""

    def test_a_complete_record_has_no_problems(self) -> None:
        self.assertEqual(check_record(record(audio(), text(), parquet())), [])

    def test_a_missing_tool_is_reported_when_required(self) -> None:
        problems = check_record(
            record(text()), require_tools=["tokenize_audio", "tokenize_text", "prepare_dataset"]
        )

        self.assertTrue(any("no tokenize_audio invocation" in problem for problem in problems))
        self.assertTrue(any("no prepare_dataset invocation" in problem for problem in problems))

    def test_a_missing_tool_is_not_reported_when_it_is_not_required(self) -> None:
        self.assertEqual(check_record(record(text())), [])

    def test_an_incomplete_flag_set_is_reported(self) -> None:
        broken = audio()
        broken["flags"].pop("device")

        problems = check_invocation(broken)

        self.assertTrue(any("device not stated" in problem for problem in problems))

    def test_tokenizing_on_mps_without_a_written_defect_fails(self) -> None:
        """Measured: 2 of 21,600 tokens differ between mps and cpu on the same wav files."""
        problems = check_record(record(audio(device="mps"), text(), parquet()))

        self.assertTrue(any("not bit-identical to cpu" in problem for problem in problems))

    def test_tokenizing_on_mps_with_a_written_defect_passes(self) -> None:
        problems = check_record(
            record(
                audio(device="mps"),
                text(),
                parquet(),
                known_defect={"flag": "device", "expected": "cpu", "actual": "mps"},
            )
        )

        self.assertEqual(problems, [])

    def test_tokenizing_on_cpu_needs_no_declaration(self) -> None:
        self.assertEqual(check_record(record(audio(device="cpu"))), [])

    def test_disagreeing_text_padding_ids_are_reported(self) -> None:
        problems = check_record(record(text(), parquet(text_padding_id=0)))

        self.assertTrue(any("spell silence" in problem for problem in problems))

    def test_a_parquet_built_from_another_run_is_reported(self) -> None:
        """The M3 failure one level up: an artifact consistent with itself, not its inputs."""
        problems = check_record(record(text(), parquet(tokenized_text_dir="d/old/tok-text")))

        self.assertTrue(any("is not where tokenize_text wrote" in problem for problem in problems))

    def test_a_parquet_reading_the_wrong_audio_directory_is_reported(self) -> None:
        problems = check_record(record(audio(), parquet(tokenized_audio_dir="d/old/tok-audio")))

        self.assertTrue(any("is not where tokenize_audio wrote" in problem for problem in problems))

    def test_a_trailing_slash_is_not_a_disagreement(self) -> None:
        self.assertEqual(check_record(record(text(output_dir="d/train/tok-text/"), parquet())), [])

    def test_an_output_prefix_that_renames_the_namespace_is_reported(self) -> None:
        problems = check_record(record(parquet(output_prefix="d/parquet/v-real-train")))

        self.assertTrue(any("dialogue_id namespace" in problem for problem in problems))

    def test_the_same_tool_recorded_twice_for_one_split_is_reported(self) -> None:
        problems = check_record(record(text(), text()))

        self.assertTrue(any("recorded twice" in problem for problem in problems))

    def test_a_record_with_no_invocations_is_reported(self) -> None:
        self.assertEqual(check_record({"invocations": []}), ["no invocations recorded"])

    def test_each_split_is_checked_on_its_own(self) -> None:
        problems = check_record(
            record(
                audio("train"),
                parquet("train"),
                audio("dev"),
                parquet("dev", tokenized_audio_dir="d/train/tok-audio"),
            )
        )

        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("dev:"))


class KnownDefectShapeTests(unittest.TestCase):
    """`v-real-v1-tokenize.json` writes one dict; a record with two defects writes a list."""

    def test_a_single_dict_is_read(self) -> None:
        self.assertEqual(len(known_defects({"known_defect": {"flag": "device"}})), 1)

    def test_a_list_is_read(self) -> None:
        defects = known_defects(
            {"known_defect": [{"flag": "device"}, {"flag": "no_whitespace_before_word"}]}
        )

        self.assertEqual(
            [defect["flag"] for defect in defects], ["device", "no_whitespace_before_word"]
        )

    def test_no_defect_is_an_empty_list(self) -> None:
        self.assertEqual(known_defects({}), [])


class ShippedSidecarTests(unittest.TestCase):
    """The same checks, run against the sidecars that are actually in the repository.

    The class above proves the gate fires on a record built to fail it. This one is the
    gate: it reads the files a training run would be launched from. Standard library only,
    because CI installs nothing but pytest.
    """

    def _sidecars(self) -> list[tuple[Path, dict]]:
        paths = sorted(MANIFEST_DIR.glob("*-tokenize.json"))
        if not paths:
            self.skipTest("no tokenize sidecars in this checkout")
        return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]

    def test_every_shipped_sidecar_is_internally_consistent(self) -> None:
        for path, record in self._sidecars():
            with self.subTest(sidecar=path.name):
                self.assertEqual(check_record(record), [], path.name)

    def test_a_dataset_that_is_not_grandfathered_records_all_three_tools(self) -> None:
        """A new dataset has to make every flag reachable from its manifest.

        The three tools are recorded per split, so this asks for each split of each sidecar
        rather than for the sidecar as a whole - a dataset whose dev split was tokenized on
        a different device would otherwise pass on the strength of its train split.
        """
        for path, record in self._sidecars():
            splits = {str(invocation.get("split", "")) for invocation in record["invocations"]}
            problems = check_record(record, require_tools=PIPELINE_ORDER)
            if not problems:
                self.assertNotIn(
                    path.name,
                    PRE_MECHANISM_SIDECARS,
                    f"{path.name} now records all three tools for {sorted(splits)}; take it "
                    "out of PRE_MECHANISM_SIDECARS",
                )
                continue
            self.assertIn(
                path.name,
                PRE_MECHANISM_SIDECARS,
                f"{path.name} does not record every tool and is not grandfathered: "
                + "; ".join(problems),
            )
            self.assertTrue(PRE_MECHANISM_SIDECARS[path.name].strip(), path.name)


class AssetGateCompatibilityTests(unittest.TestCase):
    """A trip-wire on the older gate, which still assumes one tool per sidecar.

    `tests/test_experiment_assets.py::TokenizeFlagRecordTests` asserts that every invocation
    of every sidecar carries exactly the `tokenize_text` flag set. That was true while the
    record covered one tool. The moment an audio or parquet invocation is written, it stops
    being true, and the suite fails on a sidecar that is more complete than before - the
    worst kind of red, because the obvious fix is to undo the improvement.

    Generalising it is small: dispatch on `tokenize_flags.invocation_tool` and compare
    against that tool's table, or call `check_invocation` and assert it returns nothing.
    This test says so at the moment it starts to matter rather than after a surprising CI
    run, and it costs nothing until then.
    """

    ASSET_TESTS = REPOSITORY_ROOT / "tests" / "test_experiment_assets.py"

    def test_the_older_gate_is_generalised_before_a_second_tool_is_recorded(self) -> None:
        if not self.ASSET_TESTS.is_file():
            self.skipTest("tests/test_experiment_assets.py is not in this checkout")
        if not MANIFEST_DIR.is_dir():
            self.skipTest("no manifests directory in this checkout")

        multi_tool = []
        for path in sorted(MANIFEST_DIR.glob("*-tokenize.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            tools = {invocation_tool(item) for item in record.get("invocations", [])}
            if tools - {TOKENIZE_TEXT.name}:
                multi_tool.append(f"{path.name} ({', '.join(sorted(tools))})")
        if not multi_tool:
            return

        source = self.ASSET_TESTS.read_text(encoding="utf-8")
        generalised = "check_invocation" in source or "invocation_tool" in source
        self.assertTrue(
            generalised,
            "these sidecars record more than one tool: "
            + "; ".join(multi_tool)
            + ". tests/test_experiment_assets.py::TokenizeFlagRecordTests::"
            "test_every_invocation_states_every_flag_explicitly still compares every "
            "invocation against TOKENIZE_TEXT_FLAGS, so it will fail on them. Point it at "
            "tools.tokenize_flags.check_invocation instead.",
        )


if __name__ == "__main__":
    unittest.main()
