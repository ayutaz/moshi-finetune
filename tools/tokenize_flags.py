"""Every flag of the three tools that turn a dialogue directory into a training parquet.

Why this module exists
----------------------
`tools/text_stream_audit.py` learned to record the argv of `tools/tokenize_text.py` after
M3 shipped a dataset tokenized without `--no_whitespace_before_word` and nobody could say
so afterwards. It records that one tool. The pipeline has three, and the two it does not
record hold a flag that is worse to lose:

`tools/tokenize_audio.py --device`. `experiments/tsukuyomi_ojousama/m3r/TOKENIZE_COMMANDS.md`
section 1 measured mps against cpu over the same wav files: 2 of 21,600 tokens differ, in
residual codebooks 4 and 6. Both parquets are well formed, both train, and the checksums
simply stop matching the ones in the manifest. Nothing downstream says a word. That is the
same defect as M3's, one tool over, and until this module the flags for those two tools
lived only in a report - `reports/m3r-tokenize.json` - which is not what a manifest points
at.

So the flag tables for all three tools live here. Each is a copy of a parser that cannot be
imported: all three build their `argparse` block inside `if __name__ == "__main__"`.
`tests/test_tokenize_flags.py` reads the three source files with `ast` and fails when a copy
goes stale, which is the only thing that keeps a copy honest.

The CLI stayed where it was
---------------------------
`python -m tools.text_stream_audit record-tokenize` is the command the procedure documents
and the command that is being run right now, so it keeps its name and its behaviour. It
grew one optional `--tool` argument. An invocation with no `tool` key is read as
`tokenize_text`, because that is the only tool the recorder could describe when the existing
sidecars were written.

What is checked, and the failure behind each check
--------------------------------------------------
Every rule here is a thing that has gone wrong, in this repository or one step away from it.

* **Every flag of the tool, stated.** A `store_true` flag that was dropped and a run that
  was never recorded look identical in a shell history. This is M3's defect exactly.
* **`--device` is not mps without a written defect.** Measured above. `cpu` is the
  reproducible one; `cuda` is a different machine, and the record is what tells the two
  apart later.
* **`prepare_dataset --text_padding_id` equals `tokenize_text --text_padding_id`.**
  `prepare_dataset` pads the text stream out to the audio length with this id. Two
  different ids mean two different tokens spelling "silence" in one stream, and the parquet
  is well formed either way.
* **`prepare_dataset` reads the directories the other two wrote.** A parquet built from a
  previous tokenize run passes every count and every row check. This is the M3 failure one
  level up: the artifact is consistent with itself and not with its inputs.
* **The basename of `--output_prefix` is the split.** It becomes the `dialogue_id`
  namespace (`train/v-001`), which is what joins the parquet to the manifest.

Nothing here imports anything heavy; the checks are the layer that has to run in CI, where
only pytest is installed.
"""

from __future__ import annotations

import argparse
import posixpath
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolFlags:
    """One tool's argparse block, copied because it cannot be imported."""

    name: str
    module: str
    source: str
    defaults: dict[str, Any]
    required: frozenset[str]
    store_true: frozenset[str]

    def __post_init__(self) -> None:
        unknown = (self.required | self.store_true) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: {sorted(unknown)} are not flags of this tool")


# `tools/tokenize_text.py`, in declaration order. The two required flags have no default,
# so their entry is None and the type of the default cannot stand in for the parser's type
# the way it does for the rest.
TOKENIZE_TEXT = ToolFlags(
    name="tokenize_text",
    module="tools.tokenize_text",
    source="tools/tokenize_text.py",
    defaults={
        "word_transcript_dir": None,
        "output_dir": None,
        "text_tokenizer_repo": "kyutai/moshiko-pytorch-bf16",
        "text_tokenizer_name": "tokenizer_spm_32k_3.model",
        "no_whitespace_before_word": False,
        "text_padding_id": 3,
        "end_of_text_padding_id": 0,
        "audio_tokenizer_frame_rate": 12.5,
        "num_workers": 1,
        "resume": False,
    },
    required=frozenset({"word_transcript_dir", "output_dir"}),
    store_true=frozenset({"no_whitespace_before_word", "resume"}),
)

# `tools/tokenize_audio.py`. `--device` defaults to cuda, so a local run that does not name
# a device does not fall back to cpu - it raises. Naming it is not optional, and which name
# was used decides whether the parquet reproduces.
TOKENIZE_AUDIO = ToolFlags(
    name="tokenize_audio",
    module="tools.tokenize_audio",
    source="tools/tokenize_audio.py",
    defaults={
        "audio_dir": None,
        "output_dir": None,
        "audio_tokenizer_repo": "kyutai/moshiko-pytorch-bf16",
        "audio_tokenizer_name": "tokenizer-e351c8d8-checkpoint125.safetensors",
        "audio_chunk_size": 1200,
        "num_workers": 1,
        "device": "cuda",
        "resume": False,
    },
    required=frozenset({"audio_dir", "output_dir"}),
    store_true=frozenset({"resume"}),
)

# `tools/prepare_dataset.py`.
PREPARE_DATASET = ToolFlags(
    name="prepare_dataset",
    module="tools.prepare_dataset",
    source="tools/prepare_dataset.py",
    defaults={
        "tokenized_text_dir": None,
        "tokenized_audio_dir": None,
        "output_prefix": None,
        "text_padding_id": 3,
        "num_examples_per_parquet": 100_000,
    },
    required=frozenset({"tokenized_text_dir", "tokenized_audio_dir", "output_prefix"}),
    store_true=frozenset(),
)

TOOLS: dict[str, ToolFlags] = {
    tool.name: tool for tool in (TOKENIZE_TEXT, TOKENIZE_AUDIO, PREPARE_DATASET)
}

# The only tool the recorder could describe when the existing sidecars were written, so an
# invocation with no `tool` key is one of these and reading it as anything else would
# rewrite history.
DEFAULT_TOOL = TOKENIZE_TEXT.name

# The order the three run in. Audio first: `prepare_dataset` trims or pads the text stream
# to the audio length, so the audio is the authority on how long a dialogue is.
PIPELINE_ORDER = (TOKENIZE_AUDIO.name, TOKENIZE_TEXT.name, PREPARE_DATASET.name)

# Measured, not assumed: TOKENIZE_COMMANDS.md section 1 tokenized the same five dialogues
# on cpu and on mps and found 2 of 21,600 tokens differ. `cuda` is absent from this set
# because it is a different machine rather than a different rounding on this one - the
# record is what lets the two be told apart afterwards.
NON_BIT_IDENTICAL_DEVICES = frozenset({"mps"})

# Backwards-compatible aliases. `tools/text_stream_audit.py` re-exports these and two test
# modules import them by these names.
TOKENIZE_TEXT_FLAGS: dict[str, Any] = TOKENIZE_TEXT.defaults
REQUIRED_FLAGS = TOKENIZE_TEXT.required
STORE_TRUE_FLAGS = TOKENIZE_TEXT.store_true


class UnknownToolError(ValueError):
    """An invocation names a tool this module holds no flag table for."""


def tool_flags(name: str) -> ToolFlags:
    try:
        return TOOLS[name]
    except KeyError:
        raise UnknownToolError(f"{name!r} is not one of {', '.join(sorted(TOOLS))}") from None


def build_parser_for(tool: ToolFlags) -> argparse.ArgumentParser:
    """Rebuild the tool's parser from the flag table, for expanding a recorded argv."""
    parser = argparse.ArgumentParser(prog=tool.source, add_help=False)
    for name, default in tool.defaults.items():
        option = f"--{name}"
        if name in tool.store_true:
            parser.add_argument(option, action="store_true")
        elif name in tool.required:
            parser.add_argument(option, type=str, required=True)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def resolve_invocation(argv: Sequence[str], *, tool: str = DEFAULT_TOOL) -> dict[str, Any]:
    """Expand one command line into a record in which every flag has a resolved value.

    The point is the expansion. `--no_whitespace_before_word` and `--resume` are
    `store_true`: a run that omits one and a run that was never written down look identical
    afterwards. Here the omission is written as `false`, which a reviewer can see and a test
    can fail on.
    """
    flags = tool_flags(tool)
    parsed = vars(build_parser_for(flags).parse_args(list(argv)))
    given = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    return {
        "tool": flags.name,
        "argv": list(argv),
        "flags": {name: parsed[name] for name in flags.defaults},
        "defaults_used": sorted(name for name in flags.defaults if f"--{name}" not in given),
    }


def render_argv(flags: Mapping[str, Any], *, tool: str = DEFAULT_TOOL) -> list[str]:
    """Render a flag mapping back into the argv that produces it.

    A `store_true` flag that is false renders as nothing, because that is what it means; a
    reader who wants to know it was false reads `flags`, not the command line. Everything
    else is emitted explicitly, defaults included - the whole reason this pipeline has a
    procedure document is that a value left to a default is a value nobody wrote down.
    """
    spec = tool_flags(tool)
    missing = [name for name in spec.defaults if name not in flags]
    if missing:
        raise ValueError(f"{tool}: no value recorded for {', '.join(missing)}")
    argv: list[str] = []
    for name in spec.defaults:
        value = flags[name]
        if name in spec.store_true:
            if value:
                argv.append(f"--{name}")
            continue
        if value is None:
            raise ValueError(f"{tool}: --{name} has no value")
        argv.extend([f"--{name}", str(value)])
    return argv


def render_command(
    flags: Mapping[str, Any], *, tool: str = DEFAULT_TOOL, prefix: Sequence[str] = ()
) -> str:
    """One shell line, quoted, ready to paste."""
    spec = tool_flags(tool)
    parts = [*prefix, "python", "-m", spec.module, *render_argv(flags, tool=tool)]
    return " ".join(shlex.quote(part) for part in parts)


def invocation_tool(invocation: Mapping[str, Any]) -> str:
    """Which tool an invocation describes, defaulting for records written before `--tool`."""
    return str(invocation.get("tool") or DEFAULT_TOOL)


def known_defects(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The declared defects of a sidecar, whether written as one dict or a list of them.

    `v-real-v1-tokenize.json` carries a single `known_defect` dict, which is the shape the
    existing tests pin. A record that has to declare two - a dropped flag and a device, say
    - writes a list. Both are read here so neither shape has to be migrated.
    """
    declared = record.get("known_defect")
    if declared is None:
        return []
    if isinstance(declared, Mapping):
        return [declared]
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        return [item for item in declared if isinstance(item, Mapping)]
    return []


def _declares_defect(record: Mapping[str, Any], flag: str) -> bool:
    return any(defect.get("flag") == flag for defect in known_defects(record))


def _same_directory(left: Any, right: Any) -> bool:
    """Compare two path flags without letting a trailing slash count as a difference."""
    if not isinstance(left, str) or not isinstance(right, str):
        return left == right
    return posixpath.normpath(left) == posixpath.normpath(right)


def check_invocation(invocation: Mapping[str, Any], *, where: str = "") -> list[str]:
    """Problems with one invocation record, as sentences. Empty means it is complete."""
    label = where or invocation_tool(invocation)
    try:
        spec = tool_flags(invocation_tool(invocation))
    except UnknownToolError as error:
        return [f"{label}: {error}"]

    problems: list[str] = []
    flags = invocation.get("flags")
    if not isinstance(flags, Mapping):
        return [f"{label}: no flags recorded"]

    missing = sorted(set(spec.defaults) - set(flags))
    extra = sorted(set(flags) - set(spec.defaults))
    if missing:
        problems.append(
            f"{label}: {', '.join(missing)} not stated; a flag left out of the record is a "
            f"flag nobody can rule out later"
        )
    if extra:
        problems.append(f"{label}: {', '.join(extra)} are not flags of {spec.source}")

    defaults_used = invocation.get("defaults_used", [])
    if not isinstance(defaults_used, Sequence) or isinstance(defaults_used, (str, bytes)):
        problems.append(f"{label}: defaults_used is not a list")
    else:
        stray = sorted(set(defaults_used) - set(spec.defaults))
        if stray:
            problems.append(f"{label}: defaults_used names {', '.join(stray)}")

    for name in spec.store_true & set(flags):
        if not isinstance(flags[name], bool):
            problems.append(f"{label}: {name} is not a stated boolean")

    return problems


def check_record(record: Mapping[str, Any], *, require_tools: Iterable[str] = ()) -> list[str]:
    """Problems with a whole tokenize sidecar, as sentences. Empty means it holds up.

    `require_tools` is what turns "the record is well formed" into "the record is complete".
    It is left to the caller because the sidecars written before this module existed
    describe `tokenize_text` alone, and rewriting them would be inventing a past.
    """
    problems: list[str] = []
    invocations = record.get("invocations")
    if not isinstance(invocations, Sequence) or not invocations:
        return ["no invocations recorded"]

    by_split: dict[str, dict[str, Mapping[str, Any]]] = {}
    for index, invocation in enumerate(invocations):
        if not isinstance(invocation, Mapping):
            problems.append(f"invocation {index}: not an object")
            continue
        tool = invocation_tool(invocation)
        split = str(invocation.get("split", ""))
        label = f"{split or '(no split)'}/{tool}"
        problems.extend(check_invocation(invocation, where=label))
        if tool in TOOLS:
            slot = by_split.setdefault(split, {})
            if tool in slot:
                problems.append(f"{label}: recorded twice for the same split")
            slot[tool] = invocation

    required = sorted(set(require_tools))
    for tool in required:
        tool_flags(tool)
    for split, slot in sorted(by_split.items()):
        for tool in required:
            if tool not in slot:
                problems.append(
                    f"{split or '(no split)'}: no {tool} invocation recorded; its flags are "
                    f"not reachable from the manifest"
                )

    for split, slot in sorted(by_split.items()):
        problems.extend(_check_split(record, split, slot))
    return problems


def _check_split(
    record: Mapping[str, Any], split: str, slot: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    problems: list[str] = []
    label = split or "(no split)"

    audio = slot.get(TOKENIZE_AUDIO.name, {}).get("flags", {})
    text = slot.get(TOKENIZE_TEXT.name, {}).get("flags", {})
    parquet = slot.get(PREPARE_DATASET.name, {}).get("flags", {})

    device = audio.get("device")
    if audio and device in NON_BIT_IDENTICAL_DEVICES and not _declares_defect(record, "device"):
        problems.append(
            f"{label}: tokenized on {device}, which is not bit-identical to cpu "
            f"(TOKENIZE_COMMANDS.md section 1: 2 of 21,600 tokens differ), and the record "
            f"declares no known_defect for device"
        )

    if text and parquet:
        if text.get("text_padding_id") != parquet.get("text_padding_id"):
            problems.append(
                f"{label}: tokenize_text --text_padding_id {text.get('text_padding_id')!r} "
                f"and prepare_dataset --text_padding_id {parquet.get('text_padding_id')!r} "
                f"disagree; two ids spell silence in one stream"
            )
        if not _same_directory(text.get("output_dir"), parquet.get("tokenized_text_dir")):
            problems.append(
                f"{label}: prepare_dataset read {parquet.get('tokenized_text_dir')!r}, which "
                f"is not where tokenize_text wrote ({text.get('output_dir')!r})"
            )
    if audio and parquet:
        if not _same_directory(audio.get("output_dir"), parquet.get("tokenized_audio_dir")):
            problems.append(
                f"{label}: prepare_dataset read {parquet.get('tokenized_audio_dir')!r}, which "
                f"is not where tokenize_audio wrote ({audio.get('output_dir')!r})"
            )
    if parquet and split:
        prefix = parquet.get("output_prefix")
        if isinstance(prefix, str) and posixpath.basename(prefix.rstrip("/")) != split:
            problems.append(
                f"{label}: --output_prefix basename is "
                f"{posixpath.basename(prefix.rstrip('/'))!r}, so the dialogue_id namespace "
                f"will not be {split!r} and the parquet will not join to the manifest"
            )
    return problems
