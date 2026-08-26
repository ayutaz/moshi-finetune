"""Write down what an artifact was built from, and check the materials still match.

Why this exists
---------------
M3-R 第2段 assembled its stereo dialogues at 20:27 on 2026-08-25. Between 22:27 and 22:42
that same evening every speaker-B backchannel wav was regenerated, because
``tools/synthesize_turns.py:95-115`` records a measured TTS defect - asking for 1.1 s of
audio for a two-mora word makes the model say a word that does not exist, 0 of 6 correct -
and the fix was to shorten them to 0.57 s. The stereo files were not rebuilt.

Nothing failed. Every file was present. Every checksum in every manifest matched the file
it described, because the stereo files were still exactly the bytes that had been written
down. The timeline report was green on all six gates. The dataset was, by every check the
repository had, in order.

It was found by hand, much later: the backchannel span of ``v-001`` - 11.47 s to 12.44 s -
was cut out of the assembled stereo and correlated against the current
``turns-B-backchannel/v-001-t2-B.wav``. The result was **-0.0012**, where the same
waveform against itself gives +1.0. The audio inside those stereo files exists nowhere
else on disk, so the whole second stage had to be built again.

A manifest checksum cannot catch this, and it is worth being precise about why. It answers
*"is this file the file I wrote down?"*, and the answer was yes throughout. The question
nobody was able to ask is *"are the materials on disk still the materials this was built
from?"* - because the materials were never written down at build time. Only the product
was.

This module writes down the materials. :func:`fingerprint_sources` collects the sha256 of
every input a build consumed, :func:`build_record` files them beside the artifact, and
:func:`verify_record` re-walks the same sources and names every file that has since been
added, removed or rewritten. Had the record existed on 2026-08-25, the 22:27 regeneration
would have made the next verification say::

    changed: data/.../m3r/turns-B-backchannel/v-001-t2-B.wav (backchannel) <20:27> -> <22:27>

seventy-eight times, and the stereo would have been rebuilt that evening rather than
discovered stale a day later.

Three decisions, and the measurements behind them
-------------------------------------------------
**Content, never mtime.** The 20:27/22:27 timestamps are how the accident was eventually
diagnosed, but they are the wrong thing to gate on in either direction. ``cp``, ``rsync``,
a restore from backup and a branch checkout all rewrite mtimes while changing nothing, so
an mtime gate cries wolf on a move; ``touch`` rewrites an mtime with no content change at
all, and ``cp -p`` preserves an mtime *across* a content change. sha256 answers the
question that was actually asked.

**Hash everything, every time; no size shortcut.** A byte_size mismatch does imply a
content mismatch, so it could stand in for a hash and skip the read - but the premise that
this is slow was measured before it was built on. The real M3-R input set is 244 files and
173.3 MB (the v2 script and split map, the M3 speaker-A stereo, the word transcripts, the
78 rebuilt backchannel wavs and the room-tone pool); ``verify`` reads and hashes all of it
in **0.182 s** wall, and the hashing loop alone runs at 899 MB/s on this machine. A
shortcut that saves a fifth of a second and costs the report its ability to state the new
checksum is not worth having. Reads are chunked at 1 MiB so memory stays flat whatever the
file size, which is what makes a 285 MB stereo directory affordable to verify as well.

Also measured, on that same real tree: a single byte rewritten inside
``turns-B-backchannel/v-001-t2-B.wav`` at the same byte count - the shape of the 22:27
regeneration - is reported as ``changed`` with both checksums and exits 1; an extra wav
dropped into that directory is reported as ``added``; a deleted one as ``removed``; and a
``.DS_Store`` written beside them changes nothing.

**Sources are re-walked, not just re-hashed.** The record keeps the *directories and globs*
that were consumed, not only the file list they expanded to. Re-hashing a stored list can
only find files that changed or vanished; re-walking finds the file that appeared. That is
the difference between "素材が変わった" and "素材が増えた", and a dataset that silently
gained an 81st dialogue is as wrong as one that silently lost one.

Layout
------
The counting is pure and the reading is not, so they are separate and the pure half is
what ``tests/test_provenance.py`` exercises: :func:`relative_posix`,
:func:`diff_fingerprints`, :func:`describe_diff` and :func:`build_record` never touch a
filesystem. Only :func:`sha256_file`, :func:`expand_source`, :func:`fingerprint_sources`
and :func:`verify_record` do.

Standard library only, like ``tools/experiment_data.py`` and the JSON half of
``tools/text_stream_audit.py``: this has to run in CI, in a pre-commit hook and on a rented
GPU box that has nothing installed.

Usage
-----
Record, immediately after a build, naming every material it consumed::

    python -m tools.provenance record \\
      --out experiments/tsukuyomi_ojousama/manifests/v-real-v2-provenance.json \\
      --artifact-id v-real-v2 --tool tools/assemble_dialogue.py \\
      --captured-at 2026-08-26 \\
      --why "M3-R 第2段: stereo rebuilt with group_size=1" \\
      --input scripts=experiments/tsukuyomi_ojousama/m3r/scripts/dialogues-v2.jsonl \\
      --input backchannel=data/experiments/tsukuyomi_ojousama/m3r/turns-B-backchannel \\
      --output stereo=data/experiments/tsukuyomi_ojousama/m3r/v-real/audio

Verify, before anything downstream trusts the artifact::

    python -m tools.provenance verify \\
      --record experiments/tsukuyomi_ojousama/manifests/v-real-v2-provenance.json

``verify`` exits 1 on any mismatch, so it is usable as a gate in a hook or a shell script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1

# Matches `tools/experiment_data.py:_sha256`. One dialect for checksums in this
# repository, not two; `tests/test_provenance.py` asserts the two still agree.
CHUNK_BYTES = 1024 * 1024

# The two groups a record carries. Inputs are what the build consumed; outputs are what it
# produced. Both are verified the same way, because "the artifact itself was overwritten"
# is the same class of accident as "the material was overwritten".
GROUPS = ("inputs", "outputs")

# Characters that make a source path a glob rather than a literal name.
GLOB_MAGIC = frozenset("*?[")

# How much of a checksum `describe_diff` prints. Enough to tell two files apart by eye
# without wrapping the line; the full value is in the JSON.
DIGEST_PREVIEW = 12


class ProvenanceError(ValueError):
    """Raised when a provenance record or the material it describes is unusable."""


@dataclass(frozen=True)
class Source:
    """One material a build consumed, as it was named at build time.

    `path` is root-relative POSIX. `kind` is derived, not asserted: a path with glob magic
    in it is a glob, and the rest are resolved against the filesystem when they are walked.
    """

    role: str
    path: str

    @property
    def is_glob(self) -> bool:
        return any(character in GLOB_MAGIC for character in self.path)

    def as_json(self) -> dict[str, Any]:
        return {"role": self.role, "path": self.path, "kind": "glob" if self.is_glob else "path"}


# --------------------------------------------------------------------------------------
# Pure logic. No filesystem below this line until the I/O section.
# --------------------------------------------------------------------------------------


def relative_posix(path: Path | PurePosixPath | str, root: Path | PurePosixPath | str) -> str:
    """Express `path` relative to `root` as POSIX, refusing anything that escapes it.

    Lexical, so it is testable without a filesystem and cannot be fooled into a `stat` on
    a path the caller has not vetted. The escape check is the same guarantee
    `tools/experiment_data.py:validate_manifest` makes with `path.relative_to(data_root)`:
    a record whose paths reach outside the root it declares is a record that cannot be
    verified on another checkout.

    `relative_to` alone is not that guarantee. `PurePosixPath("/a/b/../c")` is lexically
    under `/a/b` and `relative_to` returns `../c` without complaint, so a source written
    as `../something` - or a glob that walks out through `..` - would be recorded as a
    root-relative path that resolves somewhere else entirely on the next checkout. A `..`
    left in the result is therefore refused outright.
    """
    pure_path = PurePosixPath(Path(path).as_posix())
    pure_root = PurePosixPath(Path(root).as_posix())
    try:
        relative = pure_path.relative_to(pure_root)
    except ValueError as error:
        raise ProvenanceError(f"{pure_path} is outside the root {pure_root}") from error
    text = relative.as_posix()
    if text == ".":
        raise ProvenanceError(f"{pure_path} is the root itself, not a file inside it")
    if ".." in relative.parts:
        raise ProvenanceError(f"{pure_path} climbs out of the root {pure_root} through '..'")
    return text


def _index(entries: Iterable[Mapping[str, Any]], *, side: str) -> dict[str, dict[str, Any]]:
    """Key fingerprints by path, collapsing exact duplicates and refusing inexact ones.

    Two sources may legitimately cover the same file - a directory and a named file inside
    it - and that is harmless as long as they agree. Two entries for one path that
    disagree about its checksum mean the record was written from an inconsistent read, and
    silently keeping either one would make the diff a coin toss.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = str(entry["path"])
        previous = indexed.get(path)
        if previous is None:
            indexed[path] = dict(entry)
            continue
        if (previous.get("sha256"), previous.get("byte_size")) != (
            entry.get("sha256"),
            entry.get("byte_size"),
        ):
            raise ProvenanceError(
                f"{side}: {path} appears twice with different fingerprints "
                f"({previous.get('sha256')} and {entry.get('sha256')})"
            )
    return indexed


def diff_fingerprints(
    recorded: Iterable[Mapping[str, Any]],
    observed: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Name every file that differs between a record and what is on disk now.

    The three cases are kept apart because they mean different things. `changed` is the
    2026-08-25 accident: the file is where it was and holds something else. `removed` is a
    material the build consumed that is no longer there, so the artifact cannot be rebuilt.
    `added` is a material that appeared under a recorded source after the build, so the
    artifact was built from less than the source now holds.

    Both sides are `{"path", "sha256", "byte_size", "role"}` mappings, in any order.
    """
    left = _index(recorded, side="recorded")
    right = _index(observed, side="observed")

    changed: list[dict[str, Any]] = []
    unchanged = 0
    for path in sorted(left.keys() & right.keys()):
        before, after = left[path], right[path]
        if before.get("sha256") == after.get("sha256"):
            unchanged += 1
            continue
        changed.append(
            {
                "path": path,
                "role": before.get("role") or after.get("role"),
                "recorded_sha256": before.get("sha256"),
                "observed_sha256": after.get("sha256"),
                "recorded_byte_size": before.get("byte_size"),
                "observed_byte_size": after.get("byte_size"),
            }
        )

    def side_entries(paths: Iterable[str], source: Mapping[str, dict[str, Any]]) -> list[dict]:
        return [
            {
                "path": path,
                "role": source[path].get("role"),
                "sha256": source[path].get("sha256"),
                "byte_size": source[path].get("byte_size"),
            }
            for path in sorted(paths)
        ]

    removed = side_entries(left.keys() - right.keys(), left)
    added = side_entries(right.keys() - left.keys(), right)
    roles = {entry.get("role") for entry in (*changed, *removed, *added)}

    return {
        "status": "match" if not (changed or removed or added) else "mismatch",
        "counts": {
            "recorded": len(left),
            "observed": len(right),
            "unchanged": unchanged,
            "changed": len(changed),
            "removed": len(removed),
            "added": len(added),
        },
        "changed": changed,
        "removed": removed,
        "added": added,
        "roles_affected": sorted(role for role in roles if role),
    }


def describe_diff(diff: Mapping[str, Any], *, group: str = "") -> list[str]:
    """One line per differing file, for a shell or an assertion message.

    A verification that says only "mismatch" sends the reader back to the JSON. The point
    of this module is that the answer is a file name, so the human-readable form carries
    the file name too.
    """
    prefix = f"{group}: " if group else ""
    if diff["status"] == "match":
        counts = diff["counts"]
        return [f"{prefix}match ({counts['unchanged']} files unchanged)"]

    def short(digest: Any) -> str:
        return str(digest)[:DIGEST_PREVIEW] if digest else "-"

    lines = []
    for entry in diff["changed"]:
        role = f" ({entry['role']})" if entry.get("role") else ""
        lines.append(
            f"{prefix}changed: {entry['path']}{role} "
            f"{short(entry['recorded_sha256'])} -> {short(entry['observed_sha256'])}"
        )
    for entry in diff["removed"]:
        role = f" ({entry['role']})" if entry.get("role") else ""
        lines.append(f"{prefix}removed: {entry['path']}{role} {short(entry['sha256'])}")
    for entry in diff["added"]:
        role = f" ({entry['role']})" if entry.get("role") else ""
        lines.append(f"{prefix}added: {entry['path']}{role} {short(entry['sha256'])}")
    return lines


def build_record(
    *,
    artifact_id: str,
    tool: str,
    captured_at: str,
    why: str,
    root: str,
    inputs: Iterable[Mapping[str, Any]],
    input_sources: Iterable[Source | Mapping[str, Any]] = (),
    outputs: Iterable[Mapping[str, Any]] = (),
    output_sources: Iterable[Source | Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the record. Pure, so the shape can be tested without building anything.

    `why` is required rather than optional, for the same reason a registry entry carries a
    rationale: a record whose reason for existing is not written down is one the next
    reader deletes.
    """
    if not artifact_id or not tool or not captured_at or not why:
        raise ProvenanceError("a record needs artifact_id, tool, captured_at and why")
    if not root:
        raise ProvenanceError("a record needs a root the paths are relative to")

    def group(
        sources: Iterable[Source | Mapping[str, Any]], files: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return {
            "sources": [
                source.as_json() if isinstance(source, Source) else dict(source)
                for source in sources
            ],
            "files": sorted(
                (dict(entry) for entry in files),
                key=lambda entry: (str(entry["path"]), str(entry.get("role") or "")),
            ),
        }

    inputs_group = group(input_sources, inputs)
    if not inputs_group["files"]:
        raise ProvenanceError("a record with no inputs proves nothing; name what was consumed")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "tool": tool,
        "captured_at": captured_at,
        "why": why,
        "root": root,
        "inputs": inputs_group,
        "outputs": group(output_sources, outputs),
    }
    record.update(extra or {})
    return record


def sources_of(record: Mapping[str, Any], group: str) -> list[Source]:
    """The sources a group declares, as `Source` objects."""
    return [
        Source(role=str(entry.get("role") or ""), path=str(entry["path"]))
        for entry in record.get(group, {}).get("sources", [])
    ]


# --------------------------------------------------------------------------------------
# I/O. Everything below reads the filesystem.
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Streaming sha256, 1 MiB at a time.

    Chunked rather than `path.read_bytes()` so a 285 MB stereo directory does not decide
    how much memory the verification needs.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hidden(relative: str) -> bool:
    """A dotted component anywhere in the path.

    macOS writes `.DS_Store` into any directory the Finder opens - there is one in
    `data/experiments/tsukuyomi_ojousama/m2/` right now - and a verification that goes red
    because somebody looked at a folder is a verification people learn to ignore. Editor
    swap files and `.git` are the same story. Recorded material is never hidden, so this
    costs nothing; `--include-hidden` is there for the case where it does.
    """
    return any(part.startswith(".") for part in PurePosixPath(relative).parts)


def expand_source(source: Source, *, root: Path, include_hidden: bool = False) -> list[Path]:
    """Every file a source names, sorted, resolved against `root`.

    A directory expands to its whole subtree; a glob expands to its matches; a file is
    itself. The result is sorted so a record built twice from the same tree is byte
    identical, which is what lets a record be diffed in review.
    """
    root = Path(root)
    if source.is_glob:
        candidates = sorted(root.glob(source.path))
    else:
        target = root / source.path
        if target.is_dir():
            candidates = sorted(target.rglob("*"))
        elif target.is_file():
            candidates = [target]
        elif target.is_symlink():
            raise ProvenanceError(f"{source.role or 'source'}: {source.path} is a broken symlink")
        else:
            raise ProvenanceError(f"{source.role or 'source'}: {source.path} does not exist")

    files = []
    for candidate in candidates:
        if candidate.is_symlink() and not candidate.exists():
            raise ProvenanceError(
                f"{source.role or 'source'}: {relative_posix(candidate, root)} is a broken symlink"
            )
        if not candidate.is_file():
            continue
        relative = relative_posix(candidate, root)
        if not include_hidden and _is_hidden(relative):
            continue
        files.append(candidate)

    if not files:
        raise ProvenanceError(
            f"{source.role or 'source'}: {source.path} matched no files. A source that "
            "matches nothing makes verification vacuously green."
        )
    return files


def fingerprint_sources(
    sources: Sequence[Source], *, root: Path, include_hidden: bool = False
) -> list[dict[str, Any]]:
    """Collect `{path, role, sha256, byte_size}` for every file the sources name.

    This is the collecting half the docstring at the top promises; `diff_fingerprints` is
    the comparing half. They are separate so the comparison can be tested against
    hand-written fingerprints with no data on disk at all.
    """
    root = Path(root)
    entries: list[dict[str, Any]] = []
    for source in sources:
        for path in expand_source(source, root=root, include_hidden=include_hidden):
            entries.append(
                {
                    "role": source.role,
                    "path": relative_posix(path, root),
                    "sha256": sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
            )
    return sorted(entries, key=lambda entry: (entry["path"], entry["role"]))


def repository_root() -> Path:
    """The checkout this file lives in. Records are repository-relative so they travel."""
    return Path(__file__).resolve().parents[1]


def resolve_root(record_root: str, *, repository: Path | None = None) -> Path:
    """Turn a record's `root` back into a directory on this machine."""
    root = Path(record_root)
    if root.is_absolute():
        return root
    base = Path(repository) if repository is not None else repository_root()
    return base if record_root == "." else base / record_root


def record_root_field(root: Path, *, repository: Path | None = None) -> str:
    """`root` as a repository-relative string, or absolute when it lies outside.

    An absolute path in a committed record is a path that only works on the machine that
    wrote it - `reports/m3r-timeline.json` already carries one - so the repository-relative
    form is preferred wherever it exists.
    """
    base = (Path(repository) if repository is not None else repository_root()).resolve()
    root = Path(root).resolve()
    if root == base:
        return "."
    try:
        return relative_posix(root, base)
    except ProvenanceError:
        return root.as_posix()


def write_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """File the record as JSON, in the shape the rest of `manifests/` is written in."""
    payload = dict(record)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def read_record(path: Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(
            f"{path}: schema_version is {record.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    for field in ("artifact_id", "root", "inputs"):
        if not record.get(field):
            raise ProvenanceError(f"{path}: no {field}")
    return record


def verify_record(
    record: Mapping[str, Any],
    *,
    root: Path | None = None,
    repository: Path | None = None,
    include_hidden: bool = False,
    groups: Sequence[str] = GROUPS,
) -> dict[str, Any]:
    """Re-walk the recorded sources and report every file that no longer matches.

    Sources are re-walked rather than the stored file list re-hashed, so a file that
    appeared under a recorded directory after the build is reported as `added` instead of
    being invisible. Where a group declares no sources - a hand-written record, or one
    written before this module existed - the stored paths are used and
    `detects_additions` says false, because in that case they genuinely cannot be found.
    """
    root_path = (
        Path(root) if root is not None else resolve_root(record["root"], repository=repository)
    )
    results: dict[str, Any] = {}
    for group in groups:
        block = record.get(group) or {}
        recorded = list(block.get("files", []))
        if not recorded and not block.get("sources"):
            continue
        sources = sources_of(record, group)
        if sources:
            observed = fingerprint_sources(sources, root=root_path, include_hidden=include_hidden)
            detects_additions = True
        else:
            observed = []
            for entry in recorded:
                path = root_path / str(entry["path"])
                if not path.is_file():
                    continue
                observed.append(
                    {
                        "role": entry.get("role"),
                        "path": str(entry["path"]),
                        "sha256": sha256_file(path),
                        "byte_size": path.stat().st_size,
                    }
                )
            detects_additions = False
        diff = diff_fingerprints(recorded, observed)
        diff["detects_additions"] = detects_additions
        results[group] = diff

    if not results:
        # `all()` over nothing is True, so this would otherwise report a clean verification
        # of a record it never looked at - the same vacuous green `expand_source` refuses.
        raise ProvenanceError(f"nothing to verify: the record declares no files in {list(groups)}")

    status = "match" if all(diff["status"] == "match" for diff in results.values()) else "mismatch"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": record.get("artifact_id"),
        "tool": record.get("tool"),
        "captured_at": record.get("captured_at"),
        "root": str(root_path),
        "status": status,
        "groups": results,
        "report": [
            line for group, diff in results.items() for line in describe_diff(diff, group=group)
        ],
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_source(argument: str) -> Source:
    role, separator, path = argument.partition("=")
    if not separator or not role.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected role=path, got {argument!r}")
    return Source(role=role.strip(), path=PurePosixPath(path.strip()).as_posix())


def _cmd_record(args: argparse.Namespace) -> None:
    repository = repository_root()
    root = Path(args.root) if args.root else repository
    inputs = fingerprint_sources(args.input, root=root, include_hidden=args.include_hidden)
    outputs = (
        fingerprint_sources(args.output, root=root, include_hidden=args.include_hidden)
        if args.output
        else []
    )
    record = build_record(
        artifact_id=args.artifact_id,
        tool=args.tool,
        captured_at=args.captured_at,
        why=args.why,
        root=record_root_field(root, repository=repository),
        inputs=inputs,
        input_sources=args.input,
        outputs=outputs,
        output_sources=args.output,
    )
    write_record(Path(args.out), record)
    counts = {"inputs": len(inputs), "outputs": len(outputs)}
    print(json.dumps({"wrote": args.out, "files": counts}, ensure_ascii=False, indent=2))


def _cmd_verify(args: argparse.Namespace) -> None:
    record = read_record(Path(args.record))
    groups = ("inputs",) if args.skip_outputs else GROUPS
    result = verify_record(
        record,
        root=Path(args.root) if args.root else None,
        include_hidden=args.include_hidden,
        groups=groups,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    if args.quiet:
        for line in result["report"]:
            print(line)
    else:
        print(payload)
    if result["status"] != "match":
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="write down what an artifact was built from")
    record.add_argument("--out", required=True, help="where to file the record")
    record.add_argument("--artifact-id", required=True)
    record.add_argument(
        "--tool", required=True, help="the builder, e.g. tools/assemble_dialogue.py"
    )
    record.add_argument("--captured-at", required=True)
    record.add_argument("--why", required=True, help="why this build happened")
    record.add_argument(
        "--root",
        default="",
        help="paths are recorded relative to this directory; defaults to the repository root",
    )
    record.add_argument(
        "--input",
        action="append",
        default=[],
        type=_parse_source,
        metavar="ROLE=PATH",
        help="a material the build consumed: a file, a directory or a glob. Repeatable.",
    )
    record.add_argument(
        "--output",
        action="append",
        default=[],
        type=_parse_source,
        metavar="ROLE=PATH",
        help="a product of the build, verified the same way. Repeatable.",
    )
    record.add_argument(
        "--include-hidden",
        action="store_true",
        help="also fingerprint dotted files such as .DS_Store",
    )
    record.set_defaults(func=_cmd_record)

    verify = sub.add_parser("verify", help="check the materials still match the record")
    verify.add_argument("--record", required=True)
    verify.add_argument("--root", default="", help="verify against this directory instead")
    verify.add_argument("--out", default="", help="also write the JSON result here")
    verify.add_argument("--skip-outputs", action="store_true", help="check inputs only")
    verify.add_argument("--include-hidden", action="store_true")
    verify.add_argument("--quiet", action="store_true", help="print the report lines, not the JSON")
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
