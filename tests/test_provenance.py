"""What `tools/provenance.py` has to get right, and the accident that says why.

On 2026-08-25 the M3-R stereo was assembled at 20:27 and the backchannel wavs it contains
were regenerated at 22:27. Every checksum in the repository still matched, because every
checksum described a product and none described a material. The tests below are written
against that shape: the pure comparison is exercised on hand-written fingerprints with no
data on disk, and the filesystem tests rebuild the accident in miniature and require the
verification to name the file.
"""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.experiment_data import _sha256 as experiment_data_sha256
from tools.provenance import (
    GROUPS,
    SCHEMA_VERSION,
    ProvenanceError,
    Source,
    build_record,
    describe_diff,
    diff_fingerprints,
    expand_source,
    fingerprint_sources,
    main,
    read_record,
    record_root_field,
    relative_posix,
    repository_root,
    resolve_root,
    sha256_file,
    sources_of,
    verify_record,
    write_record,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path: str, digest: str, *, role: str = "material", size: int = 10) -> dict:
    """A fingerprint as the record holds one, without touching a filesystem."""
    return {"path": path, "role": role, "sha256": digest, "byte_size": size}


class RelativePathTests(unittest.TestCase):
    """A record that cannot state a portable path cannot be verified on another checkout."""

    def test_a_path_under_the_root_becomes_root_relative_posix(self) -> None:
        self.assertEqual(
            relative_posix("/repo/data/m3r/turns-B-backchannel/v-001-t2-B.wav", "/repo"),
            "data/m3r/turns-B-backchannel/v-001-t2-B.wav",
        )

    def test_relative_inputs_work_the_same_way(self) -> None:
        self.assertEqual(
            relative_posix("data/m3r/roomtone/index.json", "data"), "m3r/roomtone/index.json"
        )

    def test_a_path_outside_the_root_is_refused(self) -> None:
        with self.assertRaises(ProvenanceError):
            relative_posix("/elsewhere/audio.wav", "/repo")

    def test_a_sibling_that_merely_shares_a_prefix_is_outside(self) -> None:
        """`/repo/data` is not a prefix of `/repo/database` in any sense that matters."""
        with self.assertRaises(ProvenanceError):
            relative_posix("/repo/database/audio.wav", "/repo/data")

    def test_the_root_itself_is_not_a_file_inside_it(self) -> None:
        with self.assertRaises(ProvenanceError):
            relative_posix("/repo/data", "/repo/data")

    def test_a_path_that_climbs_out_through_dotdot_is_refused(self) -> None:
        """`relative_to` alone would allow this, and it is the escape that matters.

        `PurePosixPath("/repo/data/../secrets")` is lexically under `/repo/data`, so
        `relative_to` hands back `../secrets` with no complaint. Recorded, that path would
        resolve somewhere else on the next checkout while still looking root-relative.
        """
        with self.assertRaises(ProvenanceError):
            relative_posix("/repo/data/../secrets/key.txt", "/repo/data")


class FingerprintDiffTests(unittest.TestCase):
    """The comparison. Pure, so it is tested on fingerprints rather than on files.

    Three outcomes are kept apart on purpose. A material rewritten in place is the
    2026-08-25 accident; a material that vanished means the artifact cannot be rebuilt; a
    material that appeared means the artifact was built from less than the source holds
    now. Collapsing them into one boolean would have made this module as uninformative as
    the checks it replaces.
    """

    def test_identical_sides_match(self) -> None:
        side = [fingerprint("a.wav", "aa"), fingerprint("b.wav", "bb")]

        diff = diff_fingerprints(side, list(side))

        self.assertEqual(diff["status"], "match")
        self.assertEqual(diff["counts"]["unchanged"], 2)
        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["roles_affected"], [])

    def test_a_rewritten_file_is_changed_and_is_named(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("turns-B-backchannel/v-001-t2-B.wav", "old", role="backchannel")],
            [fingerprint("turns-B-backchannel/v-001-t2-B.wav", "new", role="backchannel")],
        )

        self.assertEqual(diff["status"], "mismatch")
        self.assertEqual(
            diff["counts"],
            {
                "recorded": 1,
                "observed": 1,
                "unchanged": 0,
                "changed": 1,
                "removed": 0,
                "added": 0,
            },
        )
        self.assertEqual(diff["changed"][0]["path"], "turns-B-backchannel/v-001-t2-B.wav")
        self.assertEqual(diff["changed"][0]["recorded_sha256"], "old")
        self.assertEqual(diff["changed"][0]["observed_sha256"], "new")
        self.assertEqual(diff["roles_affected"], ["backchannel"])

    def test_a_file_that_is_gone_is_removed_not_changed(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("a.wav", "aa"), fingerprint("b.wav", "bb")], [fingerprint("a.wav", "aa")]
        )

        self.assertEqual(diff["counts"]["removed"], 1)
        self.assertEqual(diff["counts"]["changed"], 0)
        self.assertEqual(diff["removed"][0]["path"], "b.wav")
        self.assertEqual(diff["removed"][0]["sha256"], "bb")

    def test_a_file_that_appeared_is_added_not_changed(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("a.wav", "aa")], [fingerprint("a.wav", "aa"), fingerprint("c.wav", "cc")]
        )

        self.assertEqual(diff["counts"]["added"], 1)
        self.assertEqual(diff["counts"]["changed"], 0)
        self.assertEqual(diff["added"][0]["path"], "c.wav")

    def test_the_three_kinds_are_reported_together(self) -> None:
        diff = diff_fingerprints(
            [
                fingerprint("keep.wav", "kk"),
                fingerprint("edit.wav", "old"),
                fingerprint("gone.wav", "gg"),
            ],
            [
                fingerprint("keep.wav", "kk"),
                fingerprint("edit.wav", "new"),
                fingerprint("new.wav", "nn"),
            ],
        )

        self.assertEqual(diff["counts"]["unchanged"], 1)
        self.assertEqual([entry["path"] for entry in diff["changed"]], ["edit.wav"])
        self.assertEqual([entry["path"] for entry in diff["removed"]], ["gone.wav"])
        self.assertEqual([entry["path"] for entry in diff["added"]], ["new.wav"])

    def test_a_size_change_alone_is_not_what_decides(self) -> None:
        """The checksum decides, not the size.

        A regenerated wav can land on exactly the same byte count - `はい。` and `ええ。`
        are both two mora and both render to the same length - so a gate that trusted
        byte_size would have passed the 22:27 regeneration for at least some files.
        """
        diff = diff_fingerprints(
            [fingerprint("v-001-t2-B.wav", "old", size=27648)],
            [fingerprint("v-001-t2-B.wav", "new", size=27648)],
        )

        self.assertEqual(diff["status"], "mismatch")
        self.assertEqual(diff["changed"][0]["recorded_byte_size"], 27648)
        self.assertEqual(diff["changed"][0]["observed_byte_size"], 27648)

    def test_entries_are_sorted_so_two_runs_produce_the_same_report(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("b.wav", "1"), fingerprint("a.wav", "1"), fingerprint("c.wav", "1")],
            [fingerprint("b.wav", "2"), fingerprint("a.wav", "2"), fingerprint("c.wav", "2")],
        )

        self.assertEqual([entry["path"] for entry in diff["changed"]], ["a.wav", "b.wav", "c.wav"])

    def test_two_sources_covering_one_file_agree_and_are_counted_once(self) -> None:
        """A directory and a named file inside it is a legitimate overlap."""
        diff = diff_fingerprints(
            [fingerprint("a.wav", "aa", role="dir"), fingerprint("a.wav", "aa", role="named")],
            [fingerprint("a.wav", "aa", role="dir")],
        )

        self.assertEqual(diff["status"], "match")
        self.assertEqual(diff["counts"]["recorded"], 1)

    def test_two_entries_for_one_path_that_disagree_are_refused(self) -> None:
        """Keeping either one would make the answer depend on iteration order."""
        with self.assertRaises(ProvenanceError):
            diff_fingerprints(
                [fingerprint("a.wav", "aa"), fingerprint("a.wav", "bb")],
                [fingerprint("a.wav", "aa")],
            )

    def test_empty_sides_match(self) -> None:
        self.assertEqual(diff_fingerprints([], [])["status"], "match")


class DiffDescriptionTests(unittest.TestCase):
    """ "mismatch" sends the reader back to the JSON. A file name does not."""

    def test_a_changed_file_is_named_with_both_checksums(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("t/v-001-t2-B.wav", "6a2f" + "0" * 60, role="backchannel")],
            [fingerprint("t/v-001-t2-B.wav", "91cd" + "1" * 60, role="backchannel")],
        )

        line = describe_diff(diff, group="inputs")[0]

        self.assertIn("t/v-001-t2-B.wav", line)
        self.assertIn("backchannel", line)
        self.assertIn("6a2f", line)
        self.assertIn("91cd", line)

    def test_a_match_says_how_many_files_were_checked(self) -> None:
        side = [fingerprint("a.wav", "aa"), fingerprint("b.wav", "bb")]

        self.assertEqual(
            describe_diff(diff_fingerprints(side, list(side))), ["match (2 files unchanged)"]
        )

    def test_every_differing_file_gets_its_own_line(self) -> None:
        diff = diff_fingerprints(
            [fingerprint("edit.wav", "old"), fingerprint("gone.wav", "gg")],
            [fingerprint("edit.wav", "new"), fingerprint("new.wav", "nn")],
        )

        lines = describe_diff(diff)

        self.assertEqual(len(lines), 3)
        self.assertTrue(any(line.startswith("changed: edit.wav") for line in lines))
        self.assertTrue(any(line.startswith("removed: gone.wav") for line in lines))
        self.assertTrue(any(line.startswith("added: new.wav") for line in lines))


class RecordShapeTests(unittest.TestCase):
    """The record is JSON that goes next to a manifest, so its shape is pinned here."""

    def _record(self, **overrides):
        arguments = {
            "artifact_id": "v-real-v2",
            "tool": "tools/assemble_dialogue.py",
            "captured_at": "2026-08-26",
            "why": "M3-R 第2段: stereo rebuilt with group_size=1",
            "root": ".",
            "inputs": [fingerprint("b.wav", "bb"), fingerprint("a.wav", "aa")],
            "input_sources": [Source("backchannel", "data/m3r/turns-B-backchannel")],
        }
        arguments.update(overrides)
        return build_record(**arguments)

    def test_the_record_carries_the_repository_schema_version(self) -> None:
        self.assertEqual(self._record()["schema_version"], SCHEMA_VERSION)

    def test_files_are_sorted_so_two_records_of_one_tree_are_byte_identical(self) -> None:
        self.assertEqual(
            [entry["path"] for entry in self._record()["inputs"]["files"]], ["a.wav", "b.wav"]
        )

    def test_sources_survive_the_round_trip_as_json(self) -> None:
        record = json.loads(json.dumps(self._record()))

        self.assertEqual(
            sources_of(record, "inputs"), [Source("backchannel", "data/m3r/turns-B-backchannel")]
        )

    def test_a_glob_source_is_labelled_as_one(self) -> None:
        record = self._record(input_sources=[Source("backchannel", "data/m3r/*.wav")])

        self.assertEqual(record["inputs"]["sources"][0]["kind"], "glob")

    def test_a_record_with_no_inputs_is_refused(self) -> None:
        """It would verify green forever while proving nothing."""
        with self.assertRaises(ProvenanceError):
            self._record(inputs=[])

    def test_a_record_without_a_reason_is_refused(self) -> None:
        """Registry entries carry a rationale; so does this."""
        with self.assertRaises(ProvenanceError):
            self._record(why="")

    def test_outputs_are_optional_and_default_to_empty(self) -> None:
        self.assertEqual(self._record()["outputs"], {"sources": [], "files": []})


class ChecksumDialectTests(unittest.TestCase):
    """One dialect for checksums in this repository, not two.

    `tools/experiment_data.py` already hashes files for the manifests, and a second
    implementation that disagreed with it - a different chunk boundary, a text-mode read -
    would put two incompatible checksums in the same tree.
    """

    def test_it_agrees_with_hashlib_over_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.bin"
            payload = bytes(range(256)) * 8192  # larger than one 1 MiB chunk
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_it_agrees_with_the_manifest_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.bin"
            path.write_bytes(b"\x00\xff" * 1000)

            self.assertEqual(sha256_file(path), experiment_data_sha256(path))

    def test_an_empty_file_still_gets_a_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.bin"
            path.write_bytes(b"")

            self.assertEqual(sha256_file(path), hashlib.sha256(b"").hexdigest())


class TreeTestCase(unittest.TestCase):
    """A material tree shaped like the M3-R one, small enough to build in a test."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        # One level down, so a test can put something *outside* the root and still have
        # the temporary directory clean it up.
        self.root = Path(self._directory.name) / "tree"
        self.root.mkdir()
        backchannel = self.root / "m3r" / "turns-B-backchannel"
        backchannel.mkdir(parents=True)
        for index in (1, 2, 3):
            (backchannel / f"v-{index:03d}-t2-B.wav").write_bytes(b"RIFF-backchannel-%d" % index)
        (backchannel / "synthesis-log.jsonl").write_text(
            '{"turn": "v-001-t2-B"}\n', encoding="utf-8"
        )
        roomtone = self.root / "m3r" / "roomtone"
        roomtone.mkdir(parents=True)
        (roomtone / "index.json").write_text('{"segments": 3}\n', encoding="utf-8")
        stereo = self.root / "m3r" / "v-real" / "audio"
        stereo.mkdir(parents=True)
        (stereo / "v-001.wav").write_bytes(b"RIFF-stereo-assembled-at-2027")

    def sources(self) -> list[Source]:
        return [
            Source("backchannel", "m3r/turns-B-backchannel"),
            Source("roomtone", "m3r/roomtone/index.json"),
        ]

    def record(self, **overrides) -> dict:
        arguments = {
            "artifact_id": "v-real-test",
            "tool": "tools/assemble_dialogue.py",
            "captured_at": "2026-08-25",
            "why": "the 20:27 assembly",
            "root": str(self.root),
            "inputs": fingerprint_sources(self.sources(), root=self.root),
            "input_sources": self.sources(),
            "outputs": fingerprint_sources([Source("stereo", "m3r/v-real/audio")], root=self.root),
            "output_sources": [Source("stereo", "m3r/v-real/audio")],
        }
        arguments.update(overrides)
        return build_record(**arguments)


class SourceExpansionTests(TreeTestCase):
    """What a source name is allowed to mean, and what it must refuse to mean."""

    def test_a_directory_expands_to_its_whole_subtree_sorted(self) -> None:
        paths = expand_source(Source("backchannel", "m3r/turns-B-backchannel"), root=self.root)

        self.assertEqual(
            [path.name for path in paths],
            ["synthesis-log.jsonl", "v-001-t2-B.wav", "v-002-t2-B.wav", "v-003-t2-B.wav"],
        )

    def test_a_named_file_expands_to_itself(self) -> None:
        paths = expand_source(Source("roomtone", "m3r/roomtone/index.json"), root=self.root)

        self.assertEqual([path.name for path in paths], ["index.json"])

    def test_a_glob_narrows_the_subtree(self) -> None:
        paths = expand_source(
            Source("backchannel", "m3r/turns-B-backchannel/*.wav"), root=self.root
        )

        self.assertEqual(len(paths), 3)
        self.assertTrue(all(path.suffix == ".wav" for path in paths))

    def test_a_source_that_matches_nothing_is_refused(self) -> None:
        """A typo in a source path would otherwise verify green forever."""
        with self.assertRaises(ProvenanceError):
            expand_source(Source("backchannel", "m3r/turns-B-backchannnel"), root=self.root)

        (self.root / "m3r" / "empty").mkdir()
        with self.assertRaises(ProvenanceError):
            expand_source(Source("empty", "m3r/empty"), root=self.root)

    def test_dotted_files_are_skipped_by_default(self) -> None:
        """There is a real `.DS_Store` in `data/experiments/tsukuyomi_ojousama/m2/`.

        A verification that goes red because somebody opened a folder in the Finder is a
        verification people learn to ignore.
        """
        (self.root / "m3r" / "turns-B-backchannel" / ".DS_Store").write_bytes(b"finder")

        default = expand_source(Source("backchannel", "m3r/turns-B-backchannel"), root=self.root)
        explicit = expand_source(
            Source("backchannel", "m3r/turns-B-backchannel"), root=self.root, include_hidden=True
        )

        self.assertNotIn(".DS_Store", [path.name for path in default])
        self.assertIn(".DS_Store", [path.name for path in explicit])

    def test_a_source_that_climbs_out_of_the_root_is_refused(self) -> None:
        """Otherwise the record would name a path that means something else elsewhere."""
        outside = self.root.parent / "outside-the-root"
        outside.mkdir()
        (outside / "material.wav").write_bytes(b"RIFF-elsewhere")

        with self.assertRaises(ProvenanceError):
            expand_source(Source("elsewhere", "../outside-the-root"), root=self.root)

    def test_a_broken_symlink_is_refused_rather_than_skipped(self) -> None:
        """`is_file()` is False for a dangling link, so it would vanish from the listing."""
        (self.root / "m3r" / "turns-B-backchannel" / "v-004-t2-B.wav").symlink_to(
            self.root / "m3r" / "turns-B-backchannel" / "does-not-exist.wav"
        )

        with self.assertRaises(ProvenanceError):
            expand_source(Source("backchannel", "m3r/turns-B-backchannel"), root=self.root)

    def test_fingerprints_carry_the_role_the_source_was_named_with(self) -> None:
        entries = fingerprint_sources(self.sources(), root=self.root)

        roles = {entry["role"] for entry in entries}
        self.assertEqual(roles, {"backchannel", "roomtone"})
        self.assertEqual(len(entries), 5)
        for entry in entries:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["byte_size"], 0)


class TheAccidentTests(TreeTestCase):
    """2026-08-25, in miniature: the product is untouched, the material is not.

    This is the case every existing check in the repository passes and this module has to
    fail. The regenerated file is written at exactly the same byte length as the one it
    replaces, so nothing but the checksum can tell.
    """

    def test_the_artifact_checksum_still_matches_while_its_material_does_not(self) -> None:
        record = self.record()
        stereo = self.root / "m3r" / "v-real" / "audio" / "v-001.wav"
        stereo_checksum = sha256_file(stereo)

        regenerated = self.root / "m3r" / "turns-B-backchannel" / "v-001-t2-B.wav"
        original = regenerated.read_bytes()
        regenerated.write_bytes(b"RIFF-backchannel-X")  # 22:27, same length as the original
        self.assertEqual(len(original), regenerated.stat().st_size)

        result = verify_record(record, root=self.root)

        # The stereo file is exactly the bytes that were written down. That is what made
        # the accident invisible to a manifest checksum.
        self.assertEqual(sha256_file(stereo), stereo_checksum)
        self.assertEqual(result["groups"]["outputs"]["status"], "match")
        # And this is the check that was missing.
        self.assertEqual(result["status"], "mismatch")
        changed = result["groups"]["inputs"]["changed"]
        self.assertEqual(
            [entry["path"] for entry in changed], ["m3r/turns-B-backchannel/v-001-t2-B.wav"]
        )
        self.assertEqual(changed[0]["role"], "backchannel")
        self.assertIn("m3r/turns-B-backchannel/v-001-t2-B.wav", "\n".join(result["report"]))

    def test_an_untouched_tree_verifies_clean(self) -> None:
        result = verify_record(self.record(), root=self.root)

        self.assertEqual(result["status"], "match")
        self.assertEqual(result["groups"]["inputs"]["counts"]["unchanged"], 5)
        self.assertEqual(result["groups"]["outputs"]["counts"]["unchanged"], 1)

    def test_a_material_that_appeared_afterwards_is_found(self) -> None:
        """The 81st dialogue nobody meant to add.

        Only re-walking the sources can see this; re-hashing the stored file list cannot.
        """
        record = self.record()
        (self.root / "m3r" / "turns-B-backchannel" / "v-004-t2-B.wav").write_bytes(b"RIFF-late")

        result = verify_record(record, root=self.root)

        self.assertEqual(result["status"], "mismatch")
        self.assertTrue(result["groups"]["inputs"]["detects_additions"])
        self.assertEqual(
            [entry["path"] for entry in result["groups"]["inputs"]["added"]],
            ["m3r/turns-B-backchannel/v-004-t2-B.wav"],
        )

    def test_a_material_that_vanished_is_found(self) -> None:
        record = self.record()
        (self.root / "m3r" / "turns-B-backchannel" / "v-003-t2-B.wav").unlink()

        result = verify_record(record, root=self.root)

        self.assertEqual(
            [entry["path"] for entry in result["groups"]["inputs"]["removed"]],
            ["m3r/turns-B-backchannel/v-003-t2-B.wav"],
        )

    def test_a_rewritten_output_is_found_too(self) -> None:
        record = self.record()
        (self.root / "m3r" / "v-real" / "audio" / "v-001.wav").write_bytes(b"RIFF-overwritten")

        result = verify_record(record, root=self.root)

        self.assertEqual(result["groups"]["inputs"]["status"], "match")
        self.assertEqual(result["groups"]["outputs"]["status"], "mismatch")
        self.assertEqual(result["status"], "mismatch")

    def test_checking_inputs_only_skips_the_output_group(self) -> None:
        record = self.record()
        (self.root / "m3r" / "v-real" / "audio" / "v-001.wav").write_bytes(b"RIFF-overwritten")

        result = verify_record(record, root=self.root, groups=("inputs",))

        self.assertEqual(result["status"], "match")
        self.assertNotIn("outputs", result["groups"])

    def test_verifying_nothing_is_refused_rather_than_reported_clean(self) -> None:
        """`all()` over an empty result is True, which is the wrong answer here."""
        with self.assertRaises(ProvenanceError):
            verify_record(self.record(), root=self.root, groups=())

    def test_a_record_without_sources_says_it_cannot_see_additions(self) -> None:
        """Hand-written records exist. They may be weaker, but not silently weaker."""
        record = self.record()
        record["inputs"]["sources"] = []
        (self.root / "m3r" / "turns-B-backchannel" / "v-004-t2-B.wav").write_bytes(b"RIFF-late")

        result = verify_record(record, root=self.root)

        self.assertEqual(result["status"], "match")
        self.assertFalse(result["groups"]["inputs"]["detects_additions"])


class ContentNotTimestampTests(TreeTestCase):
    """mtime is what diagnosed the accident and the wrong thing to gate on.

    Copying a tree, restoring a backup and checking out a branch all rewrite every mtime
    while changing nothing; `touch` rewrites an mtime with no content change at all. A
    verification that cried wolf on `cp -R` would be turned off within a week.
    """

    def test_touching_every_file_changes_nothing(self) -> None:
        record = self.record()
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                path.touch()

        self.assertEqual(verify_record(record, root=self.root)["status"], "match")

    def test_a_copy_of_the_tree_verifies_against_the_same_record(self) -> None:
        """Root-relative paths plus content checksums, so a record survives a move."""
        record = self.record()
        with tempfile.TemporaryDirectory() as elsewhere:
            copy = Path(elsewhere) / "moved"
            shutil.copytree(self.root, copy)

            self.assertEqual(verify_record(record, root=copy)["status"], "match")

    def test_a_copy_with_one_file_rewritten_still_fails(self) -> None:
        record = self.record()
        with tempfile.TemporaryDirectory() as elsewhere:
            copy = Path(elsewhere) / "moved"
            shutil.copytree(self.root, copy)
            (copy / "m3r" / "roomtone" / "index.json").write_text(
                '{"segments": 4}\n', encoding="utf-8"
            )

            result = verify_record(record, root=copy)

            self.assertEqual(result["status"], "mismatch")
            self.assertEqual(
                [entry["path"] for entry in result["groups"]["inputs"]["changed"]],
                ["m3r/roomtone/index.json"],
            )


class RecordFileTests(TreeTestCase):
    """Reading a record back, and refusing one that cannot be trusted."""

    def test_a_record_round_trips_through_disk(self) -> None:
        path = self.root / "provenance.json"
        write_record(path, self.record())

        self.assertEqual(read_record(path)["artifact_id"], "v-real-test")

    def test_a_record_from_another_schema_version_is_refused(self) -> None:
        path = self.root / "provenance.json"
        record = self.record()
        record["schema_version"] = 99
        path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaises(ProvenanceError):
            read_record(path)

    def test_a_record_missing_its_root_is_refused(self) -> None:
        path = self.root / "provenance.json"
        record = self.record()
        record.pop("root")
        path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaises(ProvenanceError):
            read_record(path)

    def test_the_json_is_written_the_way_the_manifests_directory_writes_it(self) -> None:
        path = self.root / "provenance.json"
        write_record(path, self.record(why="日本語の理由", captured_at="2026-08-26"))
        text = path.read_text(encoding="utf-8")

        self.assertIn("日本語の理由", text)  # ensure_ascii=False, like the sidecars
        self.assertTrue(text.endswith("}\n"))


class RepositoryRootTests(unittest.TestCase):
    """A committed record has to work on the next checkout, so no absolute paths in it."""

    def test_a_root_inside_the_repository_is_recorded_relative(self) -> None:
        root = REPOSITORY_ROOT / "data" / "experiments"

        self.assertEqual(record_root_field(root), "data/experiments")

    def test_the_repository_root_itself_is_a_dot(self) -> None:
        self.assertEqual(record_root_field(REPOSITORY_ROOT), ".")

    def test_a_relative_root_resolves_back_to_the_same_directory(self) -> None:
        self.assertEqual(resolve_root("data/experiments"), REPOSITORY_ROOT / "data" / "experiments")
        self.assertEqual(resolve_root("."), REPOSITORY_ROOT)

    def test_the_module_finds_its_own_checkout(self) -> None:
        self.assertEqual(repository_root(), REPOSITORY_ROOT)

    def test_a_root_outside_the_repository_stays_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            self.assertEqual(
                record_root_field(Path(elsewhere)), Path(elsewhere).resolve().as_posix()
            )


class CommandLineTests(TreeTestCase):
    """The gate. `verify` exits non-zero, so a hook or a shell script can stop on it."""

    def _record_argv(self, out: Path) -> list[str]:
        return [
            "record",
            "--out",
            str(out),
            "--artifact-id",
            "v-real-test",
            "--tool",
            "tools/assemble_dialogue.py",
            "--captured-at",
            "2026-08-25",
            "--why",
            "the 20:27 assembly",
            "--root",
            str(self.root),
            "--input",
            "backchannel=m3r/turns-B-backchannel",
            "--input",
            "roomtone=m3r/roomtone/index.json",
            "--output",
            "stereo=m3r/v-real/audio",
        ]

    def test_record_then_verify_succeeds(self) -> None:
        out = self.root / "provenance.json"
        main(self._record_argv(out))

        main(["verify", "--record", str(out), "--quiet"])  # no SystemExit

        record = read_record(out)
        self.assertEqual(len(record["inputs"]["files"]), 5)
        self.assertEqual(len(record["outputs"]["files"]), 1)
        self.assertEqual(set(GROUPS), {"inputs", "outputs"})

    def test_verify_exits_one_when_a_material_was_rewritten(self) -> None:
        out = self.root / "provenance.json"
        main(self._record_argv(out))
        (self.root / "m3r" / "turns-B-backchannel" / "v-001-t2-B.wav").write_bytes(
            b"RIFF-backchannel-X"
        )

        with self.assertRaises(SystemExit) as raised:
            main(["verify", "--record", str(out)])

        self.assertEqual(raised.exception.code, 1)

    def test_verify_can_write_its_result_for_the_record(self) -> None:
        out = self.root / "provenance.json"
        result_path = self.root / "verification.json"
        main(self._record_argv(out))
        (self.root / "m3r" / "roomtone" / "index.json").write_text("{}\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            main(["verify", "--record", str(out), "--out", str(result_path)])

        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(
            [entry["path"] for entry in result["groups"]["inputs"]["changed"]],
            ["m3r/roomtone/index.json"],
        )

    def test_a_source_without_a_role_is_refused_at_the_command_line(self) -> None:
        """`--input m3r/roomtone` would record a nameless material."""
        with self.assertRaises(SystemExit):
            main(self._record_argv(self.root / "p.json")[:-2] + ["--input", "m3r/roomtone"])


if __name__ == "__main__":
    unittest.main()
