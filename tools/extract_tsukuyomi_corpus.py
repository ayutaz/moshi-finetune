from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

CORPUS_NAME = re.compile(r"VOICEACTRESS100_(\d{3})\.wav$")
DOCUMENT_NAMES = {
    "01 台本について.txt": "script-license.txt",
    "02 ライセンスについて.txt": "license.txt",
    "03 つくよみちゃんコーパスの利用規約.txt": "tsukuyomi-terms.txt",
    "01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt": "corpus-transcript.txt",
    "02 補足つき台本（つくよみちゃんコーパス版）.txt": "corpus-transcript-annotated.txt",
    "01 同梱している台本について.txt": "transcript-notes.txt",
    "02 読み仮名・アクセントについて.txt": "pronunciation-accent-notes.txt",
    "03 アクセントについての調査結果.txt": "accent-research-notes.txt",
    "01 収録・編集方法の説明.txt": "recording-editing-notes.txt",
    "ReadMe.txt": "readme.txt",
}


class CorpusExtractionError(ValueError):
    """Raised when the corpus archive does not match the expected 100-file layout."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_member_name(member: zipfile.ZipInfo) -> str:
    if member.flag_bits & 0x800:
        return member.filename
    try:
        return member.filename.encode("cp437").decode("cp932")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return member.filename


def _write_member(
    zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path
) -> dict[str, Any]:
    expected_crc32 = f"{member.CRC:08x}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as file_handle:
            crc32 = 0
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                crc32 = zlib.crc32(chunk, crc32)
        if f"{crc32 & 0xFFFFFFFF:08x}" != expected_crc32:
            raise CorpusExtractionError(
                f"existing output differs from archive member: {destination}"
            )
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            with zip_file.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()
    return {
        "path": destination.name,
        "byte_size": destination.stat().st_size,
        "sha256": _sha256(destination),
        "zip_crc32": expected_crc32,
    }


def extract_corpus(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Extract the archive's first complete audio variant under canonical ASCII names."""
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    candidates: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    document_members: dict[str, zipfile.ZipInfo] = {}

    with zipfile.ZipFile(archive_path) as zip_file:
        for member in zip_file.infolist():
            basename = PurePosixPath(member.filename).name
            match = CORPUS_NAME.fullmatch(basename)
            if match and 1 <= int(match.group(1)) <= 100:
                candidates[basename].append(member)
            decoded_basename = PurePosixPath(_decoded_member_name(member)).name
            if decoded_basename in DOCUMENT_NAMES:
                if decoded_basename in document_members:
                    raise CorpusExtractionError(
                        f"duplicate documentation member: {decoded_basename}"
                    )
                document_members[decoded_basename] = member

        expected_names = {f"VOICEACTRESS100_{index:03d}.wav" for index in range(1, 101)}
        missing_names = sorted(expected_names - candidates.keys())
        if missing_names:
            raise CorpusExtractionError(
                f"archive must contain all 100 utterances; missing {len(missing_names)}"
            )
        variant_counts = {len(candidates[name]) for name in expected_names}
        if len(variant_counts) != 1:
            raise CorpusExtractionError(
                f"inconsistent variant counts across the 100 utterances: {sorted(variant_counts)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        for basename in sorted(expected_names):
            selected = candidates[basename][0]
            destination = output_dir / basename
            extracted_files.append(_write_member(zip_file, selected, destination))

        extracted_documents = []
        for decoded_basename, canonical_name in sorted(DOCUMENT_NAMES.items()):
            member = document_members.get(decoded_basename)
            if member is None:
                continue
            destination = output_dir / "documentation" / canonical_name
            document_report = _write_member(zip_file, member, destination)
            document_report["source_basename"] = decoded_basename
            extracted_documents.append(document_report)

    return {
        "schema_version": 1,
        "archive_path": archive_path.name,
        "archive_sha256": _sha256(archive_path),
        "selection_rule": "first archive member for each canonical utterance name",
        "selected_variant_position": 0,
        "variant_count": variant_counts.pop(),
        "selected_file_count": len(extracted_files),
        "files": extracted_files,
        "documentation_file_count": len(extracted_documents),
        "documentation": extracted_documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely extract Tsukuyomi Corpus Vol.1")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = extract_corpus(args.archive, args.output_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "selected_file_count": report["selected_file_count"],
                "variant_count": report["variant_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
