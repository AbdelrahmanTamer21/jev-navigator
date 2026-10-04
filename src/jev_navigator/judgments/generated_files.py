"""Jev's generated-file judgment (G1): one Noul per file the scope could not decide in code.

``scope.resolve_scope`` decides a file whose shape flags it wherever code can (a linguist attribute, a
generated header, a vendored or output folder) and hands the rest on in
``ResolvedScope.awaiting_generated_judgment``. Each reaches Jev as one entry: its path, its measured
facts, up to ``MAX_IMPORTERS`` files that import it with their true count, and two excerpts. The
line the answer draws is André's (04.10.2026, about 13:00): generated means no person edits the file
as source. A file the secret scan refuses is never sent; it is named as not judged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index import tools
from ..index.code_index import CodeIndex
from ..index.file_shape import FileShape
from .judge import CheckResult, Judge
from .questions import Check, Criterion
from .secrets import SecretInRequestError, mask_request, refuse_if_secret

FILES = "files"
EXCERPT_CHARS = 2_000
MAX_IMPORTERS = 10
NOT_JUDGED_SECRET = "not judged: the secret scan refused its excerpts"
_PACKAGE_ENTRY_STEMS = frozenset({"index", "__init__"})

GENERATED_FILE = Check(
    name="generated_file",
    instructions="Look only at `{item}`. Is this file generated, meaning no person edits it as source?",
    yes=Criterion(
        "A tool writes or re-creates the file: a bundle, minified code, a code generator's output, or a"
        " recording, extract or dump that people re-create from data.",
        not_for="A long data table, inline SVG or image string that a person keeps up to date by hand.",
    ),
    no=Criterion(
        "A person writes or edits the file as source, even when it is long or holds data.",
        not_for="A recording, extract or dump that people re-create from data instead of editing it.",
    ),
)


@dataclass(frozen=True)
class GeneratedJudgments:
    """Jev's answer for each judged file, and the reason each other file was not judged."""

    judged: Mapping[str, CheckResult]
    not_judged: Mapping[str, str]


def judge_generated_files(
    judge: Judge, index: CodeIndex, awaiting: Mapping[str, FileShape]
) -> GeneratedJudgments:
    entries = {path: generated_file_entry(index, path, awaiting[path]) for path in sorted(awaiting)}
    refused = {path for path, entry in entries.items() if _refused_by_secret_scan(judge, entry)}
    sendable = [path for path in entries if path not in refused]
    results = judge.check_each(GENERATED_FILE, [entries[path] for path in sendable], list_name=FILES)
    return GeneratedJudgments(
        {result.item["file"]: result for result in results}, dict.fromkeys(sorted(refused), NOT_JUDGED_SECRET)
    )


def generated_file_entry(index: CodeIndex, path: str, shape: FileShape) -> dict:
    """One file as Jev sees it: path, measured facts, importers and the two excerpts."""
    importers = importers_of(index, path)
    text = "\n".join(index.lines(path))
    return {
        "file": path,
        "size_bytes": shape.size_bytes,
        "line_count": shape.line_count,
        "longest_line": shape.longest_line,
        "chars_per_line": round(shape.chars_per_line, 1),
        "importers": list(importers[:MAX_IMPORTERS]),
        "importer_count": len(importers),
        **_excerpts(text),
    }


def importers_of(index: CodeIndex, path: str) -> tuple[str, ...]:
    """The scope files whose imports resolve to ``path``: one ripgrep for its import stem narrows the
    candidates, and the index's own text import reader decides, so nothing is parsed."""
    others = [file for file in index.files if file != path]
    candidates = tools.ripgrep_files(_import_stem(path), others, index.root)
    return tuple(sorted(file for file in candidates if path in index.imports(file)))


def _import_stem(path: str) -> str:
    """The name an import of ``path`` spells: the file's stem, or its folder's name for a package entry."""
    pure = PurePosixPath(path)
    return pure.parent.name if pure.stem in _PACKAGE_ENTRY_STEMS else pure.stem


def _excerpts(text: str) -> dict[str, str]:
    if len(text) <= 2 * EXCERPT_CHARS:
        return {"opening": text}
    middle = (len(text) - EXCERPT_CHARS) // 2
    return {"opening": text[:EXCERPT_CHARS], "middle": text[middle : middle + EXCERPT_CHARS]}


def _refused_by_secret_scan(judge: Judge, entry: Mapping) -> bool:
    """Whether the judge's own mask-then-scan would refuse a request holding only this entry."""
    state: Mapping = {FILES: [entry]}
    masked: frozenset[str] = frozenset()
    if judge.masker is not None:
        state, _, masked = mask_request(state, {}, judge.masker)
    try:
        refuse_if_secret(state, {}, judge.scanner, masked)
    except SecretInRequestError:
        return True
    return False
