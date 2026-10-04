"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO

from .file_shape import MAX_PARSE_PEAK_MB, Placement, placement_of
from .spans import TextHit

AST_GREP = "ast-grep"
RIPGREP = "rg"
# `--no-config` keeps ripgrep from reading `RIPGREP_CONFIG_PATH`: over an untrusted repository, a
# config file could otherwise inject flags such as `--pre=<program>`, which runs an arbitrary
# program. It also keeps a personal rg config from changing what the index sees.
_RIPGREP_SAFE = (RIPGREP, "--no-config")
_NO_MATCHES_EXIT = 1
NEUTRAL_AST_GREP_CONFIG = "ruleDirs: []\n"
"""The smallest sgconfig ast-grep accepts. Passed with ``--config`` it replaces the discovery of the
analysed repository's own sgconfig.yml, which is customer content: its ``languageGlobs`` would change
what a file is parsed as, and its ``customLanguages`` makes ast-grep load a library the repository
names. Confirmed with ast-grep 0.45.1 that ``--config`` replaces discovery and is not merged with it."""
MAX_FILES_PER_COMMAND = 300
MAX_ARGUMENT_BYTES = 128 * 1024


class ToolFailedError(RuntimeError):
    """A command-line tool failed for a reason other than finding nothing."""


def run_command(
    arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None, stdin: str | None = None
) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found";
    ``stdin``, when given, is written to the command's standard input."""
    return command_output(arguments, cwd, no_match_exit=no_match_exit, stdin=stdin).decode()


def command_output(
    arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None, stdin: str | None = None
) -> bytes:
    """``run_command``'s output as the bytes the command wrote, for output that quotes file content."""
    completed = subprocess.run(
        list(arguments), cwd=cwd, input=None if stdin is None else stdin.encode(), capture_output=True
    )
    if completed.returncode not in (0, no_match_exit):
        detail = completed.stderr.decode(errors="replace").strip()[:300]
        raise ToolFailedError(f"{arguments[0]} exited {completed.returncode}: {detail}")
    return completed.stdout


@cache
def ast_grep_version() -> str:
    return run_command([AST_GREP, "--version"], Path.cwd()).strip()


def ast_grep_rules(
    rules_yaml: str,
    files: Sequence[str],
    cwd: Path,
    config: str | None = None,
    *,
    refused: dict[str, str],
) -> Iterator[dict]:
    """The matches of ``rules_yaml`` over ``files``, one at a time as ast-grep prints them, so no
    process's whole output is ever held. Every parse passes through here, placed by its estimated parse
    peak (``file_shape``): files within ``MAX_PARSE_PEAK_MB`` are parsed side by side; a file over it
    but within ``single_parse_limit_mb()`` is parsed alone, one at a time; a file over that
    is never handed to ast-grep and is added to ``refused`` with its reason when the iteration starts,
    so read ``refused`` after the matches. ast-grep always runs with a JVN-owned sgconfig: ``config``,
    when given, is sgconfig YAML text (a ``languageGlobs`` remapping, say), otherwise
    ``NEUTRAL_AST_GREP_CONFIG``. It is written to a temporary file outside every repository and passed
    with ``--config``, so the repository being analysed never configures the parser."""
    placed = _placed(files, cwd, single_parse_limit_mb())
    refused.update(placed.refused)
    if not placed.side_by_side and not placed.alone:
        return
    with ExitStack() as resources:
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="jev-navigator-sgconfig-"))
        path = Path(directory) / "sgconfig.yml"
        path.write_text(NEUTRAL_AST_GREP_CONFIG if config is None else config)
        command = [AST_GREP, "scan", "--inline-rules", rules_yaml, "--config", str(path)]
        for chunk in file_chunks(placed.side_by_side):
            yield from _json_lines([*command, "--json=stream", "--", *chunk], cwd)
        for file in placed.alone:
            yield from _json_lines([*command, "--threads", "1", "--json=stream", "--", file], cwd)


def single_parse_limit_mb() -> float:
    """The most one file may take when parsed alone. It is the side-by-side bound, so no file is parsed
    alone, until JVN's memory settings provide a single-file limit."""
    return MAX_PARSE_PEAK_MB


@dataclass
class _Placed:
    side_by_side: list[str] = field(default_factory=list)
    alone: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)


def _placed(files: Sequence[str], cwd: Path, single_parse_limit: float) -> _Placed:
    placed = _Placed()
    for file in files:
        try:
            placement, reason = placement_of(cwd, file, single_parse_limit)
        except OSError as error:
            placement, reason = Placement.REFUSED, f"could not be measured: {type(error).__name__}: {error}"
        if placement is Placement.SIDE_BY_SIDE:
            placed.side_by_side.append(file)
        elif placement is Placement.ALONE:
            placed.alone.append(file)
        else:
            placed.refused[file] = reason
    return placed


def file_chunks(files: Sequence[str], *, bytes_only: bool = False) -> Iterator[Sequence[str]]:
    """``files`` in order, split so no command gets more than ``MAX_ARGUMENT_BYTES`` of paths, and,
    unless ``bytes_only``, no more than ``MAX_FILES_PER_COMMAND`` of them, which bounds what one
    parser process holds. A text search holds no file, so only the argument limit applies to it."""
    most_files = None if bytes_only else MAX_FILES_PER_COMMAND
    start, size = 0, 0
    for position, file in enumerate(files):
        length = len(file.encode()) + 1
        too_many = most_files is not None and position - start >= most_files
        full = too_many or size + length > MAX_ARGUMENT_BYTES
        if position > start and full:
            yield files[start:position]
            start, size = position, 0
        size += length
    if start < len(files):
        yield files[start:]


def _json_lines(arguments: Sequence[str], cwd: Path) -> Iterator[dict]:
    """Each line the command prints, parsed as JSON while it runs. stderr goes to a file, so a full
    stderr pipe cannot stall the command; the process is killed if the reader stops early. A line
    that is no JSON (the process died partway through it) fails with the process's exit code and
    stderr, which say why it stopped."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(list(arguments), cwd=cwd, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in process.stdout:
                if line.strip():
                    yield _json_object(line, process, errors, arguments[0])
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
            returncode = process.wait()
        if returncode not in (0, _NO_MATCHES_EXIT):
            raise _tool_failed(arguments[0], returncode, errors)


def _json_object(line: str, process: subprocess.Popen, errors: IO[bytes], tool: str) -> dict:
    try:
        return json.loads(line)
    except ValueError as malformed:
        process.kill()
        raise _tool_failed(tool, process.wait(), errors) from malformed


def _tool_failed(tool: str, returncode: int, errors: IO[bytes]) -> ToolFailedError:
    errors.seek(0)
    return ToolFailedError(f"{tool} exited {returncode}: {errors.read().decode(errors='replace').strip()}")


def ripgrep_fixed(
    text: str, files: Sequence[str], cwd: Path, max_hits: int, context_bytes: int, *, whole_word: bool = False
) -> list[TextHit]:
    """The lines holding ``text``, at most ``max_hits`` per file, each as the bytes around one hit:
    up to ``context_bytes`` before and after, so a one-line bundle costs no more than a short line.
    ``whole_word`` keeps only hits no word character touches. The match runs on to the end of the
    line, so each line matches once, and ``--replace`` prints only its window; ripgrep's JSON would
    carry the whole line."""
    pattern = _hit_window(text, context_bytes, whole_word)
    command = [*_RIPGREP_SAFE, "--only-matching", "--line-number", "--with-filename", "--null"]
    command += ["--max-count", str(max_hits), "--replace", "$window", "--regexp", pattern, "--"]
    hits: dict[tuple[str, int], TextHit] = {}
    for chunk in file_chunks(files, bytes_only=True):
        for hit in _windows(command_output([*command, *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)):
            hits.setdefault((hit.file, hit.line), hit)
    return list(hits.values())


def _hit_window(text: str, context_bytes: int, whole_word: bool) -> str:
    """A regular expression capturing, as ``window``, ``text`` with up to ``context_bytes`` of any
    bytes on either side, then matching the rest of the line. The text's characters other than
    letters, digits and underscores are written as code points, so no character of it is read as
    syntax."""
    context = f"(?-u:.){{0,{context_bytes}}}"
    literal = "".join(char if char.isalnum() or char == "_" else f"\\x{{{ord(char):x}}}" for char in text)
    if whole_word:
        literal = rf"(?:^|\W){literal}(?:\W|$)"
    return f"(?P<window>{context}{literal}{context})(?-u:.)*"


def _windows(output: bytes) -> Iterator[TextHit]:
    """The hits of ripgrep's ``--null`` printer, ``path NUL line:window`` per line of output. A window
    holds no newline, and a path ends at its NUL, so a newline in a path cannot split a record.
    Bytes that are not UTF-8 are decoded the way the index reads files, with invalid bytes replaced."""
    position = 0
    while position < len(output):
        path_end = output.index(b"\0", position)
        number_end = output.index(b":", path_end)
        window_end = output.find(b"\n", number_end)
        window_end = len(output) if window_end < 0 else window_end
        path = output[position:path_end].decode(errors="replace").removeprefix("./")
        window = output[number_end + 1 : window_end].decode(errors="replace").rstrip("\r")
        yield TextHit(path, int(output[path_end + 1 : number_end]), window)
        position = window_end + 1


def listed_files(cwd: Path, prefixes: Sequence[str] = ()) -> tuple[str, ...]:
    """Regular, non-symlink files owned by this working directory, including hidden paths.

    A Git worktree uses its tracked and untracked, non-ignored inventory, which naturally excludes
    nested repositories and managed worktrees. A non-Git directory uses ripgrep's ignore policy.
    """
    if inside_git_worktree(cwd):
        output = git(["ls-files", "-z", "-c", "-o", "--exclude-standard", "--", *prefixes], cwd)
    else:
        output = run_command(
            [
                *_RIPGREP_SAFE,
                "--files",
                "--hidden",
                "--null",
                "--glob",
                "!.git",
                "--glob",
                "!.git/**",
                *prefixes,
            ],
            cwd,
        )
    files = []
    for raw in output.split("\0"):
        path = raw.removeprefix("./")
        candidate = cwd / path
        if path and candidate.is_file() and not candidate.is_symlink():
            files.append(path)
    return tuple(sorted(dict.fromkeys(files)))


def inside_git_worktree(cwd: Path) -> bool:
    try:
        return git(["rev-parse", "--is-inside-work-tree"], cwd).strip() == "true"
    except ToolFailedError:
        return False


def git(arguments: Sequence[str], cwd: Path, *, stdin: str | None = None) -> str:
    return run_command(["git", *arguments], cwd, stdin=stdin)


def export_blobs(repository: Path, blobs: Mapping[str, str], destination: Path) -> None:
    """Writes each blob, keyed by its path, into ``destination``, all read with one
    ``git cat-file --batch``. Blobs are asked for by object id, so any byte in a path is safe. A path
    that would leave ``destination`` and an object git does not have both raise ``ToolFailedError``."""
    if not blobs:
        return
    for path, content in zip(blobs, _cat_file_batch(repository, blobs.values()), strict=True):
        _write_inside(destination, path, content)


def git_blob(repository: Path, object_id: str) -> bytes:
    """The bytes of one blob, asked for by object id; an object git does not have raises
    ``ToolFailedError``."""
    (content,) = _cat_file_batch(repository, [object_id])
    return content


def _cat_file_batch(repository: Path, object_ids: Iterable[str]) -> list[bytes]:
    requests = "".join(f"{object_id}\n" for object_id in object_ids).encode()
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=requests,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ToolFailedError(
            f"git cat-file exited {completed.returncode}: {completed.stderr.decode()[:300]}"
        )
    return _batch_contents(completed.stdout)


def _batch_contents(output: bytes) -> list[bytes]:
    """The object contents in ``git cat-file --batch`` output, in request order."""
    contents = []
    position = 0
    while position < len(output):
        header_end = output.index(b"\n", position)
        object_id, _, details = output[position:header_end].partition(b" ")
        if details == b"missing":
            raise ToolFailedError(f"git has no object {object_id.decode()}")
        start = header_end + 1
        end = start + int(details.split()[1])
        contents.append(output[start:end])
        position = end + 1
    return contents


def _write_inside(destination: Path, path: str, content: bytes) -> None:
    root = destination.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ToolFailedError(f"{path!r} would be written outside {root}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
