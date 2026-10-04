"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git. Every process
starts through ``memory_limit.started``, so JVN's memory allowance and ceiling cover all of them."""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO

from .. import memory_limit
from .file_shape import Placement, placement_of
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
    arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None, timeout: float | None = None
) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found".
    A command still running after ``timeout`` seconds is stopped and raises ``subprocess.TimeoutExpired``."""
    with memory_limit.started(
        arguments, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        output, errors = process.communicate(timeout=timeout)
    if process.returncode not in (0, no_match_exit):
        raise ToolFailedError(f"{arguments[0]} exited {process.returncode}: {errors.strip()[:300]}")
    return output


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
    peak (``file_shape``) against JVN's memory settings (``memory_limit.MemoryLimit``): files within
    ``MAX_PARSE_PEAK_MB`` are parsed side by side, as many at once as the allowance affords; a file over
    it but within ``single_parse_mb`` is parsed alone, on one thread, one at a time, after them; a file
    over that is never handed to ast-grep and is added to ``refused`` with its reason when the iteration
    starts, so read ``refused`` after the matches. One scan runs at a time in a process
    (``memory_limit.parsing``). ast-grep always runs with a JVN-owned sgconfig: ``config``, when given,
    is sgconfig YAML text (a ``languageGlobs`` remapping, say), otherwise ``NEUTRAL_AST_GREP_CONFIG``. It
    is written to a temporary file outside every repository and passed with ``--config``, so the
    repository being analysed never configures the parser."""
    limit = memory_limit.process_guard().limit
    placed = _placed(files, cwd, limit.single_parse_mb)
    refused.update(placed.refused)
    if not placed.side_by_side and not placed.alone:
        return
    with ExitStack() as resources:
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="jev-navigator-sgconfig-"))
        path = Path(directory) / "sgconfig.yml"
        path.write_text(NEUTRAL_AST_GREP_CONFIG if config is None else config)
        resources.enter_context(memory_limit.parsing())
        command = [AST_GREP, "scan", "--inline-rules", rules_yaml, "--config", str(path)]
        side_by_side = [*command, "--threads", str(limit.parse_threads)]
        for chunk in file_chunks(placed.side_by_side):
            yield from _json_lines([*side_by_side, "--json=stream", "--", *chunk], cwd)
        for file in placed.alone:
            yield from _json_lines([*command, "--threads", "1", "--json=stream", "--", file], cwd)


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


def file_chunks(files: Sequence[str]) -> Iterator[Sequence[str]]:
    """``files`` in order, split so no command gets more than ``MAX_FILES_PER_COMMAND`` paths or
    ``MAX_ARGUMENT_BYTES`` of them, whatever the size of the scope."""
    start, size = 0, 0
    for position, file in enumerate(files):
        length = len(file.encode()) + 1
        full = position - start >= MAX_FILES_PER_COMMAND or size + length > MAX_ARGUMENT_BYTES
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
        with memory_limit.started(
            arguments, cwd=cwd, stdout=subprocess.PIPE, stderr=errors, text=True
        ) as process:
            for line in process.stdout:
                if line.strip():
                    yield _json_object(line, process, errors, arguments[0])
        if process.returncode not in (0, _NO_MATCHES_EXIT):
            raise _tool_failed(arguments[0], process.returncode, errors)


def _json_object(line: str, process: subprocess.Popen, errors: IO[bytes], tool: str) -> dict:
    try:
        return json.loads(line)
    except ValueError as malformed:
        process.kill()
        raise _tool_failed(tool, process.wait(), errors) from malformed


def _tool_failed(tool: str, returncode: int, errors: IO[bytes]) -> ToolFailedError:
    errors.seek(0)
    return ToolFailedError(f"{tool} exited {returncode}: {errors.read().decode(errors='replace').strip()}")


def ripgrep_fixed(text: str, files: Sequence[str], cwd: Path, max_hits: int) -> list[TextHit]:
    """The lines holding ``text``. JSON events are split at newlines only, since a line of code may
    hold a Unicode line separator that ``str.splitlines`` would split."""
    command = [*_RIPGREP_SAFE, "--json", "--fixed-strings", "--max-count", str(max_hits), "--", text]
    hits = []
    for chunk in file_chunks(files):
        output = run_command([*command, *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)
        events = (json.loads(line) for line in output.split("\n") if line.strip())
        hits += [_text_hit(event["data"]) for event in events if event.get("type") == "match"]
    return hits


def ripgrep_files(texts: str | Sequence[str], files: Sequence[str], cwd: Path) -> tuple[str, ...]:
    """Every supplied file containing any of the exact ``texts``, without a result-count cutoff. The
    texts go to ripgrep in a pattern file, one per line, so their number never meets the argument
    limit."""
    patterns = [texts] if isinstance(texts, str) else list(texts)
    if not files or not patterns:
        return ()
    if any("\n" in pattern for pattern in patterns):
        raise ValueError("a text searched for by file cannot hold a line break")
    with tempfile.NamedTemporaryFile("w", prefix="jev-navigator-patterns-", suffix=".txt") as pattern_file:
        pattern_file.write("".join(f"{pattern}\n" for pattern in patterns))
        pattern_file.flush()
        command = [
            *_RIPGREP_SAFE,
            "--files-with-matches",
            "--null",
            "--fixed-strings",
            "-f",
            pattern_file.name,
        ]
        found: list[str] = []
        for chunk in file_chunks(files):
            output = run_command([*command, "--", *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)
            found += [path.removeprefix("./") for path in output.split("\0") if path]
    return tuple(found)


def listed_files(cwd: Path, prefixes: Sequence[str] = ()) -> tuple[str, ...]:
    """Regular, non-symlink files owned by this working directory, including hidden paths.

    A Git worktree uses its tracked and untracked, non-ignored inventory, which naturally excludes
    nested repositories and managed worktrees. A non-Git directory uses ripgrep's ignore policy.
    """
    try:
        inside_git = git(["rev-parse", "--is-inside-work-tree"], cwd).strip() == "true"
    except ToolFailedError:
        inside_git = False
    if inside_git:
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


def _text_hit(match: dict) -> TextHit:
    return TextHit(_decoded(match["path"]), match["line_number"], _decoded(match["lines"]).rstrip("\r\n"))


def _decoded(field: dict) -> str:
    """ripgrep reports a path or line that is not valid UTF-8 as base64 ``bytes`` instead of ``text``;
    it is decoded the way the index reads files, with invalid bytes replaced."""
    if "text" in field:
        return field["text"]
    return base64.b64decode(field["bytes"]).decode("utf-8", errors="replace")


def git(arguments: Sequence[str], cwd: Path, *, timeout: float | None = None) -> str:
    return run_command(["git", *arguments], cwd, timeout=timeout)


def export_blobs(repository: Path, blobs: Mapping[str, str], destination: Path) -> None:
    """Writes each blob, keyed by its path, into ``destination``, all read with one
    ``git cat-file --batch``. Blobs are asked for by object id, so any byte in a path is safe. A path
    that would leave ``destination`` and an object git does not have both raise ``ToolFailedError``."""
    if not blobs:
        return
    requests = "".join(f"{object_id}\n" for object_id in blobs.values()).encode()
    with memory_limit.started(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        output, errors = process.communicate(requests)
    if process.returncode != 0:
        raise ToolFailedError(f"git cat-file exited {process.returncode}: {errors.decode()[:300]}")
    for path, content in zip(blobs, _batch_contents(output), strict=True):
        _write_inside(destination, path, content)


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
