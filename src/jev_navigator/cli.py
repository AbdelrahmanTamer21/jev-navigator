"""Command-line evidence pack for one live ``find_code`` search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from .adapters.typesafe import TypeSafeJevClient
from .directives.entry import EntrySelection, choose_initial_candidates
from .directives.find_code import FindResult, SearchBudget, Visit, find_code
from .directives.places import Place, place_for_line
from .index.code_index import CodeIndex
from .judgments.client import JevClient
from .judgments.judge import Judge
from .judgments.store import JsonlAnswerStore
from .judgments.thresholds import Thresholds
from .progress import ProgressJournal, TerminalProgress

SCHEMA_VERSION = "jev-navigator.evidence-pack/v1"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "find":
        raise AssertionError(f"unhandled command: {args.command}")
    budget = SearchBudget(
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        max_calls=args.max_calls,
        beam_width=args.beam_width,
        neighbours_per_kind=args.neighbours_per_kind,
        preview_lines=args.preview_lines,
        max_slice_chars=args.max_slice_chars,
        max_line_chars=args.max_line_chars,
    )
    repository = Path(args.repo).resolve()
    output = Path(args.out).expanduser() if args.out else _default_output(repository)
    client: TypeSafeJevClient | None = None
    try:
        _load_typesafe_environment(os.environ)
        client = TypeSafeJevClient()
        manifest = create_evidence_pack(
            repository,
            tuple(args.prefix),
            args.target,
            tuple(args.start),
            output,
            budget,
            client,
            thresholds=Thresholds.from_env(),
            verbose=args.verbose,
        )
    except Exception as error:
        print(f"jvn find: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("jvn find: cancelled", file=sys.stderr)
        return 130
    finally:
        if client is not None:
            client.close()
    print(f"evidence pack: {output.resolve()}")
    search_outcome = manifest["search"]["outcome"]
    print(f"outcome: {search_outcome} ({manifest['search']['calls']} live calls)")
    return 130 if search_outcome == "cancelled" else 0


def create_evidence_pack(
    repository: Path,
    prefixes: tuple[str, ...],
    target: str,
    starts: tuple[str, ...],
    output: Path,
    budget: SearchBudget,
    client: JevClient,
    *,
    thresholds: Thresholds | None = None,
    verbose: bool = False,
    fact_cache_dir: Path | None = None,
) -> dict:
    """Run the real index/search owners and persist their reviewable evidence."""
    repository = repository.resolve()
    output = output.resolve()
    _validate_budget(budget)
    _prepare_output(output)
    journal_path = output / "journal.jsonl"
    journal_path.touch()
    progress = TerminalProgress(journal_path, verbose=verbose)
    journal = ProgressJournal(journal_path, progress)
    progress.start()
    outcome = "failed"
    try:
        progress.phase("indexing files")
        index = CodeIndex.from_directory(
            repository,
            prefixes=prefixes,
            exclude_paths=(output, Path.cwd() / "jvn-results"),
            scan_observer=progress.scan,
            fact_cache_dir=fact_cache_dir,
        )
        if warning := _scope_warning(len(index.files)):
            print(warning, file=sys.stderr)
        thresholds = thresholds or Thresholds()
        judge = Judge(
            client,
            thresholds=thresholds,
            max_calls=budget.max_calls,
            journal=journal,
            store=JsonlAnswerStore(output / "answers.jsonl"),
        )
        selection: EntrySelection | None = None
        if starts:
            start_places = [_parse_start(index, start) for start in starts]
            initial_candidates: tuple[tuple[Place, float], ...] = ()
        else:
            progress.phase("choosing an entry point")
            start_places = []
            selection = choose_initial_candidates(index, judge, target)
            initial_candidates = tuple(
                (
                    candidate.place,
                    candidate.selection_probability if candidate.selection_probability is not None else 0.0,
                )
                for candidate in selection.candidates
            )
        progress.phase("navigating code")
        started = monotonic()
        result = find_code(
            index,
            judge,
            target,
            start_places,
            budget=budget,
            commit=None,
            initial_candidates=initial_candidates,
        )
        duration_seconds = monotonic() - started
        progress.phase("writing evidence pack")
        manifest = _manifest(
            repository,
            prefixes,
            target,
            starts,
            budget,
            thresholds,
            index,
            result,
            requested_model=getattr(client, "model", "unknown"),
            served_model=judge.served_model,
            input_tokens=judge.input_tokens,
            duration_seconds=duration_seconds,
            total_calls=judge.calls,
            entry_selection=selection,
        )
        _write_json(output / "manifest.json", manifest)
        (output / "report.md").write_text(_report(manifest))
        outcome = str(result.outcome)
        return manifest
    except KeyboardInterrupt:
        outcome = "cancelled"
        raise
    finally:
        journal.record_terminal(outcome)
        progress.close(outcome)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jvn",
        description="Navigate code with reviewable Jev judgments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser(
        "find",
        help="find semantically described code and write a versioned evidence pack",
        description="Run one live find_code search and write a versioned evidence pack.",
    )
    find.add_argument("target", help="Semantic description of the code to find")
    find.add_argument("--repo", default=".", help="Git repository to inspect (default: current directory)")
    find.add_argument(
        "--prefix", action="append", default=[], help="Optional tracked file or directory scope; repeatable"
    )
    find.add_argument(
        "--start",
        action="append",
        default=[],
        metavar="PATH:LINE",
        help="Known entry or caller line; repeatable. Without one, jvn chooses a narrow entry point.",
    )
    find.add_argument(
        "--out",
        help="New or empty output directory (default: a unique run under ./jvn-results)",
    )
    defaults = SearchBudget()
    find.add_argument("--max-depth", type=int, default=defaults.max_depth)
    find.add_argument("--max-steps", type=int, default=defaults.max_steps)
    find.add_argument("--max-calls", type=int, default=defaults.max_calls)
    find.add_argument("--beam-width", type=int, default=defaults.beam_width)
    find.add_argument("--neighbours-per-kind", type=int, default=defaults.neighbours_per_kind)
    find.add_argument("--preview-lines", type=int, default=defaults.preview_lines)
    find.add_argument("--max-slice-chars", type=int, default=defaults.max_slice_chars)
    find.add_argument("--max-line-chars", type=int, default=defaults.max_line_chars)
    find.add_argument(
        "--verbose",
        action="store_true",
        help="print expanded masked requests as they are sent",
    )
    return parser


def _validate_budget(budget: SearchBudget) -> None:
    non_negative = ("max_depth", "max_steps", "max_calls", "neighbours_per_kind", "preview_lines")
    positive = ("beam_width", "max_slice_chars", "max_line_chars")
    invalid = [name for name in non_negative if (value := getattr(budget, name)) is not None and value < 0]
    invalid += [name for name in positive if getattr(budget, name) < 1]
    if invalid:
        raise ValueError(f"invalid search budget fields: {', '.join(invalid)}")


def _prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _default_output(repository: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path.cwd() / "jvn-results" / f"{repository.name}-{stamp}"


def _scope_warning(file_count: int) -> str | None:
    if file_count <= 20_000:
        return None
    return f"jvn: large scope contains {file_count:,} tracked files; indexing may take longer"


def _load_typesafe_environment(
    environment: MutableMapping[str, str],
    path: Path | None = None,
) -> None:
    """Load official TypeSafe SDK settings, with each process value taking precedence."""
    from dotenv import dotenv_values

    path = path or Path.home() / ".config/jvn/env"
    configured = dotenv_values(path) if path.is_file() else {}
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"):
        if environment.get(name, "").strip():
            continue
        value = configured.get(name)
        if isinstance(value, str) and value.strip():
            environment[name] = value
    if not environment.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError("TYPESAFE_API_KEY is unset and ~/.config/jvn/env does not provide it")


def _parse_start(index: CodeIndex, value: str) -> Place:
    path, separator, raw_line = value.rpartition(":")
    if not separator or not path:
        raise ValueError(f"start must be PATH:LINE, got {value!r}")
    try:
        line = int(raw_line)
    except ValueError as error:
        raise ValueError(f"start line must be an integer, got {value!r}") from error
    if line < 1 or line > len(index.lines(path)):
        raise ValueError(f"start line is outside {path}: {line}")
    return place_for_line(index, path, line, "caller-provided start")


def _manifest(
    repository: Path,
    prefixes: tuple[str, ...],
    target: str,
    starts: tuple[str, ...],
    budget: SearchBudget,
    thresholds: Thresholds,
    index: CodeIndex,
    result: FindResult,
    *,
    requested_model: str,
    served_model: str | None,
    input_tokens: int,
    duration_seconds: float,
    total_calls: int,
    entry_selection: EntrySelection | None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "navigator": _navigator_provenance(),
        "source": {
            "repository": str(repository),
            "revision": index.commit,
            "prefixes": list(prefixes),
            "tracked_files": len(index.files),
        },
        "target": target,
        "requested_starts": list(starts),
        "entry_selection": entry_selection.to_json() if entry_selection else None,
        "budget": asdict(budget),
        "thresholds": thresholds.as_dict(),
        "provider": {
            "requested_model": requested_model,
            "served_model": served_model,
            "input_tokens": input_tokens,
        },
        "search": {
            "outcome": result.outcome,
            "steps": result.steps,
            "calls": total_calls,
            "entry_calls": total_calls - result.calls,
            "navigation_calls": result.calls,
            "duration_seconds": round(duration_seconds, 3),
            "moves": list(result.moves),
            "found": [_visit(visit) for visit in result.found],
            "starts": [_visit(visit) for visit in result.starts],
            "searched": [_visit(visit) for visit in result.searched],
            "unsure": [_visit(visit) for visit in result.unsure],
            "not_inspected": [
                {
                    "place": entry.place_key,
                    "signature": entry.signature,
                    "reason": entry.reason,
                    "priority": entry.priority,
                    "depth": entry.depth,
                    "path": list(entry.path),
                    "tier": entry.tier.name.lower(),
                }
                for entry in result.not_inspected
            ],
            "unparsed_files": sorted(result.unparsed_files),
            "parser_scans": {
                "completed": list(result.parser_scans_completed),
                "pending": list(result.parser_scans_pending),
            },
            "unavailable_files": dict(result.unavailable_files),
            "history": [step.to_json() for step in result.history.steps] if result.history else [],
        },
    }


def _navigator_provenance() -> dict:
    package_root = Path(__file__).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    repository = next((parent for parent in package_root.parents if (parent / ".git").exists()), None)
    revision = None
    dirty = None
    if repository is not None:
        revision = _git(repository, "rev-parse", "HEAD")
        dirty = bool(_git(repository, "status", "--porcelain", "--untracked-files=all"))
    return {
        "package_version": importlib.metadata.version("jev-navigator"),
        "source_revision": revision,
        "source_dirty": dirty,
        "source_tree_sha256": digest.hexdigest(),
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _visit(visit: Visit) -> dict:
    return {
        "place": visit.place_key,
        "source": visit.code.source(),
        "code": visit.code.text,
        "probability": visit.probability,
        "verdict": visit.verdict,
        "path": list(visit.path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _report(manifest: dict) -> str:
    source = manifest["source"]
    search = manifest["search"]
    lines = [
        "# Jev navigator evidence pack",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Navigator: `{manifest['navigator']['package_version']}` at "
        f"`{manifest['navigator']['source_revision'] or manifest['navigator']['source_tree_sha256']}`",
        f"- Revision: `{source['revision']}`",
        f"- Scope: {', '.join(f'`{prefix}`' for prefix in source['prefixes'])}",
        f"- Target: {manifest['target']}",
        f"- Outcome: **{search['outcome']}**",
        f"- Search: {search['steps']} opened places, {search['calls']} live calls",
        f"- Provider: requested `{manifest['provider']['requested_model']}`, served "
        f"`{manifest['provider']['served_model']}`",
        f"- Elapsed: {search['duration_seconds']:.3f} seconds",
        f"- Coverage caveat: {len(search['not_inspected'])} places were not inspected; "
        f"{len(search['unparsed_files'])} files failed a completed parser scan. "
        f"Pending parser scans: {', '.join(search['parser_scans']['pending']) or 'none'}.",
        f"- Files that disappeared after inventory: {len(search['unavailable_files'])}.",
        "",
        "## Opened code",
        "",
        "| Set | Probability | Verdict | Source |",
        "| --- | ---: | --- | --- |",
    ]
    for name in ("found", "starts", "unsure", "searched"):
        for visit in search[name]:
            source = visit["source"]
            lines.append(
                f"| {name} | {visit['probability']:.3f} | {visit['verdict']} | "
                f"`{source['file']}:{source['lines'][0]}-{source['lines'][1]}` |"
            )
    if not any(search[name] for name in ("found", "starts", "unsure", "searched")):
        lines.append("| — | — | — | No code was opened. |")
    lines += ["", "## Found spans", ""]
    if not search["found"]:
        lines.append("No span crossed the configured yes threshold.")
    for visit in search["found"]:
        source = visit["source"]
        language = Path(source["file"]).suffix.removeprefix(".")
        lines += [
            f"### `{source['file']}:{source['lines'][0]}-{source['lines'][1]}`",
            "",
            f"Raw P(contains target): **{visit['probability']:.3f}**. Reached by `{source['reached_by']}`.",
            "",
            f"```{language}",
            visit["code"],
            "```",
            "",
        ]
    lines += ["## Not inspected", ""]
    if not search["not_inspected"]:
        lines.append("The search left no frontier places uninspected.")
    else:
        lines += ["| Reason | Priority | Place |", "| --- | ---: | --- |"]
        for entry in search["not_inspected"]:
            lines.append(f"| {entry['reason']} | {entry['priority']:.3f} | `{entry['place']}` |")
    lines += [
        "",
        "The complete source spans, raw probabilities, decisions, and history are in "
        "`manifest.json`; exact provider responses and request hashes are in `journal.jsonl`.",
        "",
    ]
    return "\n".join(lines)
