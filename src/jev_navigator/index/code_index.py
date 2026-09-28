"""Mechanical code lookups over a narrowed set of files: no model, only ast-grep, ripgrep and git.

Every lookup stays inside the scope the index was built with. A scope wider than ``max_files`` is
refused, because whole-repository searches are slow and send far more code onward than any
decision needs; the caller narrows first (a directory, a changed-file list, a module).
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path, PurePosixPath

from . import tools
from .bindings import Binding, BindingResolver, CallFacts, binding_from_facts
from .imports import imported_modules, imported_names, resolve_import
from .languages import (
    language_of,
)
from .scope_scan import FileStructure, ReferenceMatch, Unparsed, scan_calls, scan_references, scan_structure
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_MAX_FILES = 400
DEFAULT_WINDOW_RADIUS = 10
MAX_TEXT_HITS = 20
CO_CHANGE_COMMITS = 200
_COMMIT_MARK = "@@commit@@"
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})


_NO_STRUCTURE = FileStructure((), (), ())


class RevisionMismatchError(ValueError):
    """A directive asked for a revision the index does not hold."""


class ScopeTooWideError(ValueError):
    """The index was asked to cover more files than its limit."""


class UnsafePathError(ValueError):
    """A scope path is a symbolic link or resolves outside the index root, so reading it could leave
    the root."""


class CodeIndex:
    def __init__(
        self,
        root: Path,
        files: Iterable[str],
        *,
        max_files: int = DEFAULT_MAX_FILES,
        commit: str = "",
        changed_files: Iterable[str] = (),
        git_root: Path | None = None,
        binding_resolver: BindingResolver | None = None,
    ) -> None:
        self.root = Path(root)
        self.git_root = Path(git_root) if git_root is not None else self.root
        self.binding_resolver = binding_resolver
        self._snapshot: tempfile.TemporaryDirectory | None = None
        self.commit = commit
        self._changed = frozenset(changed_files)
        self.files = tuple(sorted(set(files)))
        if len(self.files) > max_files:
            raise ScopeTooWideError(
                f"{len(self.files)} files is wider than the limit of {max_files}; narrow the scope"
            )
        _require_inside(self.root, self.files)
        self._scope = frozenset(self.files)
        self._code_files = tuple(path for path in self.files if language_of(path))
        self._lines_of = cache(self._read_lines)
        self._script_paths_in = cache(self._read_script_paths)
        self._unparsed = Unparsed()
        self._structure = cache(
            lambda: scan_structure(self._code_files, self.root, self._lines_of, self._unparsed)
        )
        self._calls = cache(lambda: scan_calls(self._code_files, self.root, self._unparsed))
        self._call_counts = cache(lambda: Counter(call.name for call in self._calls()))
        self._reference_matches = cache(lambda: scan_references(self._code_files, self.root, self._unparsed))
        self._definitions = cache(self._definitions_by_name)
        self._callables = cache(self._callable_spans)
        self._top_level_in = cache(self._top_level_spans)
        self._names_imported = cache(self._read_imported_names)
        self._binding = cache(self._compute_binding)

    @classmethod
    def from_git(
        cls,
        root: Path,
        prefixes: Sequence[str] = (),
        *,
        max_files: int = DEFAULT_MAX_FILES,
        binding_resolver: BindingResolver | None = None,
    ) -> CodeIndex:
        """The tracked regular files under ``prefixes`` (every one when none are given). Symbolic links
        and submodules are left out: a link can point outside the scope, or at a directory."""
        root = Path(root)
        listed = _regular_files(tools.git(["ls-files", "--stage", "-z", "--", *prefixes], root))
        commit = tools.git(["rev-parse", "HEAD"], root).strip()
        changed = _changed_paths(tools.git(["status", "--porcelain", "-z", "--", *prefixes], root))
        return cls(
            root,
            listed,
            max_files=max_files,
            commit=commit,
            changed_files=changed,
            binding_resolver=binding_resolver,
        )

    @classmethod
    def at_commit(
        cls,
        repository: Path,
        commit: str,
        prefixes: Sequence[str] = (),
        *,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> CodeIndex:
        """The regular files under ``prefixes`` as they were at ``commit``, read from git objects into a
        private temporary directory; the checkout is never touched. History lookups still run in
        ``repository``. Symbolic links and submodules are left out, as in ``from_git``. The commit's
        tsconfig files come along (outside the scope), so path aliases resolve."""
        repository = Path(repository)
        sha = tools.git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repository).strip()
        listed = _regular_files(tools.git(["ls-tree", "-r", "-z", sha, "--", *prefixes], repository))
        if len(listed) > max_files:
            raise ScopeTooWideError(
                f"{len(listed)} files is wider than the limit of {max_files}; narrow the scope"
            )
        snapshot = tempfile.TemporaryDirectory(prefix=f"jev-navigator-{sha[:8]}-")
        tools.export_blobs(repository, _blobs_to_export(repository, sha, listed), Path(snapshot.name))
        index = cls(snapshot.name, listed, max_files=max_files, commit=sha, git_root=repository)
        index._snapshot = snapshot
        return index

    def require_commit(self, commit: str) -> None:
        """Raises unless this index reads exactly ``commit``: a working-tree index with uncommitted
        changes in scope never counts. Use ``CodeIndex.at_commit`` for a historical revision."""
        if not self.commit or not self.commit.startswith(commit) or len(commit) < 7:
            raise RevisionMismatchError(
                f"the index reads {self.commit or 'an unknown revision'}, not {commit}"
            )
        if self._changed:
            raise RevisionMismatchError(
                f"the working tree has uncommitted changes in {sorted(self._changed)[:3]}"
            )

    @property
    def unparsed_files(self) -> frozenset[str]:
        """Files a scan could not parse in time, or that the grammar reports ERROR nodes on, so what
        the parser recovered from them is partial. Reading it runs any scan not yet run, so the list is
        complete. Their definitions and calls are unknown, not absent: bindings that may depend on
        them say ``unknown``, and a search over the scope never reports ``nothing_left``."""
        self._structure()
        self._calls()
        self._reference_matches()
        return self._unparsed.files

    def enclosing_symbol(self, file: str, line: int) -> Span | None:
        containing = [span for span in self.functions_in(file) if span.contains(line)]
        return min(containing, key=Span.size, default=None)

    def functions_in(self, file: str) -> tuple[Span, ...]:
        return self._file_structure(file).functions

    def symbols_in(self, file: str) -> tuple[Span, ...]:
        """Functions and classes."""
        return self._file_structure(file).symbols

    def declarations_in(self, file: str) -> tuple[Span, ...]:
        """Module-level constants, assignments, types, interfaces and enums."""
        return self._file_structure(file).declarations

    def find_definition(self, name: str) -> tuple[Span, ...]:
        """Functions, classes, and module-level constants, assignments, types, interfaces and enums."""
        return self._definitions().get(name, ())

    def find_callers(self, name: str) -> tuple[CallSite, ...]:
        """Calls to ``name`` found by name in the syntax tree, each with its binding status. When one
        line holds both ``x.name(...)`` and ``name(...)``, the plain call stands for that line."""
        sites: dict[tuple[str, int], str | None] = {}
        for call in self._calls():
            key = (call.file, call.line)
            if call.name == name and (key not in sites or call.receiver is None):
                sites[key] = call.receiver
        return tuple(
            CallSite(
                file, line, self.enclosing_symbol(file, line), self.binding_of(file, line, name, receiver)
            )
            for (file, line), receiver in sorted(sites.items())
        )

    def call_site_count(self, name: str) -> int:
        """How many call sites in scope call ``name``; a name called from fewer places is more specific."""
        return self._call_counts()[name]

    def find_callees(self, function: Span) -> tuple[str, ...]:
        """Names called inside ``function``; see ``callee_edges`` for their bindings."""
        return tuple(dict.fromkeys(edge.name for edge in self.callee_edges(function)))

    def callee_edges(self, function: Span) -> tuple[CallEdge, ...]:
        self._require_in_scope(function.file)
        edges: dict[str, CallEdge] = {}
        for call in self._calls():
            inside = call.file == function.file and function.start <= call.line <= function.end
            if inside and call.name not in edges:
                binding = self.binding_of(call.file, call.line, call.name, call.receiver)
                edges[call.name] = CallEdge(call.name, call.line, binding)
        return tuple(edges.values())

    def find_references(self, name: str) -> tuple[Reference, ...]:
        """Uses of ``name`` that are not calls: arguments, collection entries, assignments,
        decorators, exports, returns, method receivers, types and conditions, each with its role,
        holder and binding. Code reached this way (a callback, a registry entry, a parameter typed
        with a class) has no call edge to follow."""
        return self._references(match for match in self._reference_matches() if match.name == name)

    def references_in(self, function: Span) -> tuple[Reference, ...]:
        """Names ``function`` passes on without calling them, limited to names defined in scope."""
        self._require_in_scope(function.file)
        inside = (
            match
            for match in self._reference_matches()
            if match.file == function.file and function.start <= match.line <= function.end
        )
        return tuple(ref for ref in self._references(inside) if self.find_definition(ref.name))

    def binding_of(self, file: str, line: int, name: str, receiver: str | None) -> Binding:
        """Computed once per call site and cached for the life of the index."""
        return self._binding(file, line, name, receiver)

    def _references(self, matches: Iterable[ReferenceMatch]) -> tuple[Reference, ...]:
        return tuple(
            Reference(
                match.name,
                match.file,
                match.line,
                match.role,
                self.enclosing_symbol(match.file, match.line),
                self.binding_of(match.file, match.line, match.name, None),
            )
            for match in sorted(set(matches))
        )

    def _compute_binding(self, file: str, line: int, name: str, receiver: str | None) -> Binding:
        if self.binding_resolver is not None:
            injected = self.binding_resolver.resolve_call(file, line, name, receiver)
            if injected is not None:
                return injected
        definitions = tuple(span for span in self.find_definition(name) if span in self._callables())
        facts = CallFacts(
            file,
            name,
            receiver,
            definitions,
            tuple(span for span in definitions if span in self._top_level_in(file)),
            self._imported_from(file, name),
            self.unparsed_files,
        )
        return binding_from_facts(facts)

    def _file_structure(self, file: str) -> FileStructure:
        self._require_in_scope(file)
        return self._structure().get(file, _NO_STRUCTURE)

    def _definitions_by_name(self) -> dict[str, tuple[Span, ...]]:
        by_name: dict[str, dict[Span, None]] = {}
        for file in self._code_files:
            for span in (*self.symbols_in(file), *self.declarations_in(file)):
                by_name.setdefault(span.name, {})[span] = None
        return {name: tuple(spans) for name, spans in by_name.items()}

    def _callable_spans(self) -> frozenset[Span]:
        return frozenset(span for path in self._code_files for span in self.symbols_in(path))

    def _top_level_spans(self, file: str) -> frozenset[Span]:
        """Symbols of ``file`` that no class or other function contains."""
        symbols = self.symbols_in(file)
        return frozenset(
            span
            for span in symbols
            if not any(other != span and other.contains(span.start) for other in symbols)
        )

    def _imported_from(self, file: str, name: str) -> tuple[str, ...]:
        specifier = self._names_imported(file).get(name)
        if specifier is None:
            return ()
        resolved = resolve_import(specifier, file, self._scope, self._script_paths(file))
        return (resolved,) if resolved else ()

    def _read_imported_names(self, file: str) -> dict[str, str]:
        return imported_names("\n".join(self._lines_of(file)), file)

    def read_slice(self, span: Span, origin: str = "") -> CodeSlice:
        lines = self._lines_of(span.file)
        return CodeSlice(
            span, "\n".join(lines[span.start - 1 : span.end]), origin, self._revision_of(span.file)
        )

    def read_window(
        self, file: str, line: int, radius: int = DEFAULT_WINDOW_RADIUS, origin: str = ""
    ) -> CodeSlice:
        span = Span(file, max(1, line - radius), min(len(self._lines_of(file)), line + radius))
        return self.read_slice(span, origin)

    def search_text(self, text: str, max_hits: int = MAX_TEXT_HITS) -> tuple[TextHit, ...]:
        found = tools.ripgrep_fixed(text, self.files, self.root, max_hits)
        hits = sorted(hit for hit in found if hit.file in self._scope)
        return tuple(hits[:max_hits])

    def imports(self, file: str) -> tuple[str, ...]:
        source = "\n".join(self._lines_of(file))
        script_paths = self._script_paths(file)
        resolved = (
            resolve_import(specifier, file, self._scope, script_paths)
            for specifier in imported_modules(source, file)
        )
        return tuple(dict.fromkeys(path for path in resolved if path))

    def dependents(self, file: str) -> tuple[str, ...]:
        self._require_in_scope(file)
        return tuple(path for path in self._code_files if path != file and file in self.imports(path))

    def co_changed_files(self, file: str, limit: int = 5) -> tuple[tuple[str, int], ...]:
        """Scope files most often committed together with ``file``, with their shared-commit counts."""
        self._require_in_scope(file)
        log = tools.git(
            [
                "-c",
                "core.quotePath=false",
                "log",
                *([self.commit] if self.commit else []),
                f"-n{CO_CHANGE_COMMITS}",
                "--full-diff",
                "--name-only",
                f"--format=format:{_COMMIT_MARK}",
                "--",
                file,
            ],
            self.git_root,
        )
        counts = Counter(
            path for commit in _commits(log) if file in commit for path in commit if path != file
        )
        in_scope = [(path, count) for path, count in counts.most_common() if path in self._scope]
        return tuple(sorted(in_scope, key=lambda item: (-item[1], item[0]))[:limit])

    def _revision_of(self, file: str) -> str:
        if not self.commit:
            return ""
        return f"{self.commit}+worktree" if file in self._changed else self.commit

    def lines(self, file: str) -> tuple[str, ...]:
        return self._lines_of(file)

    def _script_paths(self, file: str) -> ScriptPaths | None:
        """The path aliases of the tsconfig.json nearest to a script file, read once per directory."""
        return None if file.endswith(".py") else self._script_paths_in(str(PurePosixPath(file).parent))

    def _read_script_paths(self, directory: str) -> ScriptPaths | None:
        return nearest_script_paths(self.root, directory)

    def _read_lines(self, file: str) -> tuple[str, ...]:
        self._require_in_scope(file)
        return _split_lines((self.root / file).read_text(errors="replace"))

    def _require_in_scope(self, file: str) -> None:
        if file not in self._scope:
            raise ValueError(f"{file} is outside the index scope")


def _blobs_to_export(repository: Path, commit: str, listed: Sequence[str]) -> dict[str, str]:
    """Object ids of the listed files and of every tsconfig file in ``commit``, keyed by path."""
    tree = _tree_blobs(tools.git(["ls-tree", "-r", "-z", commit], repository))
    return {path: tree[path] for path in [*listed, *_script_configs(tree)]}


def _tree_blobs(listing: str) -> dict[str, str]:
    """Object ids of the regular files in ``git ls-tree -r -z`` output, keyed by path."""
    blobs = {}
    for line in listing.split("\0"):
        details, _, path = line.partition("\t")
        mode, _, object_id = details.partition(" blob ")
        if mode in _REGULAR_FILE_MODES:
            blobs[path] = object_id
    return blobs


def _script_configs(paths: Iterable[str]) -> list[str]:
    return [
        path
        for path in paths
        if PurePosixPath(path).name.startswith("tsconfig")
        and path.endswith(".json")
        and "node_modules/" not in path
    ]


def _require_inside(root: Path, files: Iterable[str]) -> None:
    """Every reader (the parser, ripgrep, plain reads) opens scope files by path, so a path that is a
    link or leads out of the root is refused before any of them runs."""
    resolved_root = root.resolve()
    for file in files:
        path = root / file
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise UnsafePathError(f"{file} is a symbolic link or lies outside {root}")


def _split_lines(text: str) -> tuple[str, ...]:
    """Lines split at newlines only, as the parser counts them; ``str.splitlines`` also splits at form
    feeds and other separators, which would shift every line number after them."""
    lines = text.replace("\r", "").split("\n")
    return tuple(lines[:-1] if lines and lines[-1] == "" else lines)


def _changed_paths(status: str) -> list[str]:
    """Paths from ``git status --porcelain -z``; a rename or copy record is followed by its old path."""
    records = status.split("\0")
    changed = []
    skip_next = False
    for record in records:
        if skip_next:
            skip_next = False
            continue
        if len(record) > 3:
            changed.append(record[3:])
            skip_next = record[0] in "RC"
    return changed


def _regular_files(listing: str) -> list[str]:
    """Paths from ``git ls-files --stage -z`` or ``git ls-tree -r -z`` output whose mode is a regular
    file. NUL separation keeps names with non-ASCII characters exactly as they are on disk."""
    files = []
    for line in listing.split("\0"):
        details, _, path = line.partition("\t")
        if details.split(" ", 1)[0] in _REGULAR_FILE_MODES:
            files.append(path)
    return list(dict.fromkeys(files))


def _commits(log: str) -> list[set[str]]:
    blocks = log.split(_COMMIT_MARK)
    return [{line.strip() for line in block.split("\n") if line.strip()} for block in blocks if block.strip()]
