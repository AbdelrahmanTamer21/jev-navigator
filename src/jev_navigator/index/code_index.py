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
    CLASS_KINDS,
    DECLARATION_RULES,
    FUNCTION_KINDS,
    declared_name,
    function_name,
    language_of,
    reference_rules,
)
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_MAX_FILES = 400
DEFAULT_WINDOW_RADIUS = 10
MAX_TEXT_HITS = 20
CO_CHANGE_COMMITS = 200
_COMMIT_MARK = "@@commit@@"
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})


class RevisionMismatchError(ValueError):
    """A directive asked for a revision the index does not hold."""


class ScopeTooWideError(ValueError):
    """The index was asked to cover more files than its limit."""


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
        self._callable_spans: frozenset[Span] | None = None
        self._snapshot: tempfile.TemporaryDirectory | None = None
        self.commit = commit
        self._changed = frozenset(changed_files)
        self.files = tuple(sorted(set(files)))
        if len(self.files) > max_files:
            raise ScopeTooWideError(
                f"{len(self.files)} files is wider than the limit of {max_files}; narrow the scope"
            )
        self._scope = frozenset(self.files)
        self._code_files = tuple(path for path in self.files if language_of(path))
        self.functions_in = cache(self._functions_in)
        self.symbols_in = cache(self._symbols_in)
        self.declarations_in = cache(self._declarations_in)
        self._lines_of = cache(self._read_lines)
        self._script_paths_in = cache(self._read_script_paths)

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
        exported = [*prefixes, *_script_configs(repository, sha)] if prefixes else []
        tools.export_tree(repository, sha, exported, Path(snapshot.name))
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

    def enclosing_symbol(self, file: str, line: int) -> Span | None:
        containing = [span for span in self.functions_in(file) if span.contains(line)]
        return min(containing, key=Span.size, default=None)

    def find_definition(self, name: str) -> tuple[Span, ...]:
        """Functions, classes, and module-level constants, assignments, types, interfaces and enums."""
        found = (
            span
            for file in self._code_files
            for span in (*self.symbols_in(file), *self.declarations_in(file))
            if span.name == name
        )
        return tuple(dict.fromkeys(found))

    def find_callers(self, name: str) -> tuple[CallSite, ...]:
        """Calls to ``name`` found by name in the syntax tree, each with its binding status."""
        sites = {}
        for match in self._call_matches(name):
            file, line = match["file"], _line_of(match)
            receiver = match.get("metaVariables", {}).get("single", {}).get("RECEIVER", {}).get("text")
            sites.setdefault((file, line), receiver)
        return tuple(
            CallSite(
                file, line, self.enclosing_symbol(file, line), self.binding_of(file, line, name, receiver)
            )
            for (file, line), receiver in sorted(sites.items())
        )

    def find_callees(self, function: Span) -> tuple[str, ...]:
        """Names called inside ``function``; see ``callee_edges`` for their bindings."""
        return tuple(dict.fromkeys(edge.name for edge in self.callee_edges(function)))

    def callee_edges(self, function: Span) -> tuple[CallEdge, ...]:
        self._require_in_scope(function.file)
        matches = tools.ast_grep_pattern("$CALLEE($$$)", [function.file], self.root)
        edges = {}
        for match in matches:
            line = _line_of(match)
            expression = match["metaVariables"]["single"]["CALLEE"]["text"]
            name = _last_identifier(expression)
            if function.start < line <= function.end and name and name not in edges:
                receiver = _receiver(expression)
                edges[name] = CallEdge(name, line, self.binding_of(function.file, line, name, receiver))
        return tuple(edges.values())

    def find_references(self, name: str) -> tuple[Reference, ...]:
        """Uses of ``name`` that are not calls: arguments, collection entries, assignments,
        decorators, exports and returns, each with its role, holder and binding. Code reached this
        way (a callback, a registry entry) has no call edge to follow."""
        matches = tools.ast_grep_rules(reference_rules(name), self._code_files, self.root)
        return self._references(matches)

    def references_in(self, function: Span) -> tuple[Reference, ...]:
        """Names ``function`` passes on without calling them, limited to names defined in scope."""
        self._require_in_scope(function.file)
        matches = tools.ast_grep_rules(reference_rules(), [function.file], self.root)
        inside = [match for match in matches if function.start < _line_of(match) <= function.end]
        return tuple(ref for ref in self._references(inside) if self.find_definition(ref.name))

    def _references(self, matches: list[dict]) -> tuple[Reference, ...]:
        found = {(match["file"], _line_of(match), match["ruleId"], match["text"]) for match in matches}
        return tuple(
            Reference(
                name,
                file,
                line,
                role,
                self.enclosing_symbol(file, line),
                self.binding_of(file, line, name, None),
            )
            for file, line, role, name in sorted(found)
        )

    def binding_of(self, file: str, line: int, name: str, receiver: str | None) -> Binding:
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
            self._top_level(file, definitions),
            self._imported_from(file, name),
        )
        return binding_from_facts(facts)

    def _callables(self) -> frozenset[Span]:
        if self._callable_spans is None:
            self._callable_spans = frozenset(
                span for path in self._code_files for span in self.symbols_in(path)
            )
        return self._callable_spans

    def _top_level(self, file: str, definitions: Sequence[Span]) -> tuple[Span, ...]:
        classes = [span for span in self.symbols_in(file) if span not in self.functions_in(file)]
        return tuple(
            span
            for span in definitions
            if span.file == file
            and not any(owner.contains(span.start) and owner != span for owner in classes)
            and not any(other != span and other.contains(span.start) for other in self.functions_in(file))
        )

    def _imported_from(self, file: str, name: str) -> tuple[str, ...]:
        source = "\n".join(self._lines_of(file))
        specifier = imported_names(source, file).get(name)
        if specifier is None:
            return ()
        resolved = resolve_import(specifier, file, self._scope, self._script_paths(file))
        return (resolved,) if resolved else ()

    def read_slice(self, span: Span, origin: str = "") -> CodeSlice:
        lines = self._lines_of(span.file)
        return CodeSlice(
            span, "\n".join(lines[span.start - 1 : span.end]), origin, self._revision_of(span.file)
        )

    def read_window(
        self, file: str, line: int, radius: int = DEFAULT_WINDOW_RADIUS, origin: str = ""
    ) -> CodeSlice:
        line_count = len(self._lines_of(file))
        span = Span(file, max(1, line - radius), min(line_count, line + radius))
        return self.read_slice(span, origin)

    def search_text(self, text: str, max_hits: int = MAX_TEXT_HITS) -> tuple[TextHit, ...]:
        matches = tools.ripgrep_fixed(text, self.files, self.root, max_hits)
        hits = sorted(
            TextHit(match["path"]["text"], match["line_number"], match["lines"]["text"].rstrip("\n"))
            for match in matches
            if match["path"]["text"] in self._scope
        )
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

    def _functions_in(self, file: str) -> tuple[Span, ...]:
        return self._spans_of_kinds(file, FUNCTION_KINDS)

    def _symbols_in(self, file: str) -> tuple[Span, ...]:
        """Functions and classes."""
        return self._spans_of_kinds(file, FUNCTION_KINDS, CLASS_KINDS)

    def _declarations_in(self, file: str) -> tuple[Span, ...]:
        self._require_in_scope(file)
        language = language_of(file)
        if language is None:
            return ()
        rule = f"id: declaration\nlanguage: {language}\nrule:\n{DECLARATION_RULES[language]}"
        lines = self._lines_of(file)
        spans = {
            Span(
                file,
                _line_of(match),
                match["range"]["end"]["line"] + 1,
                declared_name(lines[_line_of(match) - 1]),
            )
            for match in tools.ast_grep_rules(rule, [file], self.root)
        }
        return tuple(sorted(spans))

    def _spans_of_kinds(self, file: str, *kind_tables: dict) -> tuple[Span, ...]:
        self._require_in_scope(file)
        language = language_of(file)
        if language is None:
            return ()
        kinds = [kind for table in kind_tables for kind in table[language]]
        matches = tools.ast_grep_rules(_rules_for(language, kinds), [file], self.root)
        lines = self._lines_of(file)
        spans = {_function_span(file, match, lines) for match in matches}
        return tuple(sorted(spans, key=lambda span: (span.start, -span.end)))

    def _call_matches(self, name: str) -> list[dict]:
        plain = tools.ast_grep_pattern(f"{name}($$$)", self._code_files, self.root)
        method = tools.ast_grep_pattern(f"$RECEIVER.{name}($$$)", self._code_files, self.root)
        return plain + method

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


def _script_configs(repository: Path, commit: str) -> list[str]:
    listed = _regular_files(tools.git(["ls-tree", "-r", "-z", commit], repository))
    return [
        path
        for path in listed
        if PurePosixPath(path).name.startswith("tsconfig")
        and path.endswith(".json")
        and "node_modules/" not in path
    ]


def _rules_for(language: str, kinds: list[str]) -> str:
    listed = "".join(f"\n    - kind: {kind}" for kind in kinds)
    return f"id: symbol\nlanguage: {language}\nrule:\n  any:{listed}"


def _function_span(file: str, match: dict, lines: Sequence[str]) -> Span:
    start = _line_of(match)
    end = match["range"]["end"]["line"] + 1
    return Span(file, start, end, function_name(lines[start - 1]))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1


def _receiver(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


def _last_identifier(expression: str) -> str:
    tail = expression.replace("?.", ".").split(".")[-1]
    return tail if tail.isidentifier() else ""


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
    return [
        {line.strip() for line in block.splitlines() if line.strip()} for block in blocks if block.strip()
    ]
