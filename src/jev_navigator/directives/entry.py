"""Choose concrete initial places for a semantic search from the tracked repository tree.

Code owns the ranking, the hierarchy and every option. Jev only chooses among the actual spans,
directories and files code offers. The spans whose words best match the target are offered first,
in one Choice; the directory walk runs only when no code shares a word with the target. When Jev
picks none of the spans, the search starts from them in the order of Jev's probabilities: a none
pick says only that no span stood out, the spans still share the target's words, and a walk from
the root costs a Choice per directory level with nothing to steer it but path names. Unchosen
options remain in the receipt as an uninspected entry frontier; a Choice probability orders
alternatives but is never treated as a Noul probability that the code contains the target.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from ..index.code_index import CodeIndex
from ..index.keywords import Corpus, matching_lines, target_terms
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.judge import Judge, PickResult
from ..judgments.questions import Pick
from .places import Place, function_place, range_place, window_place
from .shown import shown_slice

MAX_OPTIONS = 200
DIRECTORY_EXAMPLES = 3
PREVIEW_LINES = 3
MATCHED_FILES = 8
MATCHED_SPANS_PER_FILE = 3
MATCHED_LINES = 2
MATCHED_LINE_CHARS = 160
MATCHED_WINDOW_RADIUS = 20
KEYWORD_ENTRY = "keyword entry selection"
NO_MATCH = "none"

CHOOSE_PATH = Pick(
    name="automatic_entry_path",
    instructions=(
        "Which actual directory or file in `options` is the best place to continue looking for the "
        "code described by `target.description`? Choose from the supplied repository entries only."
    ),
)
CHOOSE_SPAN = Pick(
    name="automatic_entry_span",
    instructions=(
        "Which actual source span in `options` is most likely to contain the code described by "
        "`target.description`? Choose from the supplied spans only."
    ),
)
CHOOSE_MATCHED_SPAN = Pick(
    name="keyword_entry_span",
    instructions=(
        "Which actual source span in `options` is most likely to contain the code described by "
        "`target.description`? The options are the spans whose words best match the description; "
        "choose from the supplied spans only."
    ),
    extra_options=((NO_MATCH, "None of these spans is likely to contain the described code."),),
)


@dataclass(frozen=True)
class EntryDecision:
    level: str
    parent: str
    chosen: str
    confidence: float | None
    probabilities: dict[str, float]
    request_sha256: str | None
    options: tuple[dict, ...]

    def to_json(self) -> dict:
        return {
            "level": self.level,
            "parent": self.parent,
            "chosen": self.chosen,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "request_sha256": self.request_sha256,
            "options": list(self.options),
        }


@dataclass(frozen=True)
class EntryCandidate:
    place: Place
    selection_probability: float | None


@dataclass(frozen=True)
class EntrySelection:
    selected_file: str
    candidates: tuple[EntryCandidate, ...]
    decisions: tuple[EntryDecision, ...]

    def to_json(self) -> dict:
        return {
            "selected_file": self.selected_file,
            "decisions": [decision.to_json() for decision in self.decisions],
            "candidates": [
                {
                    "place": candidate.place.key,
                    "signature": candidate.place.signature,
                    "selection_probability": candidate.selection_probability,
                    "selected": position == 0,
                }
                for position, candidate in enumerate(self.candidates)
            ],
        }


@dataclass(frozen=True)
class _PathEntry:
    kind: str
    path: str
    files: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class _SpanEntry:
    span: Span
    description: str
    focus: int | None = None


def choose_initial_candidates(index: CodeIndex, judge: Judge, target: str) -> EntrySelection:
    """Select one file and one span, retaining every closed-choice receipt and span alternative.

    Every matched span the pick did not select stays a candidate, so a wrong first guess leaves the
    search other files to go to rather than only the rest of one file."""
    files = tuple(file for file in index.available_files if language_of(file))
    if not files:
        raise ValueError("the repository scope contains no supported code files")
    matched = _matched_spans(index, files, target)
    if not matched:
        return _walk(index, judge, target, files, [])
    chosen, decision, probabilities = _choose_matched(judge, target, matched)
    alternatives = sorted(
        (entry for entry in matched if entry != chosen),
        key=lambda entry: -probabilities.get(entry.span.key, 0.0),
    )  # stable: equal probabilities keep the keyword ranking's order
    reserve = tuple(
        EntryCandidate(_matched_place(index, entry), probabilities.get(entry.span.key))
        for entry in alternatives
    )
    if chosen is None:
        return EntrySelection(alternatives[0].span.file, reserve, (decision,))
    first = EntryCandidate(_matched_place(index, chosen), probabilities.get(chosen.span.key))
    return EntrySelection(chosen.span.file, (first, *reserve), (decision,))


def _matched_spans(index: CodeIndex, files: tuple[str, ...], target: str) -> tuple[_SpanEntry, ...]:
    """The best-matching spans of the files whose words best match the target, best first: at most
    ``MATCHED_SPANS_PER_FILE`` of each of the ``MATCHED_FILES`` best files, weighed by how rare
    each word is across every file. A file's path counts as part of its text. Each span is described
    by its first lines and its best-matching later lines."""
    terms = target_terms(target)
    if not terms:
        return ()
    corpus = Corpus.of(terms, {file: "\n".join((file, *index.lines(file))) for file in files})
    rarity = corpus.idf()
    scored: list[tuple[float, _SpanEntry]] = []
    for file, _score in corpus.ranked()[:MATCHED_FILES]:
        spans = {entry.span.key: entry for entry in _source_spans(index, file)}
        lines = index.lines(file)
        texts = {
            key: "\n".join((file, *lines[entry.span.start - 1 : entry.span.end]))
            for key, entry in spans.items()
        }
        for key, score in Corpus.of(terms, texts).ranked(rarity)[:MATCHED_SPANS_PER_FILE]:
            entry = spans[key]
            best = _best_lines(index, entry.span, rarity)
            matched = replace(
                entry,
                description=_matched_description(index, entry.span, best),
                focus=_focus(index, entry.span, best),
            )
            scored.append((score, matched))
    return tuple(entry for _score, entry in sorted(scored, key=lambda pair: (-pair[0], pair[1].span.key)))


def _best_lines(index: CodeIndex, span: Span, rarity: Mapping[str, float]) -> list[int]:
    """The lines after the span's opening that share target words, those sharing the rarest first."""
    first = span.start + PREVIEW_LINES
    return [
        first + position for position in matching_lines(index.lines(span.file)[first - 1 : span.end], rarity)
    ]


def _matched_description(index: CodeIndex, span: Span, best: list[int]) -> str:
    """The span's opening and, after it, its ``MATCHED_LINES`` best-matching lines with their line
    numbers, so a long span is offered by what it shares with the target."""
    lines = index.lines(span.file)
    later = "; ".join(
        f"line {number}: {lines[number - 1].strip()[:MATCHED_LINE_CHARS]}"
        for number in sorted(best[:MATCHED_LINES])
    )
    opening = _span_description(index, span)
    return f"{opening} ... {later}" if later else opening


def _focus(index: CodeIndex, span: Span, best: list[int]) -> int | None:
    """The best-matching line when the span, opened whole at the default slice size, would be cut
    before it; None when opening the whole span shows that line."""
    if not best:
        return None
    shown = shown_slice(index.read_slice(span))
    return best[0] if shown is None or best[0] > shown.span.end else None


def _matched_place(index: CodeIndex, entry: _SpanEntry) -> Place:
    """The span whole or, when that would cut off its best-matching line, the window around that
    line under the span's name, as a long declaration is opened from one of its lines."""
    if entry.focus is None:
        return function_place(index, entry.span, KEYWORD_ENTRY)
    return window_place(
        index, entry.span.file, entry.focus, KEYWORD_ENTRY, radius=MATCHED_WINDOW_RADIUS, name=entry.span.name
    )


def _choose_matched(
    judge: Judge, target: str, matched: tuple[_SpanEntry, ...]
) -> tuple[_SpanEntry | None, EntryDecision, dict[str, float]]:
    """One Choice among the matched spans; ``None`` when Jev picks the no-match option."""
    descriptions = [entry.description for entry in matched]
    options = {str(position): description for position, description in enumerate(descriptions)}
    result: PickResult | None = judge.pick(CHOOSE_MATCHED_SPAN, options, {"target": {"description": target}})
    if result is None:
        raise RuntimeError("automatic entry selection had no safe matched span options")
    chosen = None if result.choice == NO_MATCH else matched[int(result.choice)]
    by_span = {
        entry.span.key: result.probabilities.get(str(position), 0.0) for position, entry in enumerate(matched)
    }
    decision = EntryDecision(
        "matched_span",
        "",
        chosen.span.key if chosen is not None else NO_MATCH,
        result.confidence,
        dict(result.probabilities),
        result.request_sha256,
        tuple(
            {"id": str(position), "entry": entry.span.key, "description": descriptions[position]}
            for position, entry in enumerate(matched)
        ),
    )
    return chosen, decision, by_span


def _walk(
    index: CodeIndex, judge: Judge, target: str, files: tuple[str, ...], decisions: list[EntryDecision]
) -> EntrySelection:
    """Choose a directory per level down to one file, then one of its spans."""
    parent = ""
    while True:
        entries = _path_entries(index, files, parent)
        chosen, decision = _choose_path(judge, target, parent, entries)
        decisions += decision
        if chosen.kind == "file":
            selected_file = chosen.path
            break
        parent = chosen.path

    spans = _source_spans(index, selected_file)
    if not spans:
        end = min(40, len(index.lines(selected_file)))
        candidate = EntryCandidate(
            range_place(index, selected_file, 1, end, "automatic entry selection"), None
        )
        return EntrySelection(selected_file, (candidate,), tuple(decisions))
    selected, span_decisions, probabilities = _choose_span(judge, target, selected_file, spans)
    decisions += span_decisions
    ordered = [selected, *(entry for entry in spans if entry != selected)]
    ordered[1:] = sorted(
        ordered[1:], key=lambda entry: (-probabilities.get(entry.span.key, 0.0), entry.span.key)
    )
    candidates = tuple(
        EntryCandidate(
            function_place(index, entry.span, "automatic entry selection"),
            probabilities.get(entry.span.key),
        )
        for entry in ordered
    )
    return EntrySelection(selected_file, candidates, tuple(decisions))


def _path_entries(index: CodeIndex, files: tuple[str, ...], parent: str) -> tuple[_PathEntry, ...]:
    prefix = PurePosixPath(parent).parts
    grouped: dict[tuple[str, str], list[str]] = {}
    for file in files:
        parts = PurePosixPath(file).parts
        if parts[: len(prefix)] != prefix:
            continue
        remainder = parts[len(prefix) :]
        if len(remainder) == 1:
            grouped.setdefault(("file", file), []).append(file)
        elif remainder:
            path = "/".join((*prefix, remainder[0]))
            grouped.setdefault(("directory", path), []).append(file)
    return tuple(
        _PathEntry(kind, path, tuple(paths), _path_description(index, kind, path, paths))
        for (kind, path), paths in sorted(grouped.items())
    )


def _path_description(index: CodeIndex, kind: str, path: str, files: list[str]) -> str:
    if kind == "file":
        return f"file {path}: {_preview(index, path, 1)}"
    examples = "; ".join(f"{file}: {_preview(index, file, 1)}" for file in files[:DIRECTORY_EXAMPLES])
    return f"directory {path}/ ({len(files)} code files); examples: {examples}"


def _source_spans(index: CodeIndex, file: str) -> tuple[_SpanEntry, ...]:
    unique = {
        span.key: _SpanEntry(span, _span_description(index, span))
        for span in (*index.symbols_in(file), *index.declarations_in(file))
    }
    return tuple(unique[key] for key in sorted(unique, key=lambda key: (unique[key].span.start, key)))


def _span_description(index: CodeIndex, span: Span) -> str:
    return f"{span.key}: {_preview(index, span.file, span.start)}"


def _preview(index: CodeIndex, file: str, start: int) -> str:
    lines = index.lines(file)[start - 1 : start - 1 + PREVIEW_LINES]
    return " ".join(line.strip() for line in lines if line.strip())[:360]


def _choose_path(
    judge: Judge,
    target: str,
    parent: str,
    entries: tuple[_PathEntry, ...],
) -> tuple[_PathEntry, list[EntryDecision]]:
    selected, decisions, _ = _choose(
        judge,
        CHOOSE_PATH,
        target,
        "path",
        parent or "/",
        entries,
        lambda entry: entry.description,
        lambda entry: entry.path,
    )
    return selected, decisions


def _choose_span(
    judge: Judge,
    target: str,
    file: str,
    entries: tuple[_SpanEntry, ...],
) -> tuple[_SpanEntry, list[EntryDecision], dict[str, float]]:
    return _choose(
        judge,
        CHOOSE_SPAN,
        target,
        "span",
        file,
        entries,
        lambda entry: entry.description,
        lambda entry: entry.span.key,
    )


def _choose(judge, question, target, level, parent, entries, describe, identify):
    remaining = list(entries)
    decisions: list[EntryDecision] = []
    while len(remaining) > MAX_OPTIONS:
        groups = [
            remaining[offset : offset + MAX_OPTIONS] for offset in range(0, len(remaining), MAX_OPTIONS)
        ]
        descriptions = [
            f"{level} options {identify(group[0])} through {identify(group[-1])} "
            f"({len(group)} actual entries)"
            for group in groups
        ]
        chosen_group, decision, _ = _pick(
            judge,
            question,
            target,
            f"{level}_group",
            parent,
            groups,
            descriptions,
            lambda group: f"{identify(group[0])}…{identify(group[-1])}",
        )
        decisions.append(decision)
        remaining = chosen_group
    if len(remaining) == 1:
        only = remaining[0]
        decision = EntryDecision(
            level,
            parent,
            identify(only),
            None,
            {},
            None,
            ({"id": "0", "entry": identify(only), "description": describe(only)},),
        )
        return only, [*decisions, decision], {}
    chosen, decision, probabilities = _pick(
        judge,
        question,
        target,
        level,
        parent,
        remaining,
        [describe(entry) for entry in remaining],
        identify,
    )
    by_entry = {
        identify(entry): probabilities.get(str(position), 0.0) for position, entry in enumerate(remaining)
    }
    return chosen, [*decisions, decision], by_entry


def _pick(judge, question, target, level, parent, entries, descriptions, identify):
    options = {str(position): description for position, description in enumerate(descriptions)}
    result: PickResult | None = judge.pick(
        question,
        options,
        {"target": {"description": target}, "current": parent},
    )
    if result is None:
        raise RuntimeError(f"automatic entry selection had no safe {level} options")
    position = int(result.choice)
    option_rows = tuple(
        {"id": str(index), "entry": identify(entry), "description": descriptions[index]}
        for index, entry in enumerate(entries)
    )
    decision = EntryDecision(
        level,
        parent,
        identify(entries[position]),
        result.confidence,
        dict(result.probabilities),
        result.request_sha256,
        option_rows,
    )
    return entries[position], decision, dict(result.probabilities)
