# Extending jev-navigator

The library is a set of small pieces you import and compose in your own functions. There is no plugin
system, registry or base class: a new use case is a plain function of 30 to 60 lines.

## The pieces

| Piece | What it gives you |
| --- | --- |
| `CodeIndex` | mechanical lookups over a narrowed scope: definitions, callers, callees, references, text, imports, git history |
| `operations` | ready-made combinations of lookups: slices, traces, similar functions, code named in a doc |
| `Check`, `Pick`, `Rate` | one closed question each: yes or no, one option of a list, a level on a scale |
| `Judge` | asks questions with masking, a secret scan, a cache, budgets and a journal; returns raw probabilities |
| `find_code` | a best-first search that opens places until the code a description names is found |
| `find_all` | seed-first function search: expand the static component, batch containment judgments, then examine disconnected functions |
| `places.MOVES` | the ways a search lists the neighbours of an opened place; pick a subset or add your own |
| `StopRule`, `History` | your own stop check over a search's history, reading only the sections you select |
| `LlmStep` | an opt-in LLM call for the cases where Jev's answer is not clear enough |

## The rule for questions

Code holds the goal, the loop and the stopping. Jev gets concrete state and one closed judgment:

1. Say what code will do with each answer, and what the costly error is.
2. Do everything mechanical in code: which functions exist, who calls whom, which files changed.
3. Ask one `Check` per item about a concrete property of supplied code, with yes and no criteria.
4. Never ask whether something is false, wrong or contradicts something; ask for the concrete
   property instead, and let code combine the answers.
5. Handle every outcome: yes, no, unsure, and low confidence.

## Worked example: which route handlers write an audit entry?

```python
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.thresholds import NoulVerdict

WRITES_AUDIT_ENTRY = Check(
    name="writes_audit_entry",
    instructions=(
        "Does `{item}.code`, or one of the functions in `{item}.called`, write an entry to the audit log?"
    ),
    yes=Criterion(
        "A line in `{item}.code` or in one of `{item}.called` writes an audit log entry, "
        "for example `audit_log.write(...)`."
    ),
    no=Criterion(
        "No line in `{item}.code` or in `{item}.called` writes an audit log entry. "
        "A name that only mentions auditing does not count."
    ),
)


def handler_item(index: CodeIndex, handler) -> dict:
    """The handler's code plus the bodies of the functions it calls, one level deep."""
    called = []
    for edge in index.callee_edges(handler):
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        for span in targets:
            code = index.read_slice(span).text
            called.append({"name": edge.name, "binding": edge.binding.status, "code": code})
    return {
        "file": handler.file,
        "lines": [handler.start, handler.end],
        "code": index.read_slice(handler).text,
        "called": called,
    }


def handlers_writing_audit_entries(index: CodeIndex, judge: Judge, handlers):
    items = [handler_item(index, handler) for handler in handlers]
    results = judge.check_each(WRITES_AUDIT_ENTRY, items)
    return {
        "writes": [result.item for result in results if result.verdict == NoulVerdict.YES],
        "does_not": [result.item for result in results if result.verdict == NoulVerdict.NO],
        "unsure": [result.item for result in results if result.verdict == NoulVerdict.UNSURE],
    }
```

Why it is built this way:

- Code, not Jev, decides which handlers exist and which functions each one calls.
- A handler that delegates the write to a helper is still judged correctly, because the helper's body
  is in `{item}.called`; Jev is never asked to guess what an unseen call does. This holds for one level
  only: a write two calls down is not in the state, so read "no" as "no write within one call", or
  expand further in code. Each entry in `called` keeps its binding status, so a call the index could
  not resolve stays visible instead of looking like a proven one. A yes can rest on the body of a
  function the index could not prove (`candidate`, `unresolved` or `unknown`); when you need proof,
  keep only the entries whose binding is `resolved`.
- Each handler is its own question, so one handler cannot hide another.
- Unsure stays unsure: it is reported, never counted as "does not write".

Test it offline with `ScriptedJevClient` and AAA tests, including the unsure path, before any paid call.

## Searching instead of listing

### Compose a seed-first Find All search

This is an ordinary function composition, not a workflow interpreter. Obtain concrete seeds from
`find_code` or a symbol lookup, follow relationships with `operations.trace_graph`, and judge each
candidate body with the existing containment question. `find_all` combines the latter two pieces:

```python
from jev_navigator.directives.find_all import find_all

seeds = index.find_definition("check_limits")
result = find_all(index, judge, "the check that limits items per order", seeds)
for match in result.matched:
    print(match.item["file"], match.item["lines"], match.probability)
```

The seed is a candidate, not an assumed match. Connected functions are examined first; the fallback
then enumerates every other in-scope function, even with unrelated names. Both phases batch atomic
questions through `Judge`; code deduplicates by source span. Entire bodies and source hashes are
retained. `CodeIndex.functions_in_files` batches fact collection instead of launching a parser scan
for each file. No default file or live-call cap is added by this composition.

Pass `include_disconnected=False` for a deliberately partial, component-only search. Supply a `Check`
through `check=` to examine another concrete property of each body; use `{item}.code` and the shared
`target.description`. Independent additional properties belong in `Judge.check_every`, which asks
them together and keeps the answers separate. The engineer authors the branches and stopping rule;
Jev does not decide whether to invent a workflow or declare the repository fully understood.
Checks in one batch need distinct names because each name identifies its returned result list.

`functions_examined` means the function inventory was examined, not that every semantic answer is
correct. Module-level statements, declarations and multi-function behaviors require a different
unit/composition. `uncertain`, parser failures, unavailable files and unsupported grammars remain
visible. A graph link marked candidate never becomes a proven call because its body matched.
Cancellation keeps coverage partial and reporting does not trigger scans of untouched files.
Provider errors propagate; retained journal receipts describe the work actually performed.

The CLI composes entry selection and `find_code` with this function. A seed-search miss still permits
the disconnected fallback. Use `jvn findall "functions that enforce the order item limit"` or
`jvn --json '{"command":"findall","target":"functions that enforce the order item limit"}'`.
The evidence pack retains the seed search, per-function answers and raw request identities.

### Find one location

When code cannot list the candidates, search: `find_code(index, judge, description, start)` opens
places (starts, then Jev's picks, then the best-scored neighbours) and returns `found`, `searched`,
`unsure`, `starts` and `not_inspected` with reasons. A start place never counts as found. Add a
`StopRule` with your own concrete check when the target is spread over several places.

Read `searched` and the outcome `nothing_left` as "opened and judged unlikely", never as "the code does
not exist": one "no" about one place can be wrong. When nothing is found, rank the opened places by
their probability and treat the best one as the likeliest place.

Documents and other files without a supported code grammar remain searchable as text and can
participate in text-based moves. Syntax operations return no symbols, calls or references for them;
they are never sent to ast-grep with an empty language rule. This does not claim their text was
parsed as code.

The CLI creates a unique run directory under `jvn-results/` in the invocation directory when `--out`
is omitted. Each run retains its report, manifest and request journal. Generated result directories
are excluded from the CLI's source inventory so repeated searches do not search their own evidence.
Library callers can similarly pass `exclude_paths` to `CodeIndex.from_directory`.

Agents can pass the same CLI request as JSON with `jvn --json request.json`, an inline JSON object,
or `jvn --json -` for stdin. `jvn schema find` emits its JSON Schema without model calls. The CLI parser remains the single owner of options, types and defaults. `target` is
required; `command` defaults to `find`. JSON mode emits the result and evidence-pack paths on
stdout while progress stays on stderr. See the README for the request and result fields.

The CLI report distinguishes candidates not independently opened from text already included inside
a larger opened span. `target_found` means the search stopped after finding a match; `budget` means an
actual configured limit stopped an otherwise viable candidate. The recorded candidate score is the
entry Choice probability for an initial alternative or the could-contain Noul probability for a
navigation neighbour; those are different judgments. Navigation elapsed time excludes indexing and
entry selection. Journal `exact` flags, not the presence of JSON, determine whether responses are
wire captures or re-encoded SDK data.

## Choosing how the search moves

A move is a plain function of the index and the opened code that returns places. `places.MOVES` maps
each built-in move's name to its function (callers, callees, references, code passed on, the same
file, quoted keys and environment variables, co-changed files, the lines before and after) and is
read-only. Pass `moves=` to `find_code`, `find_code_async` or `context_for_comment` to use a subset,
for example `{name: MOVES[name] for name in ("callers", "callees")}`, or add a function of your own.
`FindResult.moves` and the final `stop` step of the history name the moves the search used, so every
result says how it was found. Your move's places go through the same filter as the built-in ones:
places that open the same lines of the same file are kept once, the first move that listed them wins,
a place wholly inside the opened code is dropped, and each move's cap counts only places no earlier
move kept. Order the places a move returns by how likely they are to matter, because the cap keeps the
first ones.

## Stopping on your own check

`find_code(..., stop_rule=StopRule(check))` asks your check after every round. `sections` chooses what
the check sees: only the `fetched` code with its sources (the default), each step's operation,
arguments and code (`history`), the search's own verdicts (`decisions`), or sections you add with
`StopRule(context={"shown_code": ...})`. Only `decisions` carries verdicts, so a check reads them only
when you select that section.
Pick the smallest view the question needs; the fewer unrelated fields, the steadier the answer.

## Adding your own steps to a history

`HistoryStep(operation, arguments, fetched, judgments, decision)` is generic. An agent's tool call and
its result can be appended next to a search's own steps, and every check reads them through the same
sections.

## When Jev's answer is not enough

Nothing in the library escalates on its own. If a low-confidence pick should go to an LLM, add an
`LlmStep` to your function: you define when it runs, what context it gets, the answer contract and the
connector.

## Async hosts

Every judge method has an `*_async` form, and `find_code_async` is the same search. Pass any object
with `model` and `async ask(state, questions)`; masking, the store, budgets and the journal behave
exactly as on the sync path.

## Testing

Test every function offline with `ScriptedJevClient` (or `AsyncScriptedJevClient`) in Arrange, Act,
Assert form. Script the unsure and low-confidence answers too, and assert what your code does with
them. `CapturingJevClient` records a function's exact first request, so you can review the question
text before any paid call.


### Parser facts and naming

Function and class names come from the matched AST node, with the containing physical line used
only when the node does not contain its binding name (for example an assigned anonymous function).
This keeps a method on a one-line TypeScript class distinct from its enclosing class.
Persistent facts are keyed by source bytes, language, parser version and `FACT_RULE_VERSION`.
A change to extracted facts must change that rule identity so existing cached results are reparsed.
