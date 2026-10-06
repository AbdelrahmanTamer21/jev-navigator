# Zero-call reach and scheduling measurements

The 110 tuning cases contain 201 deciding lines. Names add one previously absent code line and
text adds nine previously absent documentation or configuration lines. With the census baseline,
reach rises from 173 to 183 lines, from 86.1% to 91.0%. This is reach, not delivered recall.
The historical delivered result remains 68 lines; a future controlled comparison is needed to
measure delivery under role judgments.

| Measurement | Result |
| --- | ---: |
| Previously absent line reached by names | 1 |
| Previously absent text lines reached | 9 of 14 |
| Those text lines judged by the neutral stand-in within the text allowance | 9 |
| Previously unjudged lines scheduled by replay | 11 of 37 |
| Scheduled lines with a matching recorded answer | 8 |
| Scheduled lines without a recorded answer | 3 |
| Still unscheduled | 26 |

Names recover `copy_sandbox_tree` at `sandbox_copy.py:129`. The text gains include five Markdown
lines, two JSON lines, one configuration line and one YAML line. The exact cases and line
locations are in `summary.json`.

The replay schedules ten lines in the primary search and one through its reserved continuation.
The primary allowance stays at 48 local requests, with 12 additional reserved for continuation.
Text has its own 12-request allowance. These are explicit stage settings, not an equal-total-budget
comparison against the historical shared cap of 48. Five text misses and 26 unjudged lines remain.
One of the latter is the oversized line. Raising the cap or changing source priorities would be
another named configuration to measure, not a result established here.

## Method and limits

The census was read completely. Its final successful reconstruction records were recovered from
its session log. All 37 unjudged and 28 never-located lines are present in those records. The
remaining baseline lines were located or delivered. The frozen dev110 case file was loaded
without loading documenso or other held-out sets.

Name reach uses the shared extraction rules, the additional names absent from the caller's old
extraction and JVN's existing name source. A neutral `ScriptedJevClient` supplies the stand-in.
Ranked discovery lists the candidate pool before its first request. Text composes anchors, files,
named paths, basename or stem matches and text name hits with existing text units and
`find_all_text_async`. It does not inject deciding-label paths into discovery. The text stand-in
uses the real request guard and the separate 12-request allowance. Its 0.5 probabilities are not
semantic evidence. The reach baseline includes previously delivered floor lines, as the census does.

Scheduling replay uses the current JVN search, the same caller inputs and source composition, and
an unresolved role-coverage policy. The recordings have no role answers, so the callback never
claims a role is covered. The continuation composes the caller's existing expansion collector and
JVN's search through a separately capped scope. Recorded probabilities from the older and
corrected newer runs are matched by exact point text, file and masked unit text. The replay never
calls a provider. An unrecorded answer is represented by an explicitly artificial low scheduling
value and is not counted among the eight replayed judgments.

Changing ordering changes batch company. Holding those item probabilities fixed supports a
scheduling simulation, not a claim about new model accuracy or valid production cache reuse. The
production answer-store key and its full batch checks are unchanged. None of the eight replayed
match answers establishes an evidence role. No role-based delivery score is claimed.

## Evidence and reproduction

The complete case-by-case measurements, original recovered census records, frozen unit-point
comparison pool, harness and price script are retained under:

`~/.local/share/jvn-takeover/2026-10-03/evidence-roles-and-reach/`

`summary.json` pins the complete measurement and recovered-census hashes. The price artifact pins
the frozen unit-point file. The source revision is the scan's `5a501e77`; the historical caller
checkout is `engine-e733ea0c`; this branch starts at JVN `d226be6b`. The harness composes the pinned
caller's input and expansion collectors without editing that checkout. Local measurements use
only scripted responses and recorded answers. No paid calls, Meta guard calls or candidate calls
were made. No secrets are included in this report.

## Validation

The focused source, composition, names, frontier and Find All checks passed 104 tests. Related
CLI, judgment, relaxed-key and search-coverage checks passed 190 tests. A final composition and
frontier rerun passed 27 tests. Ruff lint and formatting checks passed across `src` and `tests`.
The basename regression fails without the basename source, and the settling regression fails
when the role prerequisite alone is disabled. The names test fails before the typed extraction
block exists. The whole suite was not run locally, following the repository's instructions.
