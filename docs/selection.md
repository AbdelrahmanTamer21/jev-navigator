# Code selection blocks

`jev_navigator.selection` supplies independent blocks for choosing which admitted code to read
next. A caller owns candidate discovery, the query, seed locations, weights and oracle. These
blocks do not change the default `find` or `find_all` recipe.

`outline(index, path, terms)` renders a compact file summary with its path, definitions and line
ranges, imports and source lines containing the supplied terms. `OutlineLimits` bounds each
part. Every omitted definition, import or match and every clipped line is recorded. Parsing
requests one file at a time.

`scent_document(id, path, symbol, source)` reduces one source unit to word counts. The spelling
map expands case, separators and identifier parts; retrieval also folds regular plurals and
acronym parts. `ScentIndex` applies BM25 to identifiers, strings and paths and supplies a separate
file-name signal. Source comments supply no scent. The index stores counts rather than parser
output or source bodies.

`graph_from_index(index, units)` joins admitted units through proven calls and references,
imports, literal named files, same-file membership and git co-change. Sparse virtual file nodes
avoid constructing a clique for every file relationship. Unresolved parser bindings are counted
and do not create guessed edges. `random_walk(graph, seeds)` uses weighted degree normalization,
restart at the supplied seeds and a degree penalty for hubs. Dangling mass restarts at those
same seeds. Numerical iterations bound computation and never impose a search-time budget.

`rank_features` supplies normalized scent, file-name and graph signals along with explicit
caller priors, graph degree and test-file status. `rank` combines them with `RankWeights` and
preserves the caller's order for ties. Hub and test weights are penalties. No weight is silently
learned or configured through an environment flag.

```python
from jev_navigator.selection import (
    ActivePolicy, ScentIndex, active_search, graph_from_index,
    rank, rank_features, scent_document,
)

# Discovery and source reading remain with the existing index and reader owners.
documents = ScentIndex(
    scent_document(unit.id, unit.path, unit.symbol, source_for(unit))
    for unit in admitted_units
)
graph = graph_from_index(index, admitted_units)
features = rank_features(documents, query, graph, seed_weights)
ordered = rank([unit.id for unit in admitted_units], features, weights)
scores = {id: weights.score(features[id]) for id in ordered}
result = active_search(ordered, scores, graph, oracle, policy=ActivePolicy())
```

An oracle implements `judge(tuple_of_ids)` and returns a mapping from offered identities to
`Observation(probability, confirmed)`. The active loop judges groups of up to 16, propagates
confirmed observations, reranks and estimates the marginal value of its next group. Ranking scores
are shifted up when their minimum is negative and scaled by the shifted maximum, preserving order
in [0, 1]. Equal nonpositive scores normalize to zero. These values feed the observed-yield gain
heuristic alongside propagated relevance. Missing
observations remain unknown and end that run without counting as negatives. The default
marginal rule is a declared heuristic, not calibrated Jev confidence. A caller can compare it
with `min_expected_gain=0` under the same request guard. Results retain offered groups, actual
observations, unknown and pending identities, and the stopping reason.

`outline_choice(query, files)` prepares one bounded Choice among up to 16 numbered outlines
and `none`. It asks where the visible outline gives a reason to read full source. It supplies
no answer about unread implementation. `selected_file(answer, files, min_confidence=...)`
admits that read only under the caller's measured confidence policy. A new Choice threshold
requires its own measurement; it does not inherit a Noul threshold. Preparation never calls
a provider. The existing request owner still masks, checks request size, judges and stores
exact context before a live call.

The frozen [selection report](../measurements/selection/REPORT.md) and summary retain historical
results, revisions and limits. Caller-specific recipes and their tests live in the external harness
linked there. Historical census data does not establish the reach of retained workflows. Strict
replay preserves exact groups; changed-group probability reuse is a development diagnostic. The
measurements do not install a new default search configuration.
