# Frozen selection measurements

The corrected Case 1 code-ranking recipe exposes 116 of 201 registered development lines and
12 of 28 permitted hard lines within its top 128 source units. These are source exposure counts,
not fresh semantic judgments or delivered recall. Its explicit weights are scent 0.5, walk 1,
test penalty 0.1, and zero path, prior and hub weights. The recipe is caller data, not a library
default. Generic ranking blocks remain in `jev_navigator.selection`.

The original paid comparison remains frozen: 2,542 physical attempts cost $2.338094374 under a
$2.50 cap, with no unresolved reservations. At the historical, 20,000-token and 36,000-token rooms,
the registered development lines delivered were:

| Configuration | Lab delivery / 201 | Native Engine delivery / 201 |
| --- | --- | --- |
| Code ranking | 79 / 114 / 133 | 69 / 83 / 85 |
| Outline Choice | 86 / 123 / 140 | 73 / 89 / 92 |
| LLM shortlist | 78 / 98 / 99 | 73 / 84 / 85 |

The historical paid comparison predates the citation rebuild. Its receipt groups and ranks were
not recomputed after that rebuild or the independent review fixes. No new measurement or paid
run was performed for the harness relocation.

## Revisions and limits

The zero-call citation rebuild measured JVN `1c13a7ba6bf0eca53ebcc73b2d41531c0aa97e0c`.
The paid source head was `a7bb65f44374bec240093aa2a961f754218854c4`, with role runtime
`e7fbd398`. Strict native replay used JVN `677ef3e13fc1992982ef00b83505fc76d27454aa`
and Engine `91f57ab4668d1f5f74ee4c9e31def8f8e0564c47`. The harness was moved from JVN
`1940b5351f538ca187714f2517b2b808bc9a14e6`.

The population is 110 development findings with 201 registered lines and thirteen permitted hard
findings with 28 registered lines. No protected labels or thresholds changed. Historical census
inputs include recordings of the rejected #149 whole-frontier configuration, which was removed
from JVN. They are archived evidence, not current retained-workflow reach. Reordering units changes
request company; reuse of individual probabilities across changed groups is a development diagnostic,
not a claim about provider accuracy or valid production cache reuse. Lab allocation and native
consumer delivery measure different boundaries.

## External evidence and reproduction

The caller-specific collection, fitting, replay, preparation, paid dispatch and reporting code,
its README, and all four harness test modules live at:

`~/.local/share/jvn-takeover/2026-10-03/search-design/case1/selection-harness/`

The harness README documents its pinned host environment and explicit zero-provider checks.
JVN CI covers the generic library blocks and their behavioral tests; it does not collect this
external harness. Run `test_paid_native.py` explicitly in the pinned Engine environment.

The complete reports, original receipts, citation comparisons and checksum manifests remain in
`~/.local/share/jvn-takeover/2026-10-03/search-design/case1/`.
`summary.json` records the measured library revisions, result boundaries, limits and artifact hashes.
Only this frozen report and summary remain in JVN's `measurements/selection` directory.
