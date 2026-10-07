# Runtime sizing provenance

Frozen measurement provenance moved from the generic source comments during PR #153 review.
These observations were documented at JVN `1940b5351f538ca187714f2517b2b808bc9a14e6`;
no constants or runtime behavior changed. They are historical measurements, not new provider
or parser measurements. Data remains under `~/.local/share/system-one-proof/jvn-eval-2026-10-03/`.

## src/jev_navigator/index/file_shape.py: module provenance

What a file's bytes say about the cost of parsing it, measured without parsing it.

ast-grep's memory follows the number of syntax nodes on a line, and a node is roughly one punctuation
character (``{ } ( ) ; , [ ]``): a 120 KB bundle on one line peaked at 681 MB, the same bundle cut into
lines of about 1,000 characters at 35 MB, and one 2.5 MB image string, which is a single node, at 58 MB.
So the estimate sums, over lines, the square of each line's punctuation count. It fits every real file
measured on 03.10.2026 and 04.10.2026 (the table is in the ``census/parse-peak`` folder of the
evaluation, ``fit-table-punctuation-5.5.md``): the 13 files that really peak above 250 MB stay refused,
and a long string of data is parsed. Code also costs by how much of it there is: on 04.10.2026,
1.2 to 15 MB files of Heedvane TypeScript, saleor Python and dense generated lines peaked at 53 to 75
MB per MB of code over the base, whatever the length of their lines. So the estimate adds a term per
byte, every byte counted as code. A long string literal is one node and costs less, but no reading of
quotes short of the language's own grammar can tell a string from code that sits between two quotes
the language does not pair (an apostrophe in a template literal, JSX text or a comment), and counting
such code as data let 4 MB files peaking at 254 MB be estimated at 26. Counting every byte costs no
real file its place: documenso's 2.5 MB SVG path estimates 226 MB and is still parsed side by side.

## src/jev_navigator/memory_limit.py: ALLOWANCE_MB

What the child processes of one JVN host may hold together. Measured on 04.10.2026 with the streaming
parser (#50) and the declaration-rule fix (#72), parsing every file of an app-sized scope: the largest
measured, saleor/graphql with 15.6 MB of code, peaked at 333 MB, and the worst, Heedvane's
packages/protocol with generated bundles parsed side by side, at 471 MB.

## src/jev_navigator/memory_limit.py: PARSE_HEADROOM_MB

Conservative headroom when sizing ast-grep concurrency and individual files. Originally based on
Python's 261 MB share for saleor/graphql, it remains parser sizing slack, not a host-heap charge.

## src/jev_navigator/judgments/client.py: REQUEST_CHARS_PER_TOKEN

ASCII-escaped characters (``serialized_chars``) per input token, the same value and meaning as the
Engine's ``REQUEST_CHARS_PER_TOKEN`` (analysis-engine ``enginepy/host/system_one.py``). A limit in
tokens becomes a box in characters with ``chars_for_tokens``, never with a second ratio: route boxes
(Drex's 8,192 tokens is 19,660 characters) use it too.

The fit and its data are in ``jvn-eval-2026-10-03/census/request-size-fit`` (``fit-table-request-size.md``).
On 3,096 real requests (03.10.2026) Jev's input is 263 tokens plus 0.22 per state character and 0.29
per question character. The 264 requests of 2,000 tokens or more cost at most 0.38 tokens per
character (2.63 characters per token), so 2.4 (0.417) keeps a 1.10 margin. The data reaches only 11,652
tokens; the limit itself rests on the Engine's measurement of the edge (32,883 pass, about 33,200
refused).

## src/jev_navigator/judgments/client.py: JEV_STATE_TOKEN_LIMIT

The input Jev accepts for the state plus the longest single question (TypeSafe Models page,
docs.typesafe.ai/models). The Engine measured it on 27.09.2026: 32,883 input tokens pass and about
33,200 are refused with ``max_tokens_exceeded``.

## src/jev_navigator/judgments/questions.py: serialized_chars

The size of a value in ASCII-escaped JSON, the one measure of every size box and of packing.

The Engine measures the escaped form (analysis-engine ``evidence_router.py``), and the fit behind
``REQUEST_CHARS_PER_TOKEN`` counts per byte of it, so a non-ASCII character costs its whole escape
here: Chinese comments of 72,043 characters are about 200,000 bytes, over Jev's 32,000 tokens.

## src/jev_navigator/adapters/routes.py: module provenance

The route table: named decision-model routes with automatic fallback, the pattern
analysis-engine proven on its System One seats.

`SYSTEM_ONE_ROUTES` orders named routes (``drex,jev`` makes Drex primary and Jev its
fallback); each route reads `SYSTEM_ONE_<NAME>_ENDPOINT`, `SYSTEM_ONE_<NAME>_API_KEY` and
`SYSTEM_ONE_<NAME>_MODEL`. Only the jev route may use `TYPESAFE_API_KEY` when it has no key of its
own: every other route is another company's or the user's own server, and must never receive the
TypeSafe key, so a route without its own key is refused before any request. A known route name
with `SYSTEM_ONE_<NAME>=1` uses the hosted defaults, so one flag per service is enough when the
defaults apply. The first route is primary; on a failed call the next route is asked, and so on.
The finetuned decider is just another route: `SYSTEM_ONE_ROUTES=decider,jev` with its endpoint, key, model,
`SYSTEM_ONE_DECIDER_INPUT_TOKENS` and `SYSTEM_ONE_DECIDER_CONCURRENCY` set. Every route declares its
input limits, and the routed client packs to the tightest of them, so whichever route answers can
take the request. Concurrency stays per route: each route's client sends at most its own number of
requests at once, so a slow route never throttles the routes after it.

Every route runs the same generic `SystemOneClient` over the official TypeSafe SDK: the SDK
builds, sends and retries the request with exact-byte capture, and the response is decoded
by jev-navigator's parser. `TypeSafeJevClient` stays the Jev-specific client (its defaults,
cancellation and Jev model semantics); routes never use it for non-Jev models. The chain
from `environment.py` fills every file-provided variable before routes resolve, so `.env`
configures routes exactly like the real environment does.

## src/jev_navigator/adapters/routes.py: DREX_INPUT_LIMITS

Drex's limit: 8,192 tokens for the state plus the longest question, which Analysis Engine measured
on 26.09.2026 (``ROUTE_LIMITS`` in ``enginepy/host/system_one.py``: HTTP 422 above it). No bound on
the whole body is measured, so Drex's own refusal stays the only one.

## src/jev_navigator/adapters/routes.py: DREX_CONCURRENCY

Requests Drex admits in flight at once: the Engine measured HTTP 429 on the third (26.09.2026,
``ROUTE_LIMITS``).

## src/jev_navigator/adapters/routes.py: JEV_CONCURRENCY

Requests sent to Jev at once: the Engine saw no 429 up to 128 and flat latency to 32 (27.09.2026,
``ROUTE_LIMITS``), so 32 is a latency choice, not a refusal bound.
