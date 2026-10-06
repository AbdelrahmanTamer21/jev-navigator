# Role questions, proposal only

This proposal replaces one match question with six independent Nouls per point and unit. State
contains only `point` and `unit` (`file`, `code`). The caller supplies a single checkable search
point, and the unit is shown whole or as an explicitly identified piece. No labels, previous
answers, history, other units or verdicts enter state. All six questions run together against
that same state. The candidate contains the exact instructions and opposing criteria.

The questions are whether the unit makes the decision, restricts the behavior with a guard,
establishes a dependent value, performs the effect, delegates the call or transports the value, and answers
the caller's search. A guard may prevent the behavior described by the point and still be useful
search evidence. Roles overlap deliberately. A check can both decide and guard. Delegation can coexist with any local role. Code derives forwarding-only when delegation is high
and all four local role probabilities are low. Satisfying the search is separate
from the truth of any statement. Neither a Noul nor the reducer returns that truth.

The host composes raw answers in code, after inference:

```python
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

ROLES = ("decide", "guard", "value", "effect")


@dataclass(frozen=True)
class EvidenceAnswers:
    unit_id: str
    probabilities: Mapping[str, float]

    @property
    def relevance(self) -> float:
        return max(self.probabilities[role] for role in (*ROLES, "satisfied"))

    @property
    def forwarding_only(self) -> bool:
        return (
            self.probabilities["delegates"] >= 0.80
            and max(self.probabilities[role] for role in ROLES) <= 0.20
        )


def compose(answers: Sequence[EvidenceAnswers], required: Sequence[str]):
    # Bands below are proposed experimental settings, not calibrated acceptance rules.
    best = {
        role: max(answers, key=lambda answer: answer.probabilities[role], default=None) for role in required
    }
    covered = {
        role for role, answer in best.items() if answer is not None and answer.probabilities[role] >= 0.80
    }
    return {
        "best_by_role": best,
        "uncovered_roles": set(required) - covered,
        "ranked": sorted(answers, key=lambda answer: -answer.relevance),
        "follow_units": [answer.unit_id for answer in answers if answer.forwarding_only],
    }
```

Retain the best candidate for every required role, even when it remains uncertain. Do not let
forwarding demote a unit that scored for a deciding role. Only code-composed forwarding-only evidence
routes to the next code operation. It does not prove the forwarded implementation absent. Settling
requires no uncovered role and no pending expansion. A weighted aggregate never hides an uncovered
role. Unknown or refused answers remain unknown. Parsing whether a unit declares a named symbol
remains a code fact. The caller chooses required roles explicitly; JVN does not infer them from a
caller domain. This proposal has not changed production questions or selection.

Compare today's match wording and this role set on the same frozen unit-point pairs from the
corrected dev110 recordings. Both arms use one point and unit per request, with the same source,
masking, model and unit boundaries. This deliberately removes unrelated batch state. Run all six
role questions in one request, rather than six requests. Freeze the units, questions, source
revision and selection rule before any trial. Measure deciding-line recall and per-role retention,
along with raw answers, price and latency. Historical multi-unit match probabilities are not an
accuracy control for these new isolated requests. No documenso data enters the comparison.

Revision `proposal-v2` replaces the combined `forward` question with the independent `delegates`
question. A wrapper that validates input and then delegates a write can score high for both `guard`
and `delegates`. Code keeps its guard evidence and does not classify it as forwarding-only.
The reducer example is executed by the offline regression test with the candidate's question ids.

The new candidate file SHA-256 is
`d1b1d84d125b65e361a56d69386836eccbb2e490ff0906103ac83555d9132a2f`.
The canonical request SHA-256 is
`56b6db1a7eeaf728881b354919e196deacde6d24752edebd8e2c871d7a0936fc`.
Earlier prepared packets and review receipts describe the earlier candidate, not this revision.

`prepare-summary.json` records Meta Builder 0.6.0's free prepare output identities and resolved
state paths. Six question packets and one workflow packet were prepared. Atomicity pairs and
semantic term contrast remain unreviewed. Prepare produced no semantic scores. No `guard` or
candidate call was run. Full prepared output is retained in `prepared.json` and the local proof folder; the summary
pins its hash. Prepare is a structural inspection, not proof of question quality.

`price.json` records the comparison estimate at the live TypeSafe price on 6 October 2026. It
includes rerunning both arms, one request per frozen unit-point pair per arm, with output tokens
free. It excludes a future Meta guard. The token estimate uses measured historical density and
fixed request overhead, and must be checked against reported usage in a future pilot. No price is
permission to run the trial.

Sources: [Noul](https://docs.typesafe.ai/primitives/noul),
[state](https://docs.typesafe.ai/concepts/state),
[composite scoring](https://docs.typesafe.ai/patterns/composite-scoring),
[pricing and limits](https://docs.typesafe.ai/models).

The frozen comparison contains 47,410 unit-point pairs and would send 94,820 requests across
both arms. The estimated combined price is $4.45 to $5.50. Today's question accounts for $1.69 to
$2.06 and the role set for $2.76 to $3.44. This estimate is not a cap or authorization.
