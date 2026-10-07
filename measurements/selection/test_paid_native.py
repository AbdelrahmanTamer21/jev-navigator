"""Integration proof executed explicitly with the pinned Engine and JVN environment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

def test_real_engine_accepts_ranked_code_and_text_and_reports_pending_units(tmp_path):
    """Runs in the pinned Engine environment; CI without that host records an explicit skip."""
    pytest.importorskip("enginepy.workflows.document_analysis.evidence_pack")
    import asyncio
    import subprocess
    from dataclasses import asdict

    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps
    from paid_native import build, packet_in_room

    from jev_navigator.index.code_index import CodeIndex
    from jev_navigator.index.units import Reading, list_units
    from jev_navigator.judgments.answers import JevResponse, NoulAnswer
    from jev_navigator.judgments.judge import Judge

    (tmp_path / "source.py").write_text(
        "\n".join(f"def helper_{n}(value):\n    return value + 1\n" for n in range(17))
    )
    (tmp_path / "README.md").write_text("The offset is one.\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    index = CodeIndex(tmp_path, ["source.py", "README.md"])
    units = (
        *list_units(index, ["README.md"], box_chars=76800, reading=Reading.TEXT).units,
        *list_units(index, ["source.py"], box_chars=76800, reading=Reading.CODE).units,
    )
    candidates = [{"unit": asdict(unit)} for unit in units]
    context = {
        "repository": str(tmp_path),
        "claim": {
            "id": "trial",
            "statement": "The helper adds one.",
            "evidence": [{"file": "source.py", "line": 2}],
        },
        "points": {"point": "The helper adds one."},
    }
    relations = CodeRelations(
        str(tmp_path), repository_import_maps(str(tmp_path), frozenset()), withheld=frozenset()
    )

    class VisiblePositive:
        model = "fixture"

        def ask(self, state, questions):
            assert len(state["items"]) == 16
            assert len(questions) == 96
            return JevResponse({name: NoulAnswer(0.9) for name in questions}, self.model)

    pack, result = asyncio.run(
        build(
            relations, index, context, candidates, Judge(VisiblePositive(), max_calls=1, items_per_request=16)
        )
    )
    assert result.stopped_by == "call_cap"
    assert len(result.not_judged) == 2
    assert len(result.judged["point"]) == 16
    packet = packet_in_room(pack, relations, context, 7200)
    assert "The offset is one." in packet.render()
    assert len(packet.render()) <= 28800
