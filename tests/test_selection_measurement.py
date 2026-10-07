"""Source geometry is the evidence boundary, including pieces not on old 60-line edges."""

import importlib.util
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import Reading, UnitReader
from jev_navigator.judgments.questions import content_hash
from jev_navigator.selection import graph_from_index, random_walk

spec = importlib.util.spec_from_file_location(
    "selection_collect", Path(__file__).parents[1] / "measurements/selection/collect.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_structured_piece_binds_literal_source_and_keeps_duplicates_unknown(tmp_path):
    path = "settings.json"
    source = '{\n  "first": 1,\n  "second": {\n    "value": 2\n  },\n  "last": 3\n}\n'
    (tmp_path / path).write_text(source)
    index = CodeIndex(tmp_path, [path])
    bodies = collector.bind_file(index, path)
    # Historical structured text can start at a key, inside an enclosing source unit.
    code = '  "second": {\n    "value": 2\n  },'
    key = content_hash({"file": path, "code": code})
    unit = collector.bind_shown(index, path, key, code, bodies)
    assert unit is not None
    assert unit.path == path
    assert unit.ranges == ((3, 5),)

    (tmp_path / path).write_text('[\n  {"value": 2},\n  {"value": 2}\n]\n')
    index = CodeIndex(tmp_path, [path])
    code = '  {"value": 2}'
    assert collector.bind_shown(index, path, "another-id", code, collector.bind_file(index, path)) is None


def test_mask_spelling_aliases_do_not_change_physical_graph_or_walk(tmp_path):
    (tmp_path / "code.py").write_text("def helper():\n    return 1\n\ndef caller():\n    return helper()\n")
    index = CodeIndex(tmp_path, ["code.py"])
    units = UnitReader(index, 76800, listed_only=True, reading=Reading.CODE).list_files(["code.py"]).units
    one = {f"first-{unit.symbol}": replace(unit, id=f"first-{unit.symbol}") for unit in units}
    additional = {f"other-{unit.symbol}": replace(unit, id=f"other-{unit.symbol}") for unit in units}
    baseline, first_names = collector.canonical_source_units(one)
    repeated, all_names = collector.canonical_source_units({**one, **additional})
    first_graph = graph_from_index(index, list(baseline.values()))
    repeated_graph = graph_from_index(index, list(repeated.values()))
    assert len(repeated) == len(units)
    assert repeated_graph.edge_counts == first_graph.edge_counts
    assert repeated_graph.adjacency == first_graph.adjacency
    assert all_names["first-caller"] == all_names["other-caller"]
    assert random_walk(repeated_graph, {all_names["other-caller"]: 1}) == random_walk(
        first_graph, {first_names["first-caller"]: 1}
    )


def test_exact_replay_runs_on_judge_thread_and_rejects_changed_context(tmp_path):
    store_spec = importlib.util.spec_from_file_location(
        "selection_store", Path(__file__).parents[1] / "measurements/selection/replay_store.py"
    )
    module = importlib.util.module_from_spec(store_spec)
    store_spec.loader.exec_module(module)
    state = {"targets": {"point": "read this code"}, "items": [{"file": "a.py", "code": "return 1"}]}
    questions = {"match#0": {"type": "noul", "instructions": "Does the code return one?"}}
    raw = {"model": "recorded-version", "answers": {"match#0": {"type": "noul", "noul": 0.91}}}
    with sqlite3.connect(tmp_path / "store.sqlite") as db:
        db.execute("create table requests(hash text primary key,sent_body blob,response blob)")
        db.execute(
            "insert into requests values (?,?,?)",
            (
                content_hash({"state": state, "questions": questions}),
                json.dumps({"state": state, "questions": questions}).encode(),
                json.dumps(raw).encode(),
            ),
        )
        db.commit()
        client = module.ExactClient(db)
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(client.ask, state, questions).result()
        assert response.noul("match#0").probability == 0.91
        assert response.from_store and response.model == "recorded-version"
        assert response.input_tokens is None
        assert response.source("match#0").request_sha256
        # Identical canonical content with a different serialized field order is not exact.
        with pytest.raises(LookupError):
            client.ask(dict(reversed(list(state.items()))), questions)
        with pytest.raises(LookupError):
            client.ask({**state, "items": [*state["items"], {"file": "b.py", "code": "return 2"}]}, questions)
        assert [attempt["exact_context"] for attempt in client.attempts] == [True, False, False]
