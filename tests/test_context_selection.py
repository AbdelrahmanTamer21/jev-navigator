"""Static context delivery from real bindings, paths and structural excerpts."""

from git_repos import commit_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.selection.context import ONE_HOP_NAMED_CONTEXT


def test_call_order_counts_distinct_neighbours_and_stops_after_one_hop(tmp_path):
    commit_files(
        tmp_path,
        {
            "batch.py": "def seed(value):\n    policy = 'settings/policy.json'\n    return value\n",
            "bridge.py": "from batch import seed\n",
            "a_single.py": "from bridge import seed\n\ndef single(value):\n    return seed(value)\n",
            "z_repeated.py": "from bridge import seed\n\ndef repeated(value):\n"
            "    seed(value)\n    seed(value)\n    return seed(value)\n",
            "grandcaller.py": "from a_single import single\n\ndef enqueue(value):\n"
            "    return single(value)\n",
            "settings/policy.json": '{"reject_negative": true}\n',
        },
    )
    selection = ONE_HOP_NAMED_CONTEXT.build(CodeIndex.from_git(tmp_path))
    graph, named = selection.queues(["batch.py"])
    assert [unit.path for unit in graph] == ["a_single.py", "z_repeated.py"]
    assert [unit.path for unit in named] == ["settings/policy.json"]
    pieces = list(ONE_HOP_NAMED_CONTEXT.interleave([graph, named]))
    assert [unit.path for unit in pieces] == ["a_single.py", "settings/policy.json", "z_repeated.py"]


def test_structural_excerpt_keeps_branch_exits_and_physical_gaps(tmp_path):
    body = "def validate(value):\n" + "".join(f"    padding_{n} = {n}\n" for n in range(30))
    body += "    if value < 0:\n        raise ValueError('deciding exit')\n    return value\n"
    commit_files(
        tmp_path,
        {
            "worker.py": "from bridge import validate\n\ndef process(value):\n    return validate(value)\n",
            "bridge.py": "from validator import validate\n",
            "validator.py": body,
        },
    )
    selection = ONE_HOP_NAMED_CONTEXT.build(CodeIndex.from_git(tmp_path))
    graph, _ = selection.queues(["worker.py"])
    validator = next(unit for unit in graph if unit.path == "validator.py")
    query = selection.query(["worker.py"], ["reject invalid input"])
    rendered = selection.excerpts.render(validator, query)
    assert "1: def validate(value):" in rendered
    assert "32:     if value < 0:" in rendered
    assert "33:         raise ValueError('deciding exit')" in rendered
    assert "34:     return value" in rendered
    assert "... ELIDED lines 4-29 (26 lines) ..." in rendered
    assert "padding_10" not in rendered
