from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jev_navigator.index.code_index import CodeIndex, ScopeTooWideError
from jev_navigator.index.spans import Span


def test_functions_in_lists_python_and_typescript_functions_with_names(sample_index: CodeIndex) -> None:
    # Act
    python_names = [span.name for span in sample_index.functions_in("app/validation.py")]
    script_names = [span.name for span in sample_index.functions_in("web/handlers.ts")]

    # Assert
    assert python_names == ["validate_order", "check_limits", "noop"]
    assert script_names == ["handleOrder", "parseOrder"]


def test_enclosing_symbol_returns_the_innermost_function(sample_index: CodeIndex) -> None:
    # Act
    span = sample_index.enclosing_symbol("app/orders.py", 6)

    # Assert
    assert span == Span("app/orders.py", 5, 7, "place")


def test_enclosing_symbol_is_none_at_module_level(sample_index: CodeIndex) -> None:
    assert sample_index.enclosing_symbol("app/orders.py", 1) is None


def test_find_definition_finds_the_function_across_files(sample_index: CodeIndex) -> None:
    # Act
    definitions = sample_index.find_definition("check_limits")

    # Assert
    assert definitions == (Span("app/validation.py", 10, 12, "check_limits"),)


def test_find_callers_binds_plain_and_method_calls_to_their_enclosing_function(
    sample_index: CodeIndex,
) -> None:
    # Act
    callers = sample_index.find_callers("validate_order")
    script_callers = sample_index.find_callers("handleOrder")

    # Assert
    assert [(site.file, site.line, site.caller.name) for site in callers] == [("app/orders.py", 6, "place")]
    assert [(site.file, site.caller.name if site.caller else None) for site in script_callers] == [
        ("web/routes.ts", "<anonymous>")
    ]


def test_find_callers_ignores_the_definition_itself(sample_index: CodeIndex) -> None:
    assert sample_index.find_callers("check_limits")[0].caller.name == "validate_order"


def test_find_callees_lists_called_names_inside_a_function(sample_index: CodeIndex) -> None:
    # Arrange
    validate = sample_index.find_definition("validate_order")[0]

    # Act
    callees = sample_index.find_callees(validate)

    # Assert
    assert callees == ("ValueError", "check_limits")


def test_read_slice_and_window_return_exact_lines(sample_index: CodeIndex) -> None:
    # Arrange
    check = sample_index.find_definition("check_limits")[0]

    # Act
    whole_function = sample_index.read_slice(check)
    window = sample_index.read_window("app/validation.py", 11, radius=1)

    # Assert
    assert whole_function.text.splitlines()[0] == "def check_limits(order):"
    assert window.span == Span("app/validation.py", 10, 12)
    assert window.text.splitlines()[1].strip() == 'limit = read_setting("orders.max_items")'


def test_search_text_finds_string_keys_in_scope_files(sample_index: CodeIndex) -> None:
    # Act
    hits = sample_index.search_text("orders.max_items")

    # Assert
    assert [(hit.file, hit.line) for hit in hits] == [("app/validation.py", 1), ("app/validation.py", 11)]


def test_imports_and_dependents_resolve_to_scope_files(sample_index: CodeIndex) -> None:
    # Act
    python_imports = sample_index.imports("app/orders.py")
    script_dependents = sample_index.dependents("web/handlers.ts")

    # Assert
    assert python_imports == ("app/validation.py",)
    assert script_dependents == ("web/routes.ts",)


def test_co_changed_files_counts_commits_shared_with_the_file(sample_index: CodeIndex) -> None:
    # Act
    co_changed = sample_index.co_changed_files("app/orders.py")

    # Assert
    assert co_changed[0] == ("app/validation.py", 2)


def test_scope_wider_than_the_limit_is_refused(sample_repo: Path) -> None:
    with pytest.raises(ScopeTooWideError):
        CodeIndex.from_git(sample_repo, max_files=2)


def test_paths_outside_the_scope_are_refused(sample_repo: Path) -> None:
    # Arrange
    index = CodeIndex.from_git(sample_repo, prefixes=("web/",))

    # Act and Assert
    assert index.find_definition("validate_order") == ()
    with pytest.raises(ValueError, match="outside the index scope"):
        index.read_window("app/orders.py", 1)


def test_every_slice_records_its_source_file_lines_commit_and_how_it_was_reached(
    sample_repo: Path,
) -> None:
    # Arrange
    (sample_repo / "web/handlers.ts").write_text("// changed in the worktree\n")
    index = CodeIndex.from_git(sample_repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sample_repo, capture_output=True, text=True
    ).stdout.strip()

    # Act
    committed = index.read_slice(index.find_definition("check_limits")[0], origin="callee of validate_order")
    edited = index.read_window("web/handlers.ts", 1, radius=0)

    # Assert
    assert committed.source() == {
        "file": "app/validation.py",
        "lines": [10, 12],
        "commit": head,
        "reached_by": "callee of validate_order",
    }
    assert edited.commit == f"{head}+worktree"


def test_find_definition_covers_classes_constants_and_module_assignments(sample_index: CodeIndex) -> None:
    # Act and Assert
    assert sample_index.find_definition("OrderService") == (Span("app/orders.py", 4, 7, "OrderService"),)
    assert sample_index.find_definition("LIMITS_KEY") == (Span("app/validation.py", 1, 1, "LIMITS_KEY"),)
    assert sample_index.find_definition("parseOrder") == (Span("web/handlers.ts", 6, 6, "parseOrder"),)


def test_at_commit_reads_the_old_version_without_touching_the_checkout(sample_repo: Path) -> None:
    # Arrange
    first_commit = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"], cwd=sample_repo, capture_output=True, text=True
    ).stdout.strip()
    checkout_before = (sample_repo / "app/validation.py").read_text()

    # Act
    old = CodeIndex.at_commit(sample_repo, first_commit, prefixes=("app/",))

    # Assert
    assert "web/routes.ts" not in old.files and "app/settings.py" not in old.files
    assert [span.name for span in old.functions_in("app/validation.py")] == ["validate_order", "check_limits"]
    assert old.read_slice(old.find_definition("check_limits")[0]).commit == first_commit
    assert set(old.co_changed_files("app/orders.py")) == {("app/__init__.py", 1), ("app/validation.py", 1)}
    assert (sample_repo / "app/validation.py").read_text() == checkout_before


def test_call_bindings_say_whether_the_target_is_proven(sample_index: CodeIndex) -> None:
    # Act
    imported = sample_index.find_callers("validate_order")[0]
    same_file = sample_index.find_callers("check_limits")[0]
    script_import = sample_index.find_callers("handleOrder")[0]

    # Assert
    assert (imported.binding.status, imported.binding.target.file) == ("resolved", "app/validation.py")
    assert same_file.binding.status == "resolved" and "same file" in same_file.binding.reason
    assert script_import.binding.status == "resolved"


def test_method_calls_and_unknown_names_are_not_claimed_as_resolved(sample_index: CodeIndex) -> None:
    # Arrange
    place = sample_index.find_definition("place")[0]

    # Act
    edges = {edge.name: edge.binding for edge in sample_index.callee_edges(place)}

    # Assert
    assert edges["validate_order"].status == "resolved"
    assert edges["save"].status == "unresolved" and "no definition" in edges["save"].reason
    assert sample_index.find_callers("read_setting")[0].binding.status == "unresolved"


def test_a_name_defined_elsewhere_without_an_import_is_only_a_candidate(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "a.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "b.py").write_text("def use():\n    return helper()\n")
    (tmp_path / "c.py").write_text(
        "class Box:\n    def helper(self):\n        return 2\n\n\ndef run(box):\n    return box.helper()\n"
    )
    index = CodeIndex(tmp_path, ["a.py", "b.py", "c.py"])

    # Act
    bindings = {site.file: site.binding for site in index.find_callers("helper")}

    # Assert
    assert bindings["b.py"].status == "candidate" and "no import" in bindings["b.py"].reason
    assert bindings["c.py"].status == "candidate" and "receiver" in bindings["c.py"].reason


def test_an_injected_binding_resolver_wins(sample_index: CodeIndex, sample_repo: Path) -> None:
    # Arrange
    from jev_navigator.index.bindings import Binding

    class CodeRelations:
        def resolve_call(self, file: str, line: int, name: str, receiver: str | None) -> Binding | None:
            return Binding("resolved", "code relation", Span("app/orders.py", 5, 7, "place"))

    index = CodeIndex.from_git(sample_repo, binding_resolver=CodeRelations())

    # Act
    site = index.find_callers("check_limits")[0]

    # Assert
    assert site.binding.reason == "code relation"


REGISTRY_PY = """\
from app.jobs import send_invoice


def notify():
    return send_invoice


@send_invoice
def decorated():
    pass


def register_jobs(scheduler):
    scheduler.add(send_invoice)
    scheduler.add(job=send_invoice)
    table = {"invoice": send_invoice}
    queue = [send_invoice]
    current = send_invoice
    send_invoice()
"""

REGISTRY_TS = """\
import { sendInvoice } from "./jobs";
export { sendInvoice };
export default sendInvoice;
const table = { invoice: sendInvoice, sendInvoice };
let current = sendInvoice;
current = sendInvoice;
register(sendInvoice);
sendInvoice();
"""

JOBS_PY = """\
def send_invoice(order):
    return order
"""


@pytest.fixture
def registry_index(tmp_path: Path) -> CodeIndex:
    (tmp_path / "app").mkdir()
    (tmp_path / "app/registry.py").write_text(REGISTRY_PY)
    (tmp_path / "app/jobs.py").write_text(JOBS_PY)
    (tmp_path / "app/registry.ts").write_text(REGISTRY_TS)
    return CodeIndex(tmp_path, ["app/registry.py", "app/jobs.py", "app/registry.ts"])


def test_find_references_lists_non_call_usages_with_their_role(registry_index: CodeIndex) -> None:
    # Act
    python = [(ref.line, ref.role) for ref in registry_index.find_references("send_invoice")]
    script = [(ref.line, ref.role) for ref in registry_index.find_references("sendInvoice")]

    # Assert
    assert python == [
        (5, "return"),
        (8, "decorator"),
        (14, "argument"),
        (15, "argument"),
        (16, "collection"),
        (17, "collection"),
        (18, "assignment"),
    ]
    assert script == [
        (2, "export"),
        (3, "export"),
        (4, "collection"),
        (5, "assignment"),
        (6, "assignment"),
        (7, "argument"),
    ]


def test_references_carry_the_enclosing_function_and_a_binding(registry_index: CodeIndex) -> None:
    # Act
    reference = registry_index.find_references("send_invoice")[2]

    # Assert
    assert reference.holder.name == "register_jobs"
    assert reference.binding.status == "resolved"
    assert reference.binding.target == Span("app/jobs.py", 1, 2, "send_invoice")


def test_references_in_lists_the_names_a_function_passes_on_without_calling(
    registry_index: CodeIndex,
) -> None:
    # Arrange
    register_jobs = registry_index.find_definition("register_jobs")[0]

    # Act
    references = registry_index.references_in(register_jobs)

    # Assert
    assert {(ref.name, ref.role) for ref in references} >= {
        ("send_invoice", "argument"),
        ("send_invoice", "collection"),
        ("send_invoice", "assignment"),
    }
    assert "scheduler" not in {ref.name for ref in references}


def test_tracked_symbolic_links_stay_out_of_the_scope(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "skills/real").mkdir(parents=True)
    (tmp_path / "skills/real/tool.py").write_text('KEY = "shared.key"\n')
    (tmp_path / "app").mkdir()
    (tmp_path / "app/main.py").write_text('SETTING = "shared.key"\n')
    (tmp_path / "app/linked_dir").symlink_to("../skills/real", target_is_directory=True)
    (tmp_path / "app/linked_file.py").symlink_to("main.py")
    (tmp_path / "app/outside.py").symlink_to("/etc/hosts")
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
    ):
        subprocess.run(["git", *command], cwd=tmp_path, check=True)

    # Act
    working = CodeIndex.from_git(tmp_path, prefixes=("app/",))
    historical = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("app/",))

    # Assert
    assert working.files == historical.files == ("app/main.py",)
    assert [hit.file for hit in working.search_text("shared.key")] == ["app/main.py"]
    assert [hit.file for hit in historical.search_text("shared.key")] == ["app/main.py"]
