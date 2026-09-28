from __future__ import annotations

from jev_navigator import operations
from jev_navigator.index.code_index import CodeIndex


def test_slice_around_returns_the_enclosing_function(sample_index: CodeIndex) -> None:
    # Act
    code = operations.slice_around(sample_index, "app/validation.py", 11)

    # Assert
    assert code.span.name == "check_limits"
    assert code.text.startswith("def check_limits(order):")


def test_slice_around_falls_back_to_a_window_at_module_level(sample_index: CodeIndex) -> None:
    # Act
    code = operations.slice_around(sample_index, "app/validation.py", 1, radius=2)

    # Assert
    assert (code.span.start, code.span.end) == (1, 3)


def test_callers_of_file_collects_calls_from_other_files_for_every_function(sample_index: CodeIndex) -> None:
    # Act
    sites = operations.callers_of_file(sample_index, "app/validation.py")

    # Assert
    assert [(site.file, site.caller.name) for site in sites] == [("app/orders.py", "place")]


def test_trace_callers_walks_outward_hop_by_hop(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callers(sample_index, "check_limits", depth=2)

    # Assert
    assert [(step.hop, step.function.name, step.reached_from) for step in steps] == [
        (1, "validate_order", "check_limits"),
        (2, "place", "validate_order"),
    ]


def test_trace_depth_is_capped_at_three(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callers(sample_index, "check_limits", depth=9)

    # Assert
    assert max(step.hop for step in steps) <= operations.MAX_TRACE_DEPTH


def test_trace_callees_follows_only_functions_defined_in_scope(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callees(sample_index, "validate_order", depth=2)

    # Assert
    assert [(step.hop, step.function.name) for step in steps] == [(1, "check_limits")]


def test_similar_functions_keeps_only_functions_sharing_calls_or_name_words(sample_index: CodeIndex) -> None:
    # Act
    names = [span.name for span in operations.similar_functions(sample_index, "handleOrder")]

    # Assert
    assert "parseOrder" in names
    assert "registerRoutes" not in names
    assert "handleOrder" not in names


def test_code_named_in_doc_finds_mentioned_functions(sample_index: CodeIndex) -> None:
    # Act
    spans = operations.code_named_in_doc(
        sample_index, "Orders are checked by `validate_order`, which calls check_limits."
    )

    # Assert
    assert [span.name for span in spans] == ["validate_order", "check_limits"]


def test_comment_above_a_function_describes_the_whole_function(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 24)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (25, 26, "Basket")


def test_comment_above_a_decorated_function_includes_the_decorator(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 3)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (4, 6, "charge")
    assert code.text.startswith("@functools.cache")


def test_comment_followed_by_blank_lines_reaches_the_next_symbol(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 9)

    # Assert
    assert (code.span.start, code.span.end, code.span.name) == (11, 12, "normalise")


def test_trailing_comment_on_a_simple_statement_describes_that_line(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 16)

    # Assert
    assert (code.span.start, code.span.end) == (16, 16)


def test_trailing_comment_on_a_block_opener_describes_the_whole_block(sample_index: CodeIndex) -> None:
    # Act
    code = operations.code_described_by_comment(sample_index, "app/comments.py", 17)

    # Assert
    assert (code.span.start, code.span.end) == (17, 19)


def test_code_named_in_doc_finds_constants_and_classes(sample_index: CodeIndex) -> None:
    # Act
    spans = operations.code_named_in_doc(sample_index, "OrderService reads LIMITS_KEY before placing.")

    # Assert
    assert [span.name for span in spans] == ["OrderService", "LIMITS_KEY"]


def test_trace_steps_carry_the_binding_of_the_linking_call(sample_index: CodeIndex) -> None:
    # Act
    steps = operations.trace_callers(sample_index, "check_limits", depth=2)

    # Assert
    assert [(step.function.name, step.binding.status) for step in steps] == [
        ("validate_order", "resolved"),
        ("place", "resolved"),
    ]
