from __future__ import annotations

import pytest

from jev_navigator.index.file_shape import MAX_PARSE_PEAK_MB, measure


def _one_line(characters: int) -> bytes:
    statement = b"function f(){return 1};"
    return (statement * (characters // len(statement) + 1))[:characters]


def test_a_file_reports_its_size_line_count_longest_line_and_characters_per_line() -> None:
    shape = measure(b"ab\ncdef\n\ng\n")

    assert (shape.size_bytes, shape.line_count, shape.longest_line) == (11, 4, 4)
    assert shape.chars_per_line == pytest.approx(11 / 4)


def test_an_empty_file_has_no_lines_and_costs_only_the_base() -> None:
    shape = measure(b"")

    assert (shape.line_count, shape.longest_line, shape.chars_per_line) == (0, 0, 0.0)
    assert not shape.too_large_to_parse


@pytest.mark.parametrize(
    ("characters", "measured_mb"),
    [(24_745, 54), (46_329, 122), (88_165, 399), (119_687, 681), (134_721, 956)],
)
def test_the_parse_peak_estimate_fits_the_measured_one_line_bundles_within_ten_percent(
    characters: int, measured_mb: int
) -> None:
    estimate = measure(_one_line(characters)).parse_peak_mb

    assert estimate == pytest.approx(measured_mb, rel=0.10)


def test_many_short_lines_stay_cheap_however_large_the_file_is() -> None:
    shape = measure(b"x = 1\n" * 120_000)

    assert shape.size_bytes == 720_000
    assert shape.parse_peak_mb < 30
    assert not shape.too_large_to_parse


def test_the_bound_sits_between_a_20000_and_a_70000_character_single_line() -> None:
    assert not measure(_one_line(20_000)).too_large_to_parse
    assert not measure(_one_line(65_000)).too_large_to_parse
    assert measure(_one_line(75_000)).too_large_to_parse
    assert measure(_one_line(75_000)).parse_peak_mb > MAX_PARSE_PEAK_MB


def test_the_refusal_names_the_estimated_peak_and_the_longest_line() -> None:
    reason = measure(_one_line(668_777)).refusal

    assert reason == "too large to parse: estimated parse peak 22 GB, longest line 668,777 characters"
    assert measure(_one_line(20_000)).refusal is None
