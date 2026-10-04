from __future__ import annotations

from pathlib import Path

import pytest

from jev_navigator.index.file_shape import (
    DENSE_AVERAGE_LINE_CHARS,
    DENSE_MINIMUM_BYTES,
    LARGE_FILE_BYTES,
    LONG_LINE_CHARS,
    MAX_PARSE_PEAK_MB,
    PARSEABLE_UP_TO_BYTES,
    Trigger,
    measure,
    placement_of,
    shape_of,
)


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
    assert shape.fits_side_by_side


def test_the_estimate_follows_the_punctuation_of_a_line_not_its_length() -> None:
    image_string = measure(b"export const background = '" + b"A" * 2_000_000 + b"';\n")
    minified_code = measure(_one_line(2_000_000))

    assert image_string.parse_peak_mb == pytest.approx(25, abs=1)
    assert minified_code.parse_peak_mb > 1_000


def test_code_of_short_lines_costs_by_its_size() -> None:
    # Measured on 04.10.2026: ordinary and dense code peaks at 53 to 75 MB per MB once it is parsed
    statement = b"    total = compute(items, limit)\n"

    side_by_side = measure(statement * 80_000)
    over = measure(statement * 90_000)

    assert side_by_side.fits_side_by_side
    assert not over.fits_side_by_side
    assert over.refusal(MAX_PARSE_PEAK_MB) is not None and "90,000 lines" in over.refusal(MAX_PARSE_PEAK_MB)


def test_every_measured_file_of_short_lines_is_estimated_at_or_above_its_real_peak() -> None:
    # (bytes of code, ast-grep's real peak in MB), 04.10.2026: Heedvane TypeScript repeated, saleor
    # Python joined, and 55,000 dense lines of 45 operands in each language
    measured = [
        (1_170_000, 90.5),
        (2_300_000, 161.8),
        (8_050_000, 480.8),
        (1_210_000, 92.1),
        (6_030_000, 387.0),
        (14_620_000, 909.0),
        (14_890_000, 1142.1),
    ]

    dense_line = b"v = " + b" + ".join(b"a%d" % operand for operand in range(45)) + b"\n"
    for code_bytes, real_peak in measured:
        assert measure(dense_line * (code_bytes // len(dense_line))).parse_peak_mb >= real_peak


def test_a_line_of_one_long_string_is_priced_by_its_punctuation_not_its_size() -> None:
    path = b'  d="' + b"M708 195.8c.4-1.5.8-3.5 2-4.7 " * 80_000 + b'"\n'

    assert measure(b"<path\n" + path + b"/>\n").parse_peak_mb < 30


def test_the_bound_for_a_one_line_bundle_sits_between_25000_and_32000_characters() -> None:
    assert measure(_one_line(25_000)).fits_side_by_side
    assert not measure(_one_line(32_000)).fits_side_by_side
    assert measure(_one_line(32_000)).parse_peak_mb > MAX_PARSE_PEAK_MB


def test_the_refusal_names_the_estimated_peak_and_the_longest_line() -> None:
    reason = measure(_one_line(668_777)).refusal(MAX_PARSE_PEAK_MB)

    assert reason is not None
    assert reason.startswith("too large to parse: estimated parse peak ")
    assert reason.endswith(", 1 line, longest line 668,777 bytes")
    assert measure(_one_line(20_000)).refusal(MAX_PARSE_PEAK_MB) is None


def _lines(count: int, width: int) -> bytes:
    return b"\n".join(b"x" * width for _ in range(count)) + b"\n"


def test_a_line_over_the_long_line_limit_fires_only_that_trigger_on_a_small_file() -> None:
    content = b"short\n" * 10 + b"y" * (LONG_LINE_CHARS + 1) + b"\n" + b"short\n" * 5000

    assert measure(content).triggers == (Trigger.LONG_LINE,)
    assert measure(b"short\n" * 10 + b"y" * LONG_LINE_CHARS + b"\n" + b"short\n" * 5000).triggers == ()


def test_dense_lines_fire_only_above_the_average_and_the_minimum_size() -> None:
    dense = _lines(count=45, width=DENSE_AVERAGE_LINE_CHARS + 5)
    at_the_average = _lines(count=40, width=DENSE_AVERAGE_LINE_CHARS - 1)
    too_small = _lines(count=30, width=DENSE_AVERAGE_LINE_CHARS + 5)

    assert measure(dense).triggers == (Trigger.DENSE_LINES,)
    assert measure(at_the_average).triggers == ()
    assert measure(too_small).size_bytes < DENSE_MINIMUM_BYTES
    assert measure(too_small).triggers == ()


def test_a_file_over_the_large_file_limit_fires_only_that_trigger_when_its_lines_are_short() -> None:
    over = _lines(count=6_000, width=LARGE_FILE_BYTES // 6_000)
    under = b"x" * 80 + b"\n"
    under = under * ((LARGE_FILE_BYTES // len(under)) - 1)

    assert measure(over).size_bytes > LARGE_FILE_BYTES
    assert measure(over).triggers == (Trigger.LARGE_FILE,)
    assert measure(under).size_bytes <= LARGE_FILE_BYTES
    assert measure(under).triggers == ()


def test_a_hand_written_module_of_many_short_lines_fires_nothing() -> None:
    module = _lines(count=6_000, width=38)

    assert measure(module).size_bytes > 200_000
    assert measure(module).triggers == ()


def test_a_component_with_one_long_svg_path_line_fires_nothing() -> None:
    component = _lines(count=95, width=30) + b"<path d='" + b"M1 2 " * 1_050 + b"'/>\n"

    shape = measure(component)

    assert shape.longest_line > 5_000
    assert shape.size_bytes < LARGE_FILE_BYTES
    assert shape.triggers == ()
    assert shape.fits_side_by_side


def test_a_one_line_minified_bundle_fires_the_line_and_density_triggers_but_is_still_parseable() -> None:
    shape = measure(_one_line(24_745))

    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES)
    assert shape.fits_side_by_side


def test_a_two_and_a_half_megabyte_image_string_fires_every_trigger_and_is_still_parseable() -> None:
    background = b"export const background = '" + b"A" * 2_504_000 + b"';\n"

    shape = measure(background)

    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES, Trigger.LARGE_FILE)
    assert shape.fits_side_by_side


def test_shape_of_reads_one_file_given_the_repository_folder_and_the_path(tmp_path: Path) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_bytes(_one_line(24_745))

    shape = shape_of(tmp_path, "dist/bundle.js")

    assert shape.size_bytes == 24_745
    assert (shape.line_count, shape.longest_line) == (1, 24_745)
    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES)


def test_the_stat_shortcut_is_safe_a_file_up_to_the_safe_size_can_never_be_over_the_bound() -> None:
    assert measure(b";" * PARSEABLE_UP_TO_BYTES).fits_side_by_side
    assert measure(b"\n" * PARSEABLE_UP_TO_BYTES).fits_side_by_side
    assert not measure(b";" * (PARSEABLE_UP_TO_BYTES + 700)).fits_side_by_side


def test_a_small_file_is_cleared_from_its_size_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "small.js").write_bytes(_one_line(PARSEABLE_UP_TO_BYTES))
    (tmp_path / "big.js").write_bytes(_one_line(668_777))
    reads: list[Path] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)

    assert placement_of(tmp_path, "small.js", MAX_PARSE_PEAK_MB)[1] is None
    assert reads == []
    assert placement_of(tmp_path, "big.js", MAX_PARSE_PEAK_MB)[1].startswith("too large to parse")
    assert [path.name for path in reads] == ["big.js"]


def test_line_lengths_are_measured_in_bytes_so_multibyte_text_errs_on_the_safe_side() -> None:
    assert measure(("é" * 1_000).encode()).longest_line == 2_000


def _minified_bundle(characters: int) -> bytes:
    schema = b'{"type":"object","properties":{"id":{"type":"string"},"tags":[1,2,3]}},'
    return (schema * (characters // len(schema) + 1))[:characters]


def test_a_launch_contract_shaped_minified_bundle_is_refused() -> None:
    assert not measure(_minified_bundle(668_777)).fits_side_by_side
    assert not measure(_minified_bundle(130_000)).fits_side_by_side


def test_a_135000_character_minified_line_is_still_refused(tmp_path: Path) -> None:
    (tmp_path / "bundle.js").write_bytes(_one_line(135_000))

    assert placement_of(tmp_path, "bundle.js", MAX_PARSE_PEAK_MB)[1].startswith("too large to parse")


def test_many_999_byte_minified_lines_are_refused_because_every_line_counts() -> None:
    line = _minified_bundle(999)
    bundle = b"\n".join([line] * 2_500)

    shape = measure(bundle)

    assert shape.longest_line == 999
    assert not shape.fits_side_by_side


def test_a_235_kb_hand_written_module_is_cleared() -> None:
    statement = b"  const total = items.reduce((sum, item) => sum + item.value, 0);\n"
    module = statement * 3_600

    shape = measure(module)

    assert shape.size_bytes > 235_000
    assert shape.fits_side_by_side
    assert shape.parse_peak_mb < 50
