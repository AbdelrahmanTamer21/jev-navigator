from __future__ import annotations

import pytest

from jev_navigator.index.keywords import Corpus, target_terms, words


@pytest.mark.parametrize(
    ("code", "prose"),
    [
        ("rasterizePages", "rasterized pages"),
        ("parseHTTPHeader", "parse the HTTP header"),
        ("GRÖSSE_LIMIT", "Größe limit"),
        ("FinanzierungenPrüfen", "Finanzierung prüfen"),
        ("ГрузПодъём", "груз подъём"),
    ],
)
def test_code_and_prose_spellings_of_the_same_words_match(code: str, prose: str) -> None:
    assert set(words(code)) <= set(words(prose))


def test_text_without_spaces_matches_a_shorter_phrase_inside_it() -> None:
    corpus = Corpus.of(target_terms("車両価格"), {"price.ts": "// 車両価格を計算する", "other.ts": "// 在庫"})

    assert [name for name, _score in corpus.ranked()] == ["price.ts"]


def test_a_rare_shared_word_outranks_a_word_every_text_has() -> None:
    texts = {
        "a.py": "the order the order the order",
        "b.py": "the rasterize step",
        "c.py": "the other thing",
    }

    ranked = Corpus.of(target_terms("the rasterize step for the order"), texts).ranked()

    assert ranked[0][0] == "b.py"
