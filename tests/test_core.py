import sqlite3

import pytest

from agent import Toolbox
from corpus_answer import extract_citations, neutralise_references
from ingest import (
    PaperStats,
    build_vocabulary,
    is_plausible_next_section,
    join_lines,
    looks_like_heading_title,
    remove_page_number,
)


@pytest.fixture
def stats() -> PaperStats:
    return PaperStats()


@pytest.fixture
def toolbox() -> Toolbox:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE papers (title TEXT, year INTEGER)")
    connection.executemany(
        "INSERT INTO papers VALUES (?, ?)",
        [("BERT", 2018), ("Attention Is All You Need", 2017)],
    )
    connection.commit()
    return Toolbox(None, None, None, [], None, connection)


def test_broken_word_is_rejoined(stats):
    result = join_lines("sequence transduc-", "tion models", set(), set(), stats)
    assert result == "sequence transduction models"
    assert stats.hyphen_joins == 1


def test_real_compound_keeps_hyphen(stats):
    vocabulary = {"sequence", "aligned"}
    result = join_lines("instead of sequence-", "aligned recurrence", vocabulary, set(), stats)
    assert result == "instead of sequence-aligned recurrence"
    assert stats.hyphens_kept == 1


def test_known_compound_keeps_hyphen(stats):
    compounds = {"position-wise"}
    result = join_lines("a simple, position-", "wise network", set(), compounds, stats)
    assert result == "a simple, position-wise network"


def test_capitalised_next_line_is_not_joined(stats):
    result = join_lines("ends with Berkeley-", "Parser results", set(), set(), stats)
    assert result == "ends with Berkeley- Parser results"


def test_vocabulary_excludes_hyphenation_fragments():
    pages = {1: ["the transduc-", "tion model is good"]}
    vocabulary, _ = build_vocabulary(pages)
    assert "transduc" not in vocabulary
    assert "tion" not in vocabulary
    assert {"the", "model", "good"} <= vocabulary


def test_compounds_are_collected():
    pages = {1: ["a position-wise feed-forward layer"]}
    _, compounds = build_vocabulary(pages)
    assert {"position-wise", "feed-forward"} <= compounds


@pytest.mark.parametrize(
    ("candidate", "seen", "expected"),
    [
        ("1", [], True),
        ("2", [], False),
        ("3.1", ["3"], True),
        ("3.2", ["3"], False),
        ("4", ["3", "3.2", "3.2.1"], True),
        ("39.2", ["6"], False),
        ("88.3", ["6", "6.3"], False),
        ("25.7", ["6", "6.2"], False),
    ],
)
def test_section_numbering_sequence(candidate, seen, expected):
    assert is_plausible_next_section(candidate, seen) is expected


def test_heading_title_detection():
    assert looks_like_heading_title("Encoder and Decoder Stacks")
    assert not looks_like_heading_title("development set, newstest2013. We used beam search")
    assert not looks_like_heading_title("1.0 · 1020")


def test_page_number_removed_only_at_end():
    assert remove_page_number(["some text", "7"], 7) == ["some text"]
    assert remove_page_number(["2", "Background", "text"], 2) == ["2", "Background", "text"]


def test_bibliography_markers_are_neutralised():
    text = "as shown in [30, 5, 34] and in [2]"
    assert neutralise_references(text) == "as shown in (ref 30, 5, 34) and in (ref 2)"


def test_citations_are_extracted():
    assert extract_citations("First claim [1][3]. Second claim [2, 4].") == [1, 2, 3, 4]
    assert extract_citations("No citations here.") == []


def test_sql_select_is_allowed(toolbox):
    result = toolbox.query_catalog({"sql": "SELECT title FROM papers WHERE year = 2018"})
    assert "BERT" in result


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM papers",
        "DROP TABLE papers",
        "SELECT 1; DROP TABLE papers",
        "WITH x AS (SELECT 1) DELETE FROM papers",
        "",
    ],
)
def test_sql_writes_are_blocked(toolbox, sql):
    assert toolbox.query_catalog({"sql": sql}).startswith("Error")


def test_calculator_evaluates_arithmetic(toolbox):
    assert toolbox.calculator({"expression": "2 ** 3 + 4 / 2"}) == "10.0"
    assert toolbox.calculator({"expression": "-(7 - 10)"}) == "3"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "open('secrets.txt')",
        "10 ** 1000",
        "1 / 0",
        "abc",
    ],
)
def test_calculator_rejects_unsafe_input(toolbox, expression):
    assert toolbox.calculator({"expression": expression}).startswith("Calculation error")