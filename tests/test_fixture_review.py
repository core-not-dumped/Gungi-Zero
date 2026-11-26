"""사람 검토 문서가 fixture 에서 만든 그대로인지 본다."""

from scripts.fixture_review import DOCUMENT, render_document
from tests.gungi_fixtures import SECTIONS

REGENERATE = "uv run python -m scripts.fixture_review"


def test_the_committed_document_is_generated_from_the_fixtures() -> None:
    assert DOCUMENT.read_text(encoding="utf-8") == render_document(), (
        f"문서를 다시 만든다: {REGENERATE}"
    )


def test_every_fixture_shows_its_rule_board_and_expected_moves() -> None:
    document = render_document()
    for section in SECTIONS:
        assert section.rule in document
        for fixture in section.fixtures:
            _, _, entry = document.partition(f"`{fixture.name}`\n")
            entry = entry.partition("\n## ")[0].partition("\n### ")[0]
            assert fixture.checks in entry
            assert fixture.note in entry
            assert f"```text\n{fixture.board}```\n" in entry
            for move in fixture.legal.split():
                assert f"`{move}`" in entry
            for then in fixture.then:
                assert f"```text\n{then.board}```\n" in entry
                assert then.note in entry


def test_the_moves_are_grouped_under_the_piece_that_makes_them() -> None:
    _, _, entry = render_document().partition("`blocking-all-around`\n")
    assert "- c3 ▲大: `c3-c4` `c3-c4^`" in entry
    assert "- a1 ▲帥: `a1-a2` `a1-b1`\n" in entry
