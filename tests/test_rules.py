import pytest

from engine.rules import RuleSet


def test_parse_rule_with_commas() -> None:
    rule = RuleSet.from_spec("B5,6,7/S6,7,8")
    assert rule.birth == {5, 6, 7}
    assert rule.survive == {6, 7, 8}


def test_parse_rule_with_multidigit_values() -> None:
    rule = RuleSet.from_spec("B18,19,20/S18,19,20,21")
    assert rule.birth == {18, 19, 20}
    assert rule.survive == {18, 19, 20, 21}


def test_parse_rule_without_commas() -> None:
    rule = RuleSet.from_spec("B3/S23")
    assert rule.birth == {3}
    assert rule.survive == {2, 3}



def test_compact_digits_are_parsed_digit_by_digit() -> None:
    rule = RuleSet.from_spec("B12/S34")
    assert rule.birth == {1, 2}
    assert rule.survive == {3, 4}


def test_invalid_rule_raises() -> None:
    with pytest.raises(ValueError):
        RuleSet.from_spec("5,6/6,7")
