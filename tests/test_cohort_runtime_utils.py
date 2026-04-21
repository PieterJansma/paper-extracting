from __future__ import annotations

from cohort_runtime_utils import (
    _as_list_str,
    _dedupe_keep_order,
    _extract_first_doi,
    _has_payload_value,
    _is_empty_value,
    _normalize_ws,
    _ordered_keys_from_template,
    _serialize_value,
)


def test_ordered_keys_from_template_returns_key_order_for_valid_json_object() -> None:
    assert _ordered_keys_from_template('{"a": 1, "b": 2, "c": 3}') == ["a", "b", "c"]


def test_ordered_keys_from_template_returns_empty_for_invalid_or_non_object_json() -> None:
    assert _ordered_keys_from_template("not-json") == []
    assert _ordered_keys_from_template('["a", "b"]') == []
    assert _ordered_keys_from_template(None) == []


def test_serialize_value_handles_none_scalars_and_json_types() -> None:
    assert _serialize_value(None) == ""
    assert _serialize_value("abc") == "abc"
    assert _serialize_value([1, "x"]) == '[1, "x"]'
    assert _serialize_value({"k": "v"}) == '{"k": "v"}'


def test_is_empty_value_detects_empty_containers_and_blank_strings() -> None:
    assert _is_empty_value(None) is True
    assert _is_empty_value("") is True
    assert _is_empty_value("  ") is True
    assert _is_empty_value([]) is True
    assert _is_empty_value({}) is True
    assert _is_empty_value([0]) is False
    assert _is_empty_value("x") is False


def test_has_payload_value_detects_nested_non_empty_values() -> None:
    assert _has_payload_value(None) is False
    assert _has_payload_value("   ") is False
    assert _has_payload_value([]) is False
    assert _has_payload_value({}) is False
    assert _has_payload_value({"a": "", "b": None}) is False
    assert _has_payload_value({"a": "", "b": {"c": "ok"}}) is True
    assert _has_payload_value(["", None, "ok"]) is True
    assert _has_payload_value(0) is True


def test_as_list_str_filters_empty_and_none_but_keeps_original_strings() -> None:
    assert _as_list_str(["a", "", None, " b "]) == ["a", " b "]
    assert _as_list_str("no-list") == []


def test_dedupe_keep_order_is_case_insensitive_and_trims_for_dedup_key() -> None:
    assert _dedupe_keep_order(["A", "a", " B ", "b", "", "C"]) == ["A", " B ", "C"]


def test_normalize_ws_normalizes_unicode_spaces_dashes_and_collapses_whitespace() -> None:
    raw = "a\u00a0b\u2013c \n\t d"
    assert _normalize_ws(raw) == "a b-c d"


def test_extract_first_doi_returns_first_match_without_trailing_punctuation() -> None:
    text = "See doi:10.1186/s12967-019-2122-x), and also 10.1000/xyz123."
    assert _extract_first_doi(text) == "10.1186/s12967-019-2122-x"
    assert _extract_first_doi("No DOI present") is None
