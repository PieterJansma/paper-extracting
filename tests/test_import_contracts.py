from __future__ import annotations

from main_cohort import (
    _build_local_organisation_ref_map,
    _ensure_organisation_ids,
    _normalize_or_infer_resource_types,
    _resolve_local_organisation_ref,
    _resource_row_from_sections,
    _serialize_multi_value,
    _serialize_runtime_array_value,
)


def test_serialize_multi_value_flattens_and_cleans_bracketed_tokens() -> None:
    assert _serialize_multi_value('["Data access provider"]') == "Data access provider"
    assert _serialize_multi_value('Data access provider"]') == "Data access provider"


def test_runtime_ref_array_serializer_normalizes_and_filters_to_allowed_values() -> None:
    meta = {
        "column_type": "ref_array",
        "allowed_values": ["Data access provider", "Data controller"],
    }
    value = '["data access provider", "unknown role", "Data access provider\"]"]'
    assert _serialize_runtime_array_value("Agents", "role", meta, value) == "Data access provider"


def test_generated_organisation_ids_enable_local_reference_resolution() -> None:
    issues: list[dict[str, str]] = []
    organisations = [
        {
            "type": "organisation",
            "name": "UK Medical Research Council",
            "id": "",
        }
    ]
    _ensure_organisation_ids(
        organisations,
        paper_label="alspac",
        pdf_path="data/alspac.pdf",
        issues=issues,
    )
    assert organisations[0]["id"]
    assert issues

    ref_map = _build_local_organisation_ref_map(organisations)
    resolved = _resolve_local_organisation_ref("UK Medical Research Council", ref_map)
    assert resolved == organisations[0]["id"]


def test_resource_row_contract_has_non_empty_type() -> None:
    row, resource_ref = _resource_row_from_sections(
        "oncolifes",
        "data/oncolifes.pdf",
        {
            "task_overview": {
                "name": "OncoLifeS",
                "pid": "10.1186/s12967-019-2122-x",
                "type": [],
                "keywords": ["biobank"],
            },
            "task_design_structure": {},
            "task_population": {},
            "task_contributors": {},
            "task_areas_of_information": {},
            "task_linkage": {},
            "task_access_conditions": {},
            "task_information": {},
            "task_subpopulations": {"subpopulations": []},
            "task_collection_events": {"collection_events": []},
        },
        [],
    )

    assert resource_ref == "10.1186/s12967-019-2122-x"
    assert row["type"]
    assert _normalize_or_infer_resource_types({"name": "OncoLifeS biobank"})
