from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from emx2_dynamic_runtime import (
    COHORT_RUNTIME_TABLES,
    DEFAULT_PROFILE,
    DEFAULT_SCHEMA_CSV,
    build_runtime_registry,
    load_profile_model_rows,
    write_task_prompts_toml,
)


TASK_PREFIX = "task_"
AUTO_GENERATED_TABLES_DISABLED: set[str] = {
    "Datasets",
    "Variables",
    "Variable mappings",
    "Variable values",
    "Profiles",
}

AUTO_SKIP_COLUMNS: Dict[str, set[str]] = {
    "Resources": {
        "overview",
        "hricore",
        "id",
        "internal identifiers",
        "external identifiers",
        "design and structure",
        "subpopulations",
        "collection events",
        "population",
        "contributors",
        "organisations involved",
        "people involved",
        "contact point",
        "linkage",
        "access conditions",
        "information",
    },
    "Internal identifiers": {"resource"},
    "External identifiers": {"resource"},
    "Subpopulations": {"resource", "counts"},
    "Subpopulation counts": {"resource", "subpopulation", "age group", "N total", "N female", "N male"},
    "Collection events": {"resource"},
    "Agents": {"resource"},
    "Organisations": {
        "resource",
        "id",
        "type",
        "name",
        "organisation",
        "other organisation",
        "department",
        "website",
        "email",
        "logo",
        "role",
    },
    "Contacts": {"resource"},
    "Publications": {"resource"},
    "Documentation": {"resource"},
}

AUTO_OWNER_TASKS: Dict[str, str] = {
    "Internal identifiers": "task_overview",
    "External identifiers": "task_overview",
    "Subpopulations": "task_subpopulations",
    "Subpopulation counts": "task_subpopulations",
    "Collection events": "task_collection_events",
    "Agents": "task_contributors_org",
    "Organisations": "task_contributors_org",
    "Contacts": "task_contributors_people",
    "Publications": "task_information",
    "Documentation": "task_information",
}

RESOURCE_TASK_HINTS: List[tuple[str, tuple[str, ...]]] = [
    ("task_access_conditions", ("access", "consent", "release", "fee", "license", "legislation", "condition")),
    ("task_design_structure", ("design", "structure", "schematic", "data collection")),
    ("task_population", ("population", "participant", "sample", "country", "region", "age", "inclusion", "exclusion", "disease", "comorbidity")),
    ("task_contributors_org", ("contributor", "organisation", "organization", "publisher", "creator")),
    ("task_contributors_people", ("contact point", "contact person")),
    ("task_information", ("publication", "funding", "acknowledg", "provenance", "document", "citation", "supplement", "theme")),
    ("task_linkage", ("linkage", "linked", "linkable")),
]

INLINE_ALLOWED_VALUES_MAX = 8
INLINE_ALLOWED_VALUES_MEDIUM_MAX = 16
_BASELINE_PROFILE_TABLES: set[str] | None = None


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _as_list_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = str(item or "").strip()
            if s:
                out.append(s)
        return out
    s = str(value or "").strip()
    return [s] if s else []


def _lookup_field_meta(registry: Dict[str, Any], table_name: str, column_name: str) -> Dict[str, Any] | None:
    table_meta = registry.get("tables", {}).get(table_name, {})
    fields = table_meta.get("fields", {})
    wanted = _normalize_key(column_name)
    for candidate, meta in fields.items():
        if _normalize_key(candidate) == wanted:
            return meta
    return None


def _schema_field(
    key: str,
    table: str,
    column: str,
    *,
    note: str | None = None,
    nullable: bool = True,
) -> Dict[str, Any]:
    return {
        "kind": "field",
        "key": key,
        "table": table,
        "column": column,
        "note": note,
        "nullable": nullable,
    }


def _helper_field(
    key: str,
    placeholder: Any,
    *,
    note: str,
    table: str | None = None,
    column: str | None = None,
) -> Dict[str, Any]:
    spec = {
        "kind": "field",
        "key": key,
        "placeholder": placeholder,
        "note": note,
        "nullable": True,
    }
    if table:
        spec["table"] = table
    if column:
        spec["column"] = column
    return spec


def _list_object(
    key: str,
    item_fields: List[Dict[str, Any]],
    *,
    note: str,
) -> Dict[str, Any]:
    return {
        "kind": "list",
        "key": key,
        "note": note,
        "item_fields": item_fields,
    }


IDENTIFIER_ITEM_FIELDS = [
    _schema_field("identifier", "Internal identifiers", "identifier", note="Use the explicit identifier value exactly as written.", nullable=False),
    _schema_field("type", "Internal identifiers", "internal identifier type", note="Use the exact allowed type label only when explicitly stated."),
    _schema_field("type_other", "Internal identifiers", "internal identifier type other", note="Only if the identifier type is described explicitly but does not match an allowed label."),
]

EXTERNAL_IDENTIFIER_ITEM_FIELDS = [
    _schema_field("identifier", "External identifiers", "identifier", note="Use the explicit identifier value exactly as written.", nullable=False),
    _schema_field("type", "External identifiers", "external identifier type", note="Use the exact allowed type label only when explicitly stated."),
    _schema_field("type_other", "External identifiers", "external identifier type other", note="Only if the identifier type is described explicitly but does not match an allowed label."),
]

SUBPOPULATION_COUNT_FIELDS = [
    _schema_field("age_group", "Subpopulation counts", "age group", note="Only include an explicit age-group label for this reported count row."),
    _schema_field("n_total", "Subpopulation counts", "N total", note="Explicit total participant count for this row only."),
    _schema_field("n_female", "Subpopulation counts", "N female", note="Explicit female count for this row only."),
    _schema_field("n_male", "Subpopulation counts", "N male", note="Explicit male count for this row only."),
]

ORGANISATION_ITEM_FIELDS = [
    _schema_field("id", "Agents", "id", note="Use an explicit acronym, identifier or short name from the PDF.", nullable=False),
    _schema_field("type", "Agents", "type", note="Use the exact allowed label for an individual versus organisation only when explicit.", nullable=False),
    _schema_field("name", "Agents", "name", note="Only for explicit individual contributors."),
    _schema_field("organisation", "Agents", "organisation", note="Use the explicit organisation name as written in the PDF."),
    _schema_field("other_organisation", "Agents", "other organisation", note="Only if an alternative organisation spelling or label is explicitly given."),
    _schema_field("department", "Agents", "department", note="Only if explicitly stated."),
    _schema_field("website", "Agents", "website", note="Only if explicitly shown."),
    _schema_field("email", "Agents", "email", note="Only if explicitly shown."),
    _schema_field("logo", "Agents", "logo", note="Only if the PDF explicitly names or embeds a logo/file reference."),
    _schema_field("role", "Agents", "role", note="Return explicit contribution-role labels only; otherwise []."),
    _schema_field("is_lead_organisation", "Organisations", "is lead organisation", note="Return true or false only when lead status is explicit."),
]

PEOPLE_ITEM_FIELDS = [
    _schema_field("role", "Contacts", "role", note="Return explicit contributor-role labels only; otherwise []."),
    _schema_field("first_name", "Contacts", "first name", note="Only include a person row when a first name is explicit.", nullable=False),
    _schema_field("last_name", "Contacts", "last name", note="Only include a person row when a last name is explicit.", nullable=False),
    _schema_field("prefix", "Contacts", "prefix", note="Only if explicitly stated."),
    _schema_field("initials", "Contacts", "initials", note="Only if explicitly stated."),
    _schema_field("title", "Contacts", "title", note="Use the exact allowed title label only when explicitly stated."),
    _schema_field("organisation", "Contacts", "organisation", note="Use the explicit affiliated organisation name as written."),
    _schema_field("email", "Contacts", "email", note="Only if explicitly shown."),
    _schema_field("orcid", "Contacts", "orcid", note="Only if explicitly shown."),
    _schema_field("homepage", "Contacts", "homepage", note="Only if explicitly shown."),
    _schema_field("photo", "Contacts", "photo", note="Only if explicitly shown."),
    _schema_field("expertise", "Contacts", "expertise", note="Only if explicitly described."),
]

PUBLICATION_ITEM_FIELDS = [
    _schema_field("doi", "Publications", "doi", note="Return the canonical DOI string. If the PDF shows a DOI URL, strip the URL prefix.", nullable=False),
    _schema_field("title", "Publications", "title", note="Use the explicit publication title.", nullable=False),
    _schema_field("is_design_publication", "Publications", "is design publication", note="Return true or false only when the paper explicitly states design-publication status."),
    _helper_field("reference", "string|null", note="Only if the PDF contains an explicit formatted reference string."),
]

DOCUMENTATION_ITEM_FIELDS = [
    _schema_field("name", "Documentation", "name", note="Use the explicit document name or title.", nullable=False),
    _schema_field("type", "Documentation", "type", note="Use the exact allowed documentation type label only when explicit."),
    _schema_field("description", "Documentation", "description", note="Only if explicitly described."),
    _schema_field("url", "Documentation", "url", note="Only if explicitly shown."),
    _schema_field("file", "Documentation", "file", note="Only if the PDF explicitly provides a file or attachment label."),
]


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "task_overview": {
        "purpose": "Extract resource-level general information for the resource as a whole.",
        "scope": [
            "Use only explicit statements from the PDF.",
            "Do not infer missing metadata from the article title, journal front-matter or external sources.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not invent identifiers, names, dates, URLs or emails.",
            "For free-text fields, keep wording verbatim or near-verbatim.",
            "For choice fields, use exact labels from the PDF only when they match current schema values.",
            "For reference-like fields, return the explicit label or name from the PDF; matching is handled later in the pipeline.",
        ],
        "fields": [
            _schema_field("pid", "Resources", "pid", note="Only if the PDF explicitly states a persistent identifier."),
            _schema_field("name", "Resources", "name", note="Only if the paper explicitly identifies the resource name."),
            _schema_field("acronym", "Resources", "acronym", note="Only if the paper explicitly marks or clearly uses an acronym for the resource."),
            _schema_field("type", "Resources", "type", note="Return exact resource-type labels only when explicit."),
            _schema_field("cohort_type", "Resources", "cohort type", note="Only if the paper explicitly states a cohort subtype."),
            _schema_field("website", "Resources", "website", note="Only if it is clearly the resource website, not a publisher or DOI URL."),
            _schema_field("description", "Resources", "description", note="Use only the explicit resource description from the PDF."),
            _schema_field("keywords", "Resources", "keywords", note="Return distinctive retrieval keywords taken from the paper wording."),
            _list_object(
                "internal_identifiers",
                IDENTIFIER_ITEM_FIELDS,
                note="Output one row per explicit internal identifier. If none are explicitly stated, return [].",
            ),
            _list_object(
                "external_identifiers",
                EXTERNAL_IDENTIFIER_ITEM_FIELDS,
                note="Output one row per explicit external identifier. If none are explicitly stated, return [].",
            ),
            _schema_field("start_year", "Resources", "start year", note="Return an integer year only when explicitly tied to the resource start."),
            _schema_field("end_year", "Resources", "end year", note="Return an integer year only when explicitly tied to the resource end."),
            _schema_field("contact_email", "Resources", "contact email", note="Only if explicitly shown."),
            _schema_field("issued", "Resources", "issued", note="Only if explicitly stated as resource metadata, not article publication date."),
            _schema_field("modified", "Resources", "modified", note="Only if explicitly stated as resource metadata, not article publication date."),
        ],
    },
    "task_design_structure": {
        "purpose": "Extract the design and data-collection structure of the resource.",
        "scope": [
            "Focus only on resource-level design statements.",
            "Do not extract collection-event, subpopulation or sample details here.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not infer study design from weak hints when the paper is ambiguous.",
        ],
        "fields": [
            _schema_field("design", "Resources", "design", note="Use the explicit design label if stated."),
            _schema_field("design_description", "Resources", "design description", note="Use a short explicit design description from the paper."),
            _schema_field("data_collection_type", "Resources", "data collection type", note="Return explicit data-collection type labels only when stated."),
        ],
    },
    "task_subpopulations": {
        "purpose": "Extract explicitly defined subpopulations only.",
        "scope": [
            "Return one row per explicit arm, subgroup, cohort split or named subpopulation.",
            "Do not create subpopulations from incidental site lists, acknowledgements or generic participant descriptions.",
        ],
        "global_rules": [
            "If no explicit subpopulations are defined, return {\"subpopulations\": []}.",
            "Only include a row when the subgroup is explicitly framed as distinct in the PDF.",
        ],
        "fields": [
            _list_object(
                "subpopulations",
                [
                    _schema_field("name", "Subpopulations", "name", note="Use the explicit subgroup label from the PDF.", nullable=False),
                    _schema_field("pid", "Subpopulations", "pid", note="Only if an explicit persistent identifier is stated."),
                    _schema_field("description", "Subpopulations", "description", note="Use the explicit subgroup description.", nullable=False),
                    _schema_field("keywords", "Subpopulations", "keywords", note="Return explicit retrieval keywords for this subgroup."),
                    _schema_field("number_of_participants", "Subpopulations", "number of participants", note="Return the explicit participant count for this subgroup only."),
                    _list_object(
                        "counts",
                        SUBPOPULATION_COUNT_FIELDS,
                        note="Output one row per explicit subgroup count row. If none are explicitly reported, return [].",
                    ),
                    _schema_field("inclusion_start", "Subpopulations", "inclusion start", note="Return an explicit calendar year only."),
                    _schema_field("inclusion_end", "Subpopulations", "inclusion end", note="Return an explicit calendar year only."),
                    _schema_field("age_groups", "Subpopulations", "age groups", note="Return exact age-group labels only when explicitly stated."),
                    _schema_field("age_min", "Subpopulations", "age min", note="Return an explicit minimum age only."),
                    _schema_field("age_max", "Subpopulations", "age max", note="Return an explicit maximum age only."),
                    _schema_field("main_medical_condition", "Subpopulations", "main medical condition", note="Return explicit disease or condition labels only."),
                    _schema_field("comorbidity", "Subpopulations", "comorbidity", note="Return explicit comorbidity labels only."),
                    _schema_field("countries", "Subpopulations", "countries", note="Return countries explicitly tied to this subgroup."),
                    _schema_field("regions", "Subpopulations", "regions", note="Return regions explicitly tied to this subgroup."),
                    _schema_field("inclusion_criteria", "Subpopulations", "inclusion criteria", note="Return exact explicit inclusion-criteria category labels only."),
                    _schema_field("other_inclusion_criteria", "Subpopulations", "other inclusion criteria", note="Copy explicit inclusion wording not captured by the category labels."),
                    _schema_field("exclusion_criteria", "Subpopulations", "exclusion criteria", note="Return exact explicit exclusion-criteria category labels only."),
                    _schema_field("other_exclusion_criteria", "Subpopulations", "other exclusion criteria", note="Copy explicit exclusion wording not captured by the category labels."),
                    _schema_field("issued", "Subpopulations", "issued", note="Only if explicitly stated as subpopulation metadata."),
                    _schema_field("modified", "Subpopulations", "modified", note="Only if explicitly stated as subpopulation metadata."),
                    _schema_field("theme", "Subpopulations", "theme", note="Return explicit theme labels only."),
                    _schema_field("access_rights", "Subpopulations", "access rights", note="Return the explicit access-rights label only."),
                    _schema_field("applicable_legislation", "Subpopulations", "applicable legislation", note="Return explicit legislation labels only."),
                ],
                note="Return one object per explicit subpopulation. If none are explicit, return [].",
            ),
        ],
    },
    "task_collection_events": {
        "purpose": "Extract explicit collection events or data-collection waves.",
        "scope": [
            "Return one object per explicit collection moment, wave, round or visit.",
            "Do not create events from generic study timelines unless they are clearly data-collection events.",
        ],
        "global_rules": [
            "If no explicit collection events are described, return {\"collection_events\": []}.",
            "Do not invent event names; if unnamed, use the explicit timing wording only.",
        ],
        "fields": [
            _list_object(
                "collection_events",
                [
                    _schema_field("name", "Collection events", "name", note="Use the explicit event label or explicit timing wording.", nullable=False),
                    _schema_field("pid", "Collection events", "pid", note="Only if an explicit persistent identifier is stated."),
                    _schema_field("description", "Collection events", "description", note="Use the explicit event description.", nullable=False),
                    _schema_field("subpopulations", "Collection events", "subpopulations", note="Return explicit subpopulation names targeted by this event."),
                    _schema_field("keywords", "Collection events", "keywords", note="Return explicit retrieval keywords for this event."),
                    _schema_field("start_date", "Collection events", "start date", note="Return the explicit event start date only."),
                    _schema_field("end_date", "Collection events", "end date", note="Return the explicit event end date only."),
                    _schema_field("age_groups", "Collection events", "age groups", note="Return exact age-group labels only when explicitly stated."),
                    _schema_field("number_of_participants", "Collection events", "number of participants", note="Return the explicit participant count for this event only."),
                    _schema_field("areas_of_information", "Collection events", "areas of information", note="Return exact explicit areas-of-information labels only."),
                    _schema_field("data_categories", "Collection events", "data categories", note="Return exact explicit data-category labels only."),
                    _schema_field("sample_categories", "Collection events", "sample categories", note="Return exact explicit sample-category labels only."),
                    _schema_field("standardized_tools", "Collection events", "standardized tools", note="Return exact explicit tool labels only."),
                    _schema_field("standardized_tools_other", "Collection events", "standardized tools other", note="Only if the PDF explicitly specifies an 'other' tool."),
                    _schema_field("core_variables", "Collection events", "core variables", note="Return explicitly listed variables only."),
                    _schema_field("issued", "Collection events", "issued", note="Only if explicitly stated as event metadata."),
                    _schema_field("modified", "Collection events", "modified", note="Only if explicitly stated as event metadata."),
                    _schema_field("theme", "Collection events", "theme", note="Return explicit theme labels only."),
                    _schema_field("access_rights", "Collection events", "access rights", note="Return the explicit access-rights label only."),
                    _schema_field("applicable_legislation", "Collection events", "applicable legislation", note="Return explicit legislation labels only."),
                ],
                note="Return one object per explicit collection event. If none are explicit, return [].",
            ),
        ],
    },
    "task_population": {
        "purpose": "Extract the resource-level population description.",
        "scope": [
            "Focus on the overall resource population, not on a single subpopulation or event.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not derive counts from percentages unless the count itself is explicit.",
        ],
        "fields": [
            _schema_field("number_of_participants", "Resources", "number of participants", note="Return the explicit total participant count for the resource."),
            _schema_field("number_of_participants_with_samples", "Resources", "number of participants with samples", note="Return the explicit count only."),
            _schema_field("countries", "Resources", "countries", note="Return countries explicitly tied to the resource population."),
            _schema_field("regions", "Resources", "regions", note="Return regions explicitly tied to the resource population."),
            _schema_field("population_age_groups", "Resources", "population age groups", note="Return exact age-group labels only when explicitly stated."),
            _schema_field("age_min", "Resources", "age min", note="Return an explicit minimum age only."),
            _schema_field("age_max", "Resources", "age max", note="Return an explicit maximum age only."),
            _schema_field("inclusion_criteria", "Resources", "inclusion criteria", note="Return exact explicit inclusion-criteria category labels only."),
            _schema_field("other_inclusion_criteria", "Resources", "other inclusion criteria", note="Copy explicit inclusion wording not captured by the category labels."),
            _schema_field("exclusion_criteria", "Resources", "exclusion criteria", note="Return exact explicit exclusion-criteria category labels only."),
            _schema_field("other_exclusion_criteria", "Resources", "other exclusion criteria", note="Copy explicit exclusion wording not captured by the category labels."),
            _helper_field("population_disease", ["string"], note="Return explicit disease labels that define the resource-level population. If none are explicit, return []."),
            _helper_field("population_oncology_topology", ["string"], note="Return explicit oncology topology labels or codes only. If none are explicit, return []."),
            _helper_field("population_oncology_morphology", ["string"], note="Return explicit oncology morphology labels or codes only. If none are explicit, return []."),
        ],
    },
    "task_contributors": {
        "purpose": "Extract resource-level organisations, people and link fields between them.",
        "scope": [
            "Use only contributor information that is explicitly tied to the resource.",
            "Do not treat author affiliation lists as resource contributors unless the PDF explicitly links them to the resource.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not invent organisations, people, roles, emails or ORCIDs.",
            "For choice fields, use exact labels from the PDF only when they match current schema values.",
            "For reference-like fields, return the explicit label or name from the PDF; matching is handled later in the pipeline.",
        ],
        "fields": [
            _list_object(
                "organisations_involved",
                ORGANISATION_ITEM_FIELDS,
                note="Return one object per explicit contributing organisation or agent. If none are explicit, return [].",
            ),
            _helper_field("publisher", "string|null", note="Set only if the PDF explicitly states the publishing organisation of the resource. Return the explicit organisation name as written in the PDF. Reference matching is handled later in the pipeline.", table="Resources", column="publisher"),
            _helper_field("creator", ["string"], note="Return only organisations explicitly stated as creator, developer or maintainer of the resource. Return the explicit organisation names as written in the PDF. Reference matching is handled later in the pipeline.", table="Resources", column="creator"),
            _list_object(
                "people_involved",
                PEOPLE_ITEM_FIELDS,
                note="Return one object per explicit person contributing to the resource. If none are explicit, return [].",
            ),
            _helper_field("contact_point_first_name", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. This helper must match one person in `people_involved[]`."),
            _helper_field("contact_point_last_name", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. This helper must match one person in `people_involved[]`."),
            _helper_field("contact_point_email", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. If explicit, it should match that same person in `people_involved[]`."),
        ],
    },
    "task_contributors_org": {
        "purpose": "Extract only resource-level contributor organisations and organisation link fields.",
        "scope": [
            "Use only explicit organisation-level contributor statements from the PDF.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not invent organisations or roles.",
        ],
        "fields": [
            _list_object(
                "organisations_involved",
                ORGANISATION_ITEM_FIELDS,
                note="Return one object per explicit contributing organisation or agent. If none are explicit, return [].",
            ),
            _helper_field("publisher", "string|null", note="Set only if the PDF explicitly states the publishing organisation of the resource. Return the explicit organisation name as written in the PDF. Reference matching is handled later in the pipeline.", table="Resources", column="publisher"),
            _helper_field("creator", ["string"], note="Return only organisations explicitly stated as creator, developer or maintainer of the resource. Return the explicit organisation names as written in the PDF. Reference matching is handled later in the pipeline.", table="Resources", column="creator"),
        ],
    },
    "task_contributors_people": {
        "purpose": "Extract only resource-level people and contact-point information.",
        "scope": [
            "Use only explicit person-level contributor or contact statements from the PDF.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not invent people, titles, roles or contact details.",
        ],
        "fields": [
            _list_object(
                "people_involved",
                PEOPLE_ITEM_FIELDS,
                note="Return one object per explicit person contributing to the resource. If none are explicit, return [].",
            ),
            _helper_field("contact_point_first_name", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. This helper must match one person in `people_involved[]`."),
            _helper_field("contact_point_last_name", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. This helper must match one person in `people_involved[]`."),
            _helper_field("contact_point_email", "string|null", note="Only if the PDF explicitly identifies a primary contact person for the resource. If explicit, it should match that same person in `people_involved[]`."),
        ],
    },
    "task_samplesets": {
        "purpose": "Extract explicit sample-set groupings when the PDF names them.",
        "scope": [
            "Use only explicit sample-set or biological-material groupings from the PDF.",
        ],
        "global_rules": [
            "If no explicit samplesets are described, return {\"samplesets\": []}.",
            "Do not invent sampleset names or sample types.",
        ],
        "fields": [
            _list_object(
                "samplesets",
                [
                    _helper_field("name", "string", note="Use the explicit sampleset name or grouping phrase.",),
                    _helper_field("sample_types", ["string"], note="Return explicit biological sample types only."),
                ],
                note="Return one object per explicit sampleset. If none are explicit, return [].",
            ),
        ],
    },
    "task_areas_of_information": {
        "purpose": "Extract explicit top-level areas of information that were collected.",
        "scope": [
            "Only return areas-of-information labels when the PDF explicitly states them.",
        ],
        "global_rules": [
            "If not explicitly stated, return {\"areas_of_information\": []}.",
        ],
        "fields": [
            _schema_field("areas_of_information", "Collection events", "areas of information", note="Return exact explicit areas-of-information labels only."),
        ],
    },
    "task_linkage": {
        "purpose": "Extract resource-level linkage status and linkage options.",
        "scope": [
            "This is resource-level linkage only, not dataset-level or subpopulation-level linkage.",
        ],
        "global_rules": [
            "Use null for missing scalar fields.",
            "Do not assume linkage just because registries, identifiers or external data sources are mentioned.",
        ],
        "fields": [
            _helper_field("prelinked", "boolean|null", note="Return true only if the resource is explicitly described as already linked to other data sources; false only if explicitly not linked."),
            _schema_field("linkage_options", "Resources", "linkage options", note="Describe explicit linkable or linked external resources in short verbatim wording."),
            _helper_field("linkage_possibility", "boolean|null", note="Return true only if the resource is explicitly described as linkable to other data sources; false only if explicitly stated impossible."),
        ],
    },
    "task_access_conditions": {
        "purpose": "Extract resource-level access and consent information.",
        "scope": [
            "Use only access-condition statements explicitly tied to the resource or its data.",
            "Do not infer access policy from generic ethics language or article publication status.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "For choice fields, use exact labels from the PDF only when they match current schema values.",
        ],
        "fields": [
            _schema_field("informed_consent_type", "Resources", "informed consent type", note="Return the explicit consent-type label only."),
            _schema_field("access_rights", "Resources", "access rights", note="Return the explicit access-rights label only."),
            _schema_field("data_access_conditions", "Resources", "data access conditions", note="Return exact explicit data-access-condition labels only."),
            _schema_field("data_use_conditions", "Resources", "data use conditions", note="Return exact explicit data-use-condition labels only."),
            _schema_field("data_access_conditions_description", "Resources", "data access conditions description", note="Use short explicit wording that describes data access or data use conditions."),
            _schema_field("data_access_fee", "Resources", "data access fee", note="Return true or false only when a fee statement is explicit."),
            _schema_field("release_type", "Resources", "release type", note="Return the explicit release-type label only."),
            _schema_field("release_description", "Resources", "release description", note="Use the explicit wording that describes the release cycle."),
        ],
    },
    "task_information": {
        "purpose": "Extract publications, documentation and other resource-level information.",
        "scope": [
            "Use only information explicitly tied to the resource.",
            "Do not treat journal front-matter or generic publisher text as resource metadata.",
        ],
        "global_rules": [
            "Use null for missing scalar fields and [] for missing lists.",
            "Do not invent DOIs, titles, URLs, file names or references.",
            "For choice fields, use exact labels from the PDF only when they match current schema values.",
        ],
        "fields": [
            _list_object(
                "publications",
                PUBLICATION_ITEM_FIELDS,
                note="Return one object per explicit publication about the resource. If none are explicit, return [].",
            ),
            _schema_field("funding_statement", "Resources", "funding statement", note="Use the explicit funding statement only."),
            _helper_field("citation_requirements", "string|null", note="Only if the PDF explicitly states how resource data should be cited or acknowledged."),
            _schema_field("acknowledgements", "Resources", "acknowledgements", note="Use the explicit acknowledgement text only."),
            _schema_field("provenance_statement", "Resources", "provenance statement", note="Use the explicit provenance statement only."),
            _list_object(
                "documentation",
                DOCUMENTATION_ITEM_FIELDS,
                note="Return one object per explicit documentation item for the resource. If none are explicit, return [].",
            ),
            _helper_field("supplementary_information", "string|null", note="Only if the PDF explicitly includes supplementary or other information relevant to the resource."),
            _schema_field("theme", "Resources", "theme", note="Return explicit theme labels only."),
            _schema_field("applicable_legislation", "Resources", "applicable legislation", note="Return explicit legislation labels only."),
        ],
    },
}


def _placeholder_from_meta(meta: Dict[str, Any], *, nullable: bool = True) -> Any:
    column_type = str(meta.get("column_type") or "").strip()
    if column_type in {"ontology_array", "ref_array", "string_array"}:
        return ["string"]
    if column_type in {"int", "non_negative_int"}:
        return "integer|null" if nullable else "integer"
    if column_type == "bool":
        return "boolean|null" if nullable else "boolean"
    if column_type in {"date", "datetime"}:
        return "string|null" if nullable else "string"
    if column_type in {"ontology", "ref", "text", "email", "hyperlink", "file"}:
        return "string|null" if nullable else "string"
    return "string|null" if nullable else "string"


def _build_template_node(spec: Dict[str, Any], registry: Dict[str, Any]) -> Any:
    if spec.get("kind") == "list":
        item: Dict[str, Any] = {}
        for child in spec.get("item_fields", []):
            item[child["key"]] = _build_template_node(child, registry)
        return [item]

    if "placeholder" in spec:
        return spec["placeholder"]

    meta = _lookup_field_meta(registry, spec["table"], spec["column"])
    if not meta:
        return "string|null" if spec.get("nullable", True) else "string"
    return _placeholder_from_meta(meta, nullable=bool(spec.get("nullable", True)))


def _render_template(specs: List[Dict[str, Any]], registry: Dict[str, Any]) -> str:
    payload: Dict[str, Any] = {}
    for spec in specs:
        payload[spec["key"]] = _build_template_node(spec, registry)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _generic_rule_from_meta(meta: Dict[str, Any], *, nullable: bool = True) -> str:
    column_type = str(meta.get("column_type") or "").strip()
    if column_type == "ontology":
        return "Use the exact explicit label from the PDF only when it exactly matches a current schema value. If not explicit or not an exact match -> null."
    if column_type == "ontology_array":
        return "Return exact explicit labels from the PDF only when they exactly match current schema values. If none are explicit or none match exactly -> []."
    if column_type == "ref":
        return "Use the explicit label or name from the PDF only. Reference matching is handled later in the pipeline. If not explicit -> null."
    if column_type == "ref_array":
        return "Return explicit labels or names from the PDF only. Reference matching is handled later in the pipeline. If none are explicit -> []."
    if column_type == "string_array":
        return "Return a list of explicit values only. If not explicit -> []."
    if column_type in {"int", "non_negative_int"}:
        return "Return an integer only when explicitly stated. If not explicit -> null."
    if column_type == "bool":
        return "Return true or false only when explicitly stated. If not explicit -> null."
    if column_type in {"date", "datetime"}:
        return "Return the explicit date or datetime string from the PDF. Do not infer. If not explicit -> null."
    if column_type in {"text", "email", "hyperlink", "file"}:
        return "Use the explicit value from the PDF only. If not explicit -> null."
    if not nullable:
        return "Only include a row when this field is explicit."
    return "Use the explicit value from the PDF only. If not explicit -> null."


def _allowed_values_note(meta: Dict[str, Any]) -> str:
    column_type = str(meta.get("column_type") or "").strip()
    if column_type not in {"ontology", "ontology_array"}:
        return ""

    values = [str(v or "").strip() for v in list(meta.get("allowed_values") or [])]
    values = [v for v in values if v]
    if not values:
        return ""

    ref_table = str(meta.get("ref_table") or "").strip()
    label = f"`{ref_table}.csv`" if ref_table else "the current schema list"
    count = len(values)

    if count <= INLINE_ALLOWED_VALUES_MAX:
        return f"Current schema values: {', '.join(values)}."
    if count <= INLINE_ALLOWED_VALUES_MEDIUM_MAX:
        return f"Current schema values ({count}): {', '.join(values)}."
    return f"Use only exact values from {label} ({count} current values)."


def _auto_allowed_values_note(meta: Dict[str, Any]) -> str:
    column_type = str(meta.get("column_type") or "").strip()
    if column_type not in {"ontology", "ontology_array"}:
        return ""

    values = [str(v or "").strip() for v in list(meta.get("allowed_values") or [])]
    values = [v for v in values if v]
    if not values:
        return ""

    ref_table = str(meta.get("ref_table") or "").strip()
    label = f"`{ref_table}.csv`" if ref_table else "the current schema list"
    count = len(values)

    if count <= INLINE_ALLOWED_VALUES_MAX:
        return f"Allowed values (exact strings only): {', '.join(values)}."
    if count <= INLINE_ALLOWED_VALUES_MEDIUM_MAX:
        return f"Allowed values (exact strings only, {count}): {', '.join(values)}."
    return f"Validate against current {label} ({count} allowed values)."


def _auto_generated_prompt_context(table_name: str, fields_meta: Dict[str, Any]) -> str:
    norm_fields = {_normalize_key(name) for name in fields_meta}
    if {"mainmedicalcondition", "comorbidity", "numberofparticipants", "countries", "regions"} & norm_fields:
        return "subgroup_like"
    if {"areasofinformation", "datacategories", "samplecategories", "standardizedtools"} & norm_fields:
        return "event_like"
    if table_name == "Resources":
        return "resource_like"
    return "generic"


def _reused_auto_generated_rules(context: str, field_key: str) -> List[str]:
    subgroup_rules = {
        "mainmedicalcondition": [
            "List explicit disease groups, codes or named conditions only; no code inference.",
        ],
        "countries": [
            'Only if explicitly tied to this subgroup or pathway group.',
            'Normalize "The Netherlands" -> "Netherlands".',
            "Do NOT inherit study-level countries unless the row itself is explicitly geographically defined.",
            "If none are explicit -> [].",
        ],
        "regions": [
            "Only if explicitly tied to this subgroup or pathway group.",
            "Do NOT inherit study-level regions unless the row itself is explicitly geographically defined.",
            "If none are explicit -> [].",
        ],
        "accessrights": [
            "Do NOT copy study-level access statements into this row unless the paper explicitly applies them to that subgroup or to all subgroup data.",
            '"Open access" ONLY if explicitly public.',
            '"Restricted access" ONLY if explicitly controlled/restricted (including "available on request").',
            'Otherwise default "Non public".',
            'IMPORTANT: article license text (e.g., "Open access article", CC-BY) is NOT evidence for dataset access_rights.',
        ],
        "applicablelegislation": [
            'Add labels ONLY if explicitly stated in the PDF as applicable or mandating legislation.',
            "Otherwise -> [].",
            "Do NOT auto-add defaults.",
        ],
    }
    resource_rules = {
        "accessrights": [
            '"Open access" ONLY if explicitly stated data are publicly accessible without restrictions.',
            '"Restricted access" ONLY if explicitly stated access requires approval/application/DAC/contract.',
            '"Non public" ONLY if explicitly stated as not publicly accessible or closed.',
            'Statements like "data available on/upon request" or "available from corresponding author on reasonable request" count as "Restricted access".',
            "If not explicitly stated -> null.",
            "NOTE: The UI may default to non-public, but extraction must not default; only extract explicit statements.",
            'IMPORTANT: publication status text (e.g., "Open access article", CC-BY license for the paper) is NOT evidence for resource data access_rights.',
        ],
        "applicablelegislation": [
            'Add labels ONLY if explicitly stated in the PDF as applicable or mandating legislation.',
            "Otherwise -> [].",
            "Do NOT auto-add defaults.",
        ],
        "countries": [
            "Include ONLY countries explicitly stated as origin of data/samples for this resource.",
            'Normalize "The Netherlands" -> "Netherlands".',
            "If none are explicit -> [].",
        ],
        "regions": [
            "Include ONLY explicitly stated geographical regions/provinces/cities relevant to where data/samples originate.",
            "Use verbatim wording (no harmonization beyond obvious Netherlands normalization above).",
            "If none are explicit -> [].",
        ],
    }
    event_rules = {
        "accessrights": [
            '"Open access" ONLY if data are explicitly publicly available.',
            '"Restricted access" ONLY if explicitly stated controlled/restricted.',
            'Otherwise default "Non public".',
            'IMPORTANT: article-level "Open access" publication status is NOT evidence that event-level data are open.',
        ],
        "applicablelegislation": [
            'Add labels ONLY if explicitly named as applicable to this event.',
            "Otherwise -> [].",
            "Do NOT auto-add defaults.",
            "Do NOT include ethics guidelines.",
        ],
    }
    by_context = {
        "subgroup_like": subgroup_rules,
        "resource_like": resource_rules,
        "event_like": event_rules,
    }
    return list(by_context.get(context, {}).get(field_key, []))


def _render_reused_auto_generated_field_lines(
    full_key: str,
    spec: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    table_name: str,
    fields_meta: Dict[str, Any],
) -> List[str]:
    context = _auto_generated_prompt_context(table_name, fields_meta)
    field_key = _normalize_key(spec.get("key"))
    rules = _reused_auto_generated_rules(context, field_key)
    if not rules:
        return []

    lines = [f"- `{full_key}`: {str(spec.get('note') or '').strip()}"]
    allowed_note = _auto_allowed_values_note(meta)
    if allowed_note:
        lines.append(f"  {allowed_note}")
    lines.append("  Rules:")
    for rule in rules:
        lines.append(f"  - {rule}")
    if field_key not in {"accessrights", "applicablelegislation"}:
        lines.append(f"  - {_generic_rule_from_meta(meta, nullable=bool(spec.get('nullable', True)))}")
    return lines


def _field_reference(spec: Dict[str, Any]) -> str:
    if "placeholder" in spec or spec.get("kind") == "list":
        return ""
    return f"{spec['table']}.{spec['column']}"


def _render_field_lines(spec: Dict[str, Any], registry: Dict[str, Any], prefix: str = "") -> List[str]:
    key = spec["key"]
    full_key = f"{prefix}{key}"
    if spec.get("kind") == "list":
        lines = [f"- `{full_key}[]`: {spec.get('note') or 'Return a list of explicit rows only.'}"]
        for child in spec.get("item_fields", []):
            child_prefix = f"{full_key}[]."
            for child_line in _render_field_lines(child, registry, prefix=child_prefix):
                lines.append(f"  {child_line}")
        return lines

    note = str(spec.get("note") or "").strip()
    if "placeholder" in spec:
        return [f"- `{full_key}`: {note}"]

    meta = _lookup_field_meta(registry, spec["table"], spec["column"]) or {}
    parts: List[str] = []
    if note:
        parts.append(note)
    desc = str(meta.get("description") or "").strip()
    if desc:
        parts.append(f"Schema meaning: {desc}")
    allowed_note = _allowed_values_note(meta)
    if allowed_note:
        parts.append(allowed_note)
    parts.append(_generic_rule_from_meta(meta, nullable=bool(spec.get("nullable", True))))
    ref = _field_reference(spec)
    if ref:
        parts.append(f"Source: {ref}")
    return [f"- `{full_key}`: {' '.join(parts)}"]


def _render_auto_generated_field_lines(
    spec: Dict[str, Any],
    registry: Dict[str, Any],
    prefix: str = "",
    *,
    table_name: str = "",
    fields_meta: Dict[str, Any] | None = None,
) -> List[str]:
    key = spec["key"]
    full_key = f"{prefix}{key}"
    if spec.get("kind") == "list":
        lines = [f"- `{full_key}[]`: {spec.get('note') or 'Return a list of explicit rows only.'}"]
        for child in spec.get("item_fields", []):
            child_prefix = f"{full_key}[]."
            for child_line in _render_auto_generated_field_lines(
                child,
                registry,
                prefix=child_prefix,
                table_name=table_name,
                fields_meta=fields_meta,
            ):
                lines.append(f"  {child_line}")
        return lines

    note = str(spec.get("note") or "").strip()
    if "placeholder" in spec:
        return [f"- `{full_key}`: {note}"]

    meta = _lookup_field_meta(registry, spec["table"], spec["column"]) or {}
    reused = _render_reused_auto_generated_field_lines(
        full_key,
        spec,
        meta,
        table_name=table_name or str(spec.get("table") or "").strip(),
        fields_meta=fields_meta or {},
    )
    if reused:
        return reused
    parts: List[str] = []
    if note:
        parts.append(note)
    allowed_note = _auto_allowed_values_note(meta)
    if allowed_note:
        parts.append(allowed_note)
    parts.append(_generic_rule_from_meta(meta, nullable=bool(spec.get("nullable", True))))
    return [f"- `{full_key}`: {' '.join(parts)}"]


def _spec_coverage(spec: Dict[str, Any], out: Dict[str, set[str]]) -> None:
    table = str(spec.get("table") or "").strip()
    column = str(spec.get("column") or "").strip()
    if "placeholder" in spec:
        if table and column:
            out.setdefault(table, set()).add(column)
        return
    if spec.get("kind") == "list":
        for child in spec.get("item_fields", []):
            _spec_coverage(child, out)
        return
    out.setdefault(spec["table"], set()).add(spec["column"])


def _snake_case(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")


def _unique_key(base_key: str, existing_keys: set[str], table_name: str = "") -> str:
    key = base_key or "field"
    if key not in existing_keys:
        return key
    if table_name:
        prefixed = f"{_snake_case(table_name)}_{key}"
        if prefixed not in existing_keys:
            return prefixed
    idx = 2
    while f"{key}_{idx}" in existing_keys:
        idx += 1
    return f"{key}_{idx}"


def _primary_table_for_list(spec: Dict[str, Any]) -> str:
    counts: Dict[str, int] = {}
    for child in spec.get("item_fields", []):
        table = str(child.get("table") or "").strip()
        if not table:
            continue
        counts[table] = counts.get(table, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _tables_in_spec(spec: Dict[str, Any], out: set[str]) -> None:
    table = str(spec.get("table") or "").strip()
    if table:
        out.add(table)
    if spec.get("kind") == "list":
        for child in spec.get("item_fields", []):
            _tables_in_spec(child, out)


def _auto_field_from_meta(column_name: str, meta: Dict[str, Any], existing_keys: set[str] | None = None) -> Dict[str, Any]:
    description = str(meta.get("description") or "").strip()
    table_name = str(meta.get("_table_name") or "").strip()
    key = _snake_case(column_name)
    if existing_keys is not None:
        key = _unique_key(key, existing_keys, table_name)
        existing_keys.add(key)
    return _schema_field(
        key,
        table_name,
        column_name,
        note=description,
        nullable=str(meta.get("required") or "").strip().upper() != "TRUE",
    )


def _resource_owner_task(column_name: str, meta: Dict[str, Any]) -> str:
    text = f"{column_name} {str(meta.get('description') or '')}".lower()
    for task_name, needles in RESOURCE_TASK_HINTS:
        if any(needle in text for needle in needles):
            return task_name
    return "task_overview"


def _owner_task_for_field(table_name: str, column_name: str, meta: Dict[str, Any]) -> str:
    if table_name == "Resources":
        return _resource_owner_task(column_name, meta)
    return AUTO_OWNER_TASKS.get(table_name, "task_overview")


def _auto_generated_task_name(table_name: str) -> str:
    return f"task_{_snake_case(table_name)}_auto_generated"


def _auto_generated_list_key(_table_name: str) -> str:
    return "rows"


def _baseline_profile_tables() -> set[str]:
    global _BASELINE_PROFILE_TABLES
    if _BASELINE_PROFILE_TABLES is not None:
        return _BASELINE_PROFILE_TABLES

    schema_path = (Path(__file__).resolve().parent.parent / DEFAULT_SCHEMA_CSV).resolve()
    tables: set[str] = set()
    if schema_path.is_file():
        with schema_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                profiles = {
                    part.strip()
                    for part in str(row.get("profiles") or "").split(",")
                    if part.strip()
                }
                if DEFAULT_PROFILE not in profiles:
                    continue
                table_name = str(row.get("tableName") or "").strip()
                if table_name:
                    tables.add(table_name)
    _BASELINE_PROFILE_TABLES = tables
    return tables


def _build_auto_generated_task_section(table_name: str, registry: Dict[str, Any]) -> Dict[str, Any] | None:
    table_meta = registry.get("tables", {}).get(table_name, {})
    fields_meta = table_meta.get("fields", {})
    if not fields_meta:
        return None

    field_specs: List[Dict[str, Any]] = []
    field_map: Dict[str, str] = {}
    existing_keys: set[str] = set()
    for column_name, meta in fields_meta.items():
        if column_name in AUTO_SKIP_COLUMNS.get(table_name, set()):
            continue
        if str(meta.get("column_type") or "").strip() == "refback":
            continue
        enriched_meta = dict(meta)
        enriched_meta["_table_name"] = table_name
        field_spec = _auto_field_from_meta(column_name, enriched_meta, existing_keys)
        field_specs.append(field_spec)
        field_map[str(field_spec.get("key") or "").strip()] = column_name

    if not field_specs:
        return None

    list_key = _auto_generated_list_key(table_name)
    template_specs = [
        _list_object(
            list_key,
            field_specs,
            note=(
                f"Return one object per explicit `{table_name}` row. "
                f"If none are explicit, return []."
            ),
        )
    ]
    lines = [
        "Return ONLY valid JSON matching the template. No extra text. Do NOT infer.",
        "",
        "SCOPE",
        f"- Extract ONLY explicit rows for `{table_name}`.",
        f"- Use this task only for information that belongs in `{table_name}`.",
        f"- If the PDF does not explicitly describe any `{table_name}` rows -> {{\"{list_key}\": []}}.",
        "",
        "GLOBAL RULES",
        "- Use only explicit statements from the PDF.",
        "- Use null for missing scalar fields and [] for missing lists.",
        "- Do not invent rows, identifiers, labels or relationships.",
        "- Keep wording verbatim or near-verbatim where possible.",
        "- For reference fields, return the explicit label or name from the PDF; matching happens later in the pipeline.",
        "",
        "FIELDS",
    ]
    for field_spec in template_specs:
        lines.extend(
            _render_auto_generated_field_lines(
                field_spec,
                registry,
                table_name=table_name,
                fields_meta=fields_meta,
            )
        )

    return {
        "template_json": _render_template(template_specs, registry),
        "instructions": "\n".join(lines).strip(),
        "auto_generated": True,
        "task_table": table_name,
        "task_list_key": list_key,
        "task_sheet_name": table_name,
        "task_field_map_json": json.dumps(field_map, ensure_ascii=False, sort_keys=True),
    }


def _materialize_spec(spec: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any] | None:
    if "placeholder" in spec:
        return dict(spec)

    if spec.get("kind") == "list":
        new_spec = dict(spec)
        materialized_children: List[Dict[str, Any]] = []
        for child in spec.get("item_fields", []):
            rendered = _materialize_spec(child, registry)
            if rendered is not None:
                materialized_children.append(rendered)

        represented_tables: set[str] = set()
        for child in spec.get("item_fields", []):
            _tables_in_spec(child, represented_tables)
        primary_table = _primary_table_for_list(spec)
        if primary_table:
            represented_tables.add(primary_table)

        covered_by_table: Dict[str, set[str]] = {}
        for child in materialized_children:
            _spec_coverage(child, covered_by_table)

        existing_keys = {
            str(child.get("key") or "").strip()
            for child in materialized_children
            if str(child.get("key") or "").strip()
        }
        covered_norm_names = {
            _normalize_key(str(child.get("key") or "").strip())
            for child in materialized_children
            if str(child.get("key") or "").strip()
        }
        for table_name in sorted(represented_tables):
            table_meta = registry.get("tables", {}).get(table_name, {}).get("fields", {})
            covered = covered_by_table.get(table_name, set())
            for column_name, meta in table_meta.items():
                if column_name in covered:
                    continue
                if column_name in AUTO_SKIP_COLUMNS.get(table_name, set()):
                    continue
                if str(meta.get("column_type") or "").strip() == "refback":
                    continue
                if _normalize_key(column_name) in covered_norm_names:
                    continue
                enriched_meta = dict(meta)
                enriched_meta["_table_name"] = table_name
                auto_field = _auto_field_from_meta(column_name, enriched_meta, existing_keys)
                materialized_children.append(auto_field)
                covered.add(column_name)
                covered_norm_names.add(_normalize_key(str(auto_field.get("key") or "").strip()))
            covered_by_table[table_name] = covered

        if not materialized_children:
            return None
        new_spec["item_fields"] = materialized_children
        return new_spec

    meta = _lookup_field_meta(registry, spec["table"], spec["column"])
    if meta is None:
        return None
    return dict(spec)


def build_dynamic_task_sections(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cfg: Dict[str, Dict[str, Any]] = {}

    task_fields: Dict[str, List[Dict[str, Any]]] = {}
    for task_name, spec in TASK_SPECS.items():
        fields = []
        for field_spec in spec.get("fields", []):
            materialized = _materialize_spec(field_spec, registry)
            if materialized is not None:
                fields.append(materialized)
        task_fields[task_name] = fields

    global_covered: Dict[str, set[str]] = {}
    for fields in task_fields.values():
        for field_spec in fields:
            _spec_coverage(field_spec, global_covered)

    for table_name, table_meta in (registry.get("tables") or {}).items():
        if table_name not in COHORT_RUNTIME_TABLES:
            continue
        for column_name, meta in (table_meta.get("fields") or {}).items():
            if column_name in global_covered.get(table_name, set()):
                continue
            if column_name in AUTO_SKIP_COLUMNS.get(table_name, set()):
                continue
            if str(meta.get("column_type") or "").strip() == "refback":
                continue
            owner_task = _owner_task_for_field(table_name, column_name, meta)
            if owner_task not in task_fields:
                continue
            existing_keys = {
                str(field.get("key") or "").strip()
                for field in task_fields[owner_task]
                if str(field.get("key") or "").strip()
            }
            enriched_meta = dict(meta)
            enriched_meta["_table_name"] = table_name
            task_fields[owner_task].append(_auto_field_from_meta(column_name, enriched_meta, existing_keys))
            global_covered.setdefault(table_name, set()).add(column_name)

    for task_name, spec in TASK_SPECS.items():
        fields = task_fields.get(task_name, [])
        lines = [
            "Return ONLY valid JSON matching the template. No extra text.",
            "",
            f"TASK",
            f"- {spec['purpose']}",
        ]
        for scope_line in spec.get("scope", []):
            lines.append(f"- {scope_line}")

        lines.extend([
            "",
            "GLOBAL RULES",
        ])
        for rule in spec.get("global_rules", []):
            lines.append(f"- {rule}")

        lines.extend([
            "",
            "FIELDS",
        ])
        for field_spec in fields:
            lines.extend(_render_field_lines(field_spec, registry))

        cfg[task_name] = {
            "template_json": _render_template(fields, registry),
            "instructions": "\n".join(lines).strip(),
        }

    baseline_tables = _baseline_profile_tables()
    for table_name in sorted(registry.get("tables") or {}):
        if table_name in COHORT_RUNTIME_TABLES:
            continue
        if table_name in baseline_tables:
            continue
        if table_name in AUTO_GENERATED_TABLES_DISABLED:
            continue
        section = _build_auto_generated_task_section(table_name, registry)
        if not section:
            continue
        cfg[_auto_generated_task_name(table_name)] = section

    return cfg


def replace_task_sections(cfg: Dict[str, Any], generated_tasks: Dict[str, Dict[str, Any]]) -> None:
    for key in [k for k in list(cfg.keys()) if str(k).startswith(TASK_PREFIX)]:
        cfg.pop(key, None)
    cfg.update(generated_tasks)


def _extract_default_names(meta: Dict[str, Any]) -> List[str]:
    default_value = str(meta.get("default_value") or "").strip()
    if not default_value:
        return []
    names = [
        a or b
        for a, b in re.findall(r"name\s*:\s*'([^']+)'|name\s*:\s*\"([^\"]+)\"", default_value)
    ]
    out: List[str] = []
    seen: set[str] = set()
    for name in names:
        value = str(name or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pick_allowed_label(meta: Dict[str, Any], *needles: str) -> str | None:
    values = list(meta.get("allowed_values") or [])
    if not values:
        return None
    wanted = [_normalize_key(x) for x in needles if x]
    for value in values:
        norm = _normalize_key(value)
        if any(needle in norm for needle in wanted):
            return value
    return None


def _is_health_context(per_section_results: Dict[str, Dict[str, Any]]) -> bool:
    overview = per_section_results.get("task_overview") or {}
    info = per_section_results.get("task_information") or {}
    types = [_normalize_key(x) for x in _as_list_str(overview.get("type"))]
    themes = [_normalize_key(x) for x in _as_list_str(info.get("theme"))]
    if "health" in themes:
        return True
    for value in types:
        if any(token in value for token in ("clinical", "cohort", "registry", "healthrecords", "biobank")):
            return True
    return False


def _ensure_default_list_values(item: Dict[str, Any], meta: Dict[str, Any]) -> None:
    defaults = _extract_default_names(meta)
    if defaults and not item.get("_explicit_dynamic_override"):
        current = item.get(meta.get("_task_key"))
        if not current:
            item[meta["_task_key"]] = list(defaults)


def postprocess_section_results_dynamic(
    per_section_results: Dict[str, Dict[str, Any]],
    _paper_text: str,
    registry: Dict[str, Any],
) -> None:
    paper_text = str(_paper_text or "")

    access = per_section_results.get("task_access_conditions")
    if isinstance(access, dict):
        meta = _lookup_field_meta(registry, "Resources", "access rights")
        if meta and not access.get("access_rights"):
            request_pat = r"available (?:upon|on) request|upon reasonable request|made available on request"
            if re.search(request_pat, paper_text, flags=re.IGNORECASE):
                restricted = _pick_allowed_label(meta, "restricted", "controlled")
                if restricted:
                    access["access_rights"] = restricted

    health_context = _is_health_context(per_section_results)
    if not health_context:
        return

    info = per_section_results.get("task_information")
    if isinstance(info, dict):
        theme_meta = _lookup_field_meta(registry, "Resources", "theme")
        law_meta = _lookup_field_meta(registry, "Resources", "applicable legislation")
        if theme_meta and not info.get("theme"):
            defaults = _extract_default_names(theme_meta)
            if defaults:
                info["theme"] = defaults
        if law_meta:
            defaults = _extract_default_names(law_meta)
            if defaults:
                current = _as_list_str(info.get("applicable_legislation"))
                merged = current + [x for x in defaults if x not in current]
                info["applicable_legislation"] = merged

    subpops = (per_section_results.get("task_subpopulations") or {}).get("subpopulations") or []
    sub_theme_meta = _lookup_field_meta(registry, "Subpopulations", "theme")
    sub_law_meta = _lookup_field_meta(registry, "Subpopulations", "applicable legislation")
    sub_theme_defaults = _extract_default_names(sub_theme_meta or {})
    sub_law_defaults = _extract_default_names(sub_law_meta or {})
    for item in subpops:
        if not isinstance(item, dict):
            continue
        if sub_theme_defaults and not item.get("theme"):
            item["theme"] = list(sub_theme_defaults)
        if sub_law_defaults:
            current = _as_list_str(item.get("applicable_legislation"))
            item["applicable_legislation"] = current + [x for x in sub_law_defaults if x not in current]

    events = (per_section_results.get("task_collection_events") or {}).get("collection_events") or []
    ev_theme_meta = _lookup_field_meta(registry, "Collection events", "theme")
    ev_law_meta = _lookup_field_meta(registry, "Collection events", "applicable legislation")
    ev_theme_defaults = _extract_default_names(ev_theme_meta or {})
    ev_law_defaults = _extract_default_names(ev_law_meta or {})
    for item in events:
        if not isinstance(item, dict):
            continue
        if ev_theme_defaults and not item.get("theme"):
            item["theme"] = list(ev_theme_defaults)
        if ev_law_defaults:
            current = _as_list_str(item.get("applicable_legislation"))
            item["applicable_legislation"] = current + [x for x in ev_law_defaults if x not in current]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fully dynamic cohort prompt sections from the current EMX2 schema.")
    parser.add_argument("--profile", default="UMCGCohortsStaging")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--fallback-schema-csv", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry = build_runtime_registry(
        args.profile,
        tables=COHORT_RUNTIME_TABLES,
        local_root=args.local_root,
        fallback_schema_csv=args.fallback_schema_csv,
        cache_dir=args.cache_dir,
    )
    cfg = build_dynamic_task_sections(registry)
    write_task_prompts_toml(cfg, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
