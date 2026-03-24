from __future__ import annotations

import os
import json
import argparse
import logging
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from llm_client import OpenAICompatibleClient
from extract_pipeline import load_pdf_text, build_context_prefix_messages

from fix_molgenis_staging_types_callable import fix_workbook
from map_countries_ontology import map_workbook_countries
from map_regions_ontology import map_workbook_regions

from main_final import (
    setup_logging,
    load_config,
    stem,
    _ordered_keys_from_template,
    _ordered_item_keys_from_template,
    _validate_list_template_keys,
    _serialize_value,
    _collect_pass_result,
    _build_resource_row,
    _build_list_rows,
    _combine_contributor_results,
    _combine_collection_event_results,
    _postprocess_section_results,
    _is_empty_value,
    _as_list_str,
)


COHORT_SHEETS: Dict[str, List[str]] = {
    "Resources": [
        "overview", "hricore", "id", "pid", "name", "acronym", "type", "cohort type", "website",
        "description", "keywords", "internal identifiers", "external identifiers", "start year",
        "end year", "contact email", "logo", "issued", "modified", "design and structure", "design",
        "design description", "design schematic", "data collection type", "subpopulations",
        "collection events", "population", "number of participants", "number of participants with samples",
        "countries", "regions", "population age groups", "age min", "age max", "inclusion criteria",
        "other inclusion criteria", "exclusion criteria", "other exclusion criteria", "contributors",
        "organisations involved", "publisher", "creator", "people involved", "contact point", "linkage",
        "linkage options", "access conditions", "informed consent type", "access rights",
        "data access conditions", "data use conditions", "data access conditions description",
        "data access fee", "release type", "release description", "information", "publications",
        "funding statement", "acknowledgements", "provenance statement", "documentation", "theme",
        "applicable legislation",
    ],
    "Subpopulations": [
        "resource", "name", "pid", "description", "keywords", "number of participants", "counts",
        "inclusion start", "inclusion end", "age groups", "age min", "age max", "main medical condition",
        "comorbidity", "countries", "regions", "inclusion criteria", "other inclusion criteria",
        "exclusion criteria", "other exclusion criteria", "issued", "modified", "theme", "access rights",
        "applicable legislation",
    ],
    "Subpopulation counts": [
        "resource", "subpopulation", "age group", "N total", "N female", "N male",
    ],
    "External identifiers": [
        "resource", "identifier", "external identifier type", "external identifier type other",
    ],
    "Internal identifiers": [
        "resource", "identifier", "internal identifier type", "internal identifier type other",
    ],
    "Collection events": [
        "resource", "name", "pid", "description", "subpopulations", "keywords", "start date", "end date",
        "age groups", "number of participants", "areas of information", "data categories",
        "sample categories", "standardized tools", "standardized tools other", "core variables", "issued",
        "modified", "theme", "access rights", "applicable legislation",
    ],
    "Datasets": [
        "resource", "name", "label", "dataset type", "unit of observation", "keywords", "description",
        "number of rows", "since version", "until version",
    ],
    "Agents": [
        "resource", "id", "type", "name", "organisation", "other organisation", "department", "website",
        "email", "logo", "role",
    ],
    "Organisations": [
        "is lead organisation",
    ],
    "Contacts": [
        "resource", "role", "first name", "last name", "statement of consent personal data", "prefix",
        "initials", "title", "organisation", "email", "orcid", "homepage", "photo", "expertise",
    ],
    "Publications": [
        "resource", "doi", "title", "is design publication",
    ],
    "Documentation": [
        "resource", "name", "type", "description", "url", "file",
    ],
}


def _display_person_name(person: Dict[str, Any]) -> str:
    parts = [
        str(person.get("first_name") or "").strip(),
        str(person.get("prefix") or "").strip(),
        str(person.get("last_name") or "").strip(),
    ]
    return " ".join(p for p in parts if p).strip()


def _resource_ref(label: str, overview: Dict[str, Any] | None) -> str:
    ov = overview or {}
    for key in ("pid", "name", "acronym"):
        val = ov.get(key)
        if not _is_empty_value(val):
            return str(val)
    return label


def _blank_row(columns: List[str]) -> Dict[str, Any]:
    return {c: "" for c in columns}


def _resolve_schema_xlsx(explicit_path: str | None = None) -> str | None:
    candidates = [
        explicit_path,
        os.environ.get("MOLGENIS_SCHEMA_XLSX"),
        os.path.join(os.getcwd(), "molgenis_UMCGCohortsStaging.xlsx"),
        os.path.join(os.path.dirname(__file__), "molgenis_UMCGCohortsStaging.xlsx"),
        "/mnt/data/molgenis_UMCGCohortsStaging.xlsx",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(str(candidate)))
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_countries_csv(explicit_path: str | None = None) -> str | None:
    candidates = [
        explicit_path,
        os.environ.get("COUNTRY_ONTOLOGY_CSV"),
        os.path.join(os.getcwd(), "Countries.csv"),
        os.path.join(os.getcwd(), "data", "ontologies", "Countries.csv"),
        "/Users/p.jansma/Downloads/Countries.csv",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(str(candidate)))
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_regions_csv(explicit_path: str | None = None) -> str | None:
    candidates = [
        explicit_path,
        os.environ.get("REGION_ONTOLOGY_CSV"),
        os.path.join(os.getcwd(), "Regions.csv"),
        os.path.join(os.getcwd(), "data", "ontologies", "Regions.csv"),
        "/Users/p.jansma/Downloads/Regions.csv",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(str(candidate)))
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _resource_row_from_sections(
    label: str,
    pdf_path: str,
    per_section_results: Dict[str, Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    row = _blank_row(COHORT_SHEETS["Resources"])
    overview = per_section_results.get("task_overview", {}) or {}
    design = per_section_results.get("task_design_structure", {}) or {}
    population = per_section_results.get("task_population", {}) or {}
    contributors = per_section_results.get("task_contributors", {}) or {}
    aoi = per_section_results.get("task_areas_of_information", {}) or {}
    linkage = per_section_results.get("task_linkage", {}) or {}
    access = per_section_results.get("task_access_conditions", {}) or {}
    info = per_section_results.get("task_information", {}) or {}
    subpops = (per_section_results.get("task_subpopulations", {}) or {}).get("subpopulations", []) or []
    collection_events = (per_section_results.get("task_collection_events", {}) or {}).get("collection_events", []) or []

    resource_ref = _resource_ref(label, overview)

    subpop_names: List[str] = []
    for idx, sp in enumerate(subpops, start=1):
        name = str(sp.get("name") or "").strip()
        if not name:
            pid = str(sp.get("pid") or "").strip()
            if pid:
                name = pid
            else:
                name = f"subpopulation_{idx}"
                issues.append({
                    "paper": label,
                    "pdf_path": pdf_path,
                    "severity": "warning",
                    "kind": "generated_subpopulation_name",
                    "message": f"Subpopulation {idx} had no explicit name; fallback name '{name}' used.",
                })
        subpop_names.append(name)

    event_names = [str(ev.get("name") or "").strip() for ev in collection_events if str(ev.get("name") or "").strip()]
    orgs = contributors.get("organisations_involved", []) or []
    people = contributors.get("people_involved", []) or []
    org_ids = [str(o.get("id") or "").strip() for o in orgs if str(o.get("id") or "").strip()]
    people_names = [_display_person_name(p) for p in people if _display_person_name(p)]

    contact_parts = [
        str(contributors.get("contact_point_first_name") or "").strip(),
        str(contributors.get("contact_point_last_name") or "").strip(),
    ]
    contact_name = " ".join(p for p in contact_parts if p).strip()
    contact_point = {
        "name": contact_name or None,
        "email": contributors.get("contact_point_email"),
    }
    if not contact_point["name"] and not contact_point["email"]:
        contact_point = {}

    internal_ids = overview.get("internal_identifiers", []) or []
    external_ids = overview.get("external_identifiers", []) or []
    publications = info.get("publications", []) or []
    documentation = info.get("documentation", []) or []

    row.update({
        "id": resource_ref,
        "pid": _serialize_value(overview.get("pid")),
        "name": str(overview.get("name") or label),
        "acronym": _serialize_value(overview.get("acronym")),
        "type": _serialize_value(overview.get("type")),
        "cohort type": _serialize_value(overview.get("cohort_type")),
        "website": _serialize_value(overview.get("website")),
        "description": _serialize_value(overview.get("description")),
        "keywords": _serialize_value(overview.get("keywords")),
        "internal identifiers": _serialize_value([x.get("identifier") for x in internal_ids if isinstance(x, dict) and x.get("identifier")]),
        "external identifiers": _serialize_value([x.get("identifier") for x in external_ids if isinstance(x, dict) and x.get("identifier")]),
        "start year": _serialize_value(overview.get("start_year")),
        "end year": _serialize_value(overview.get("end_year")),
        "contact email": _serialize_value(overview.get("contact_email") or contributors.get("contact_point_email")),
        "issued": _serialize_value(overview.get("issued")),
        "modified": _serialize_value(overview.get("modified")),
        "design": _serialize_value(design.get("design")),
        "design description": _serialize_value(design.get("design_description")),
        "data collection type": _serialize_value(design.get("data_collection_type")),
        "subpopulations": _serialize_value(subpop_names),
        "collection events": _serialize_value(event_names),
        "number of participants": _serialize_value(population.get("number_of_participants")),
        "number of participants with samples": _serialize_value(population.get("number_of_participants_with_samples")),
        "countries": _serialize_value(population.get("countries")),
        "regions": _serialize_value(population.get("regions")),
        "population age groups": _serialize_value(population.get("population_age_groups")),
        "age min": _serialize_value(population.get("age_min")),
        "age max": _serialize_value(population.get("age_max")),
        "inclusion criteria": _serialize_value(population.get("inclusion_criteria")),
        "other inclusion criteria": _serialize_value(population.get("other_inclusion_criteria")),
        "exclusion criteria": _serialize_value(population.get("exclusion_criteria")),
        "other exclusion criteria": _serialize_value(population.get("other_exclusion_criteria")),
        "organisations involved": _serialize_value(org_ids),
        "publisher": _serialize_value(contributors.get("publisher")),
        "creator": _serialize_value(contributors.get("creator")),
        "people involved": _serialize_value(people_names),
        "contact point": _serialize_value(contact_point),
        "linkage": _serialize_value({
            "prelinked": linkage.get("prelinked"),
            "linkage_possibility": linkage.get("linkage_possibility"),
        }),
        "linkage options": _serialize_value(linkage.get("linkage_options")),
        "informed consent type": _serialize_value(access.get("informed_consent_type")),
        "access rights": _serialize_value(access.get("access_rights")),
        "data access conditions": _serialize_value(access.get("data_access_conditions")),
        "data use conditions": _serialize_value(access.get("data_use_conditions")),
        "data access conditions description": _serialize_value(access.get("data_access_conditions_description")),
        "data access fee": _serialize_value(access.get("data_access_fee")),
        "release type": _serialize_value(access.get("release_type")),
        "release description": _serialize_value(access.get("release_description")),
        "publications": _serialize_value([
            p.get("doi") or p.get("title") for p in publications if isinstance(p, dict) and (p.get("doi") or p.get("title"))
        ]),
        "funding statement": _serialize_value(info.get("funding_statement")),
        "acknowledgements": _serialize_value(info.get("acknowledgements")),
        "provenance statement": _serialize_value(info.get("provenance_statement")),
        "documentation": _serialize_value([
            d.get("name") or d.get("url") for d in documentation if isinstance(d, dict) and (d.get("name") or d.get("url"))
        ]),
        "theme": _serialize_value(info.get("theme")),
        "applicable legislation": _serialize_value(info.get("applicable_legislation")),
    })
    return row, resource_ref


def _iter_subpopulation_count_rows(
    label: str,
    pdf_path: str,
    resource_ref: str,
    subpop_name: str,
    counts: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(counts, list):
        return out

    # Cohort-shaped counts already present
    cohort_like = any(isinstance(x, dict) and {"age_group", "n_total", "n_female", "n_male"} & set(x.keys()) for x in counts)
    if cohort_like:
        for c in counts:
            if not isinstance(c, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Subpopulation counts"])
            row.update({
                "resource": resource_ref,
                "subpopulation": subpop_name,
                "age group": _serialize_value(c.get("age_group")),
                "N total": _serialize_value(c.get("n_total")),
                "N female": _serialize_value(c.get("n_female")),
                "N male": _serialize_value(c.get("n_male")),
            })
            if any(str(row[k]).strip() for k in ("age group", "N total", "N female", "N male")):
                out.append(row)
        return out

    # Backward compatibility with old {year, age_group, gender, n_unique_individuals} shape
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in counts:
        if not isinstance(c, dict):
            continue
        age_group = str(c.get("age_group") or "").strip()
        year = str(c.get("year") or "").strip()
        key = (age_group, year)
        bucket = buckets.setdefault(key, {"age_group": age_group or None, "year": year or None, "n_total": None, "n_female": None, "n_male": None})
        gender = str(c.get("gender") or "").strip().lower()
        nval = c.get("n_unique_individuals")
        if gender in {"female", "women", "woman", "f"}:
            bucket["n_female"] = nval
        elif gender in {"male", "men", "man", "m"}:
            bucket["n_male"] = nval
        else:
            bucket["n_total"] = nval

    for bucket in buckets.values():
        age_group = bucket["age_group"]
        if bucket["year"]:
            age_group = f"{age_group} ({bucket['year']})" if age_group else str(bucket["year"])
            issues.append({
                "paper": label,
                "pdf_path": pdf_path,
                "severity": "warning",
                "kind": "subpopulation_count_year_collapsed",
                "message": f"Subpopulation counts for '{subpop_name}' included year information; year was preserved inside age group label.",
            })
        row = _blank_row(COHORT_SHEETS["Subpopulation counts"])
        row.update({
            "resource": resource_ref,
            "subpopulation": subpop_name,
            "age group": _serialize_value(age_group),
            "N total": _serialize_value(bucket["n_total"]),
            "N female": _serialize_value(bucket["n_female"]),
            "N male": _serialize_value(bucket["n_male"]),
        })
        if any(str(row[k]).strip() for k in ("age group", "N total", "N female", "N male")):
            out.append(row)

    return out


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run PDF extraction passes and write direct UMCG cohort workbook.")
    parser.add_argument(
        "-p", "--passes",
        nargs="+",
        default=["all"],
        help="Specify passes: A, B, C, D/D1/D2, E, F/F1/F2, G, H, X1, X2, X3, Y or 'all'.",
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        default=None,
        help="One or more PDF paths. Overrides config [pdf].path.",
    )
    parser.add_argument(
        "--paper-names",
        nargs="+",
        default=None,
        help="Optional labels (same length as --pdfs). If omitted, uses filename stem.",
    )
    parser.add_argument(
        "-o", "--output",
        default="final_result_cohort.xlsx",
        help="Output Excel filename.",
    )
    parser.add_argument(
        "--schema-xlsx",
        default=None,
        help="Optional path to molgenis_UMCGCohortsStaging.xlsx for post-write datatype normalization.",
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]
    if "F" in selected_passes:
        selected_passes.extend(["F1", "F2"])
    if "D" in selected_passes:
        selected_passes.extend(["D1", "D2"])

    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.final.toml")
    prompts_path = (os.environ.get("PDF_EXTRACT_PROMPTS") or "").strip() or None
    cfg = load_config(cfg_path, prompts_path=prompts_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main_cohort")

    llm_cfg = cfg["llm"]
    pdf_cfg = cfg["pdf"]

    pdf_paths = args.pdfs if args.pdfs else [pdf_cfg["path"]]
    if args.paper_names and len(args.paper_names) != len(pdf_paths):
        raise SystemExit("--paper-names must have same length as --pdfs")

    labels = args.paper_names if args.paper_names else [stem(p) for p in pdf_paths]

    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        api_key=llm_cfg.get("api_key", "sk-local"),
        model=llm_cfg.get("model", "numind/NuExtract-2.0-8B"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
        use_session=bool(llm_cfg.get("sticky_session", False)),
    )

    has_split_contributors = "task_contributors_org" in cfg and "task_contributors_people" in cfg
    has_split_collection_events = (
        "task_collection_events_core" in cfg
        and "task_collection_events_enrichment" in cfg
    )

    pass_defs = [
        ("A", "PASS A: Overview", "task_overview"),
        ("B", "PASS B: Design & structure", "task_design_structure"),
        ("C", "PASS C: Subpopulations", "task_subpopulations"),
    ]
    if has_split_collection_events:
        pass_defs.extend([
            ("D1", "PASS D1: Collection events (core)", "task_collection_events_core"),
            ("D2", "PASS D2: Collection events (enrichment)", "task_collection_events_enrichment"),
        ])
    else:
        pass_defs.append(("D", "PASS D: Collection events", "task_collection_events"))
    pass_defs.append(("E", "PASS E: Population", "task_population"))
    if has_split_contributors:
        pass_defs.extend([
            ("F1", "PASS F1: Contributors (organisations)", "task_contributors_org"),
            ("F2", "PASS F2: Contributors (people/contact)", "task_contributors_people"),
        ])
    else:
        pass_defs.append(("F", "PASS F: Contributors", "task_contributors"))
    pass_defs.extend([
        ("X1", "PASS X1: Datasets", "task_datasets"),
        ("X2", "PASS X2: Samplesets", "task_samplesets"),
        ("X3", "PASS X3: Areas of information", "task_areas_of_information"),
        ("Y", "PASS Y: Linkage", "task_linkage"),
        ("G", "PASS G: Access conditions", "task_access_conditions"),
        ("H", "PASS H: Information", "task_information"),
    ])

    section_orders: Dict[str, List[str]] = {}
    for _, _, section_key in pass_defs:
        task_cfg = cfg.get(section_key, {})
        section_orders[section_key] = _ordered_keys_from_template(task_cfg.get("template_json"))

    list_sections = {
        "task_subpopulations": "subpopulations",
        "task_collection_events": "collection_events",
        "task_datasets": "datasets",
        "task_samplesets": "samplesets",
    }
    list_section_orders: Dict[str, List[str]] = {}
    for section_key, list_name in list_sections.items():
        task_cfg = cfg.get(section_key, {})
        keys = _ordered_item_keys_from_template(task_cfg.get("template_json"), list_name)
        _validate_list_template_keys(section_key, list_name, keys)
        list_section_orders[section_key] = keys

    resource_section_keys: Dict[str, List[str]] = {
        "task_contributors": [
            "publisher", "creator", "contact_point_first_name", "contact_point_last_name", "contact_point_email",
        ],
        "task_information": [
            "funding_statement", "citation_requirements", "acknowledgements",
            "provenance_statement", "supplementary_information", "theme", "applicable_legislation",
        ],
    }

    resource_rows: List[Dict[str, Any]] = []
    subpopulation_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    external_identifier_rows: List[Dict[str, Any]] = []
    internal_identifier_rows: List[Dict[str, Any]] = []
    collection_event_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    organisation_extension_rows: List[Dict[str, Any]] = []
    contact_rows: List[Dict[str, Any]] = []
    publication_rows: List[Dict[str, Any]] = []
    documentation_rows: List[Dict[str, Any]] = []
    run_issues: List[Dict[str, Any]] = []

    issue_file = os.environ.get("PIPELINE_ISSUES_FILE", "").strip() or f"{args.output}.issues.json"

    core_payload_sections = {
        "task_overview",
        "task_design_structure",
        "task_population",
        "task_access_conditions",
        "task_information",
    }

    for pdf_path, label in zip(pdf_paths, labels):
        paper_text = load_pdf_text(pdf_path, max_pages=pdf_cfg.get("max_pages"))
        log.info("PDF '%s' loaded (%d chars)", pdf_path, len(paper_text))
        if not paper_text.strip():
            run_issues.append({
                "paper": label,
                "pdf_path": pdf_path,
                "severity": "error",
                "kind": "empty_pdf_text",
                "message": "No text extracted from PDF after fallback chain.",
            })

        prefix_messages: List[Dict[str, str]] | None = None
        prompt_cache_enabled = bool(llm_cfg.get("prompt_cache", False))
        chunking_enabled = bool(llm_cfg.get("chunking_enabled", True))
        long_text_threshold = int(llm_cfg.get("long_text_threshold_chars", 60000))
        chunk_size_chars = int(llm_cfg.get("chunk_size_chars", 45000))
        chunk_overlap_chars = int(llm_cfg.get("chunk_overlap_chars", 4000))
        max_chunks = llm_cfg.get("max_chunks")

        if prompt_cache_enabled:
            prefix_messages = build_context_prefix_messages(paper_text)

        per_section_results: Dict[str, Dict[str, Any]] = {}

        for pass_code, pass_label, section_key in pass_defs:
            if "ALL" not in selected_passes and pass_code.upper() not in selected_passes:
                continue

            result = _collect_pass_result(
                client,
                paper_text,
                cfg.get(section_key, {}),
                llm_cfg,
                log,
                pass_label,
                prefix_messages=prefix_messages if prompt_cache_enabled else None,
            )
            per_section_results[section_key] = result

            if section_key in core_payload_sections:
                has_payload = any(
                    not _is_empty_value(v)
                    for v in result.values()
                ) if isinstance(result, dict) else False
                if not has_payload:
                    run_issues.append({
                        "paper": label,
                        "pdf_path": pdf_path,
                        "severity": "warning",
                        "kind": "empty_core_section",
                        "section": section_key,
                        "message": f"{section_key} returned an empty payload.",
                    })

        if has_split_contributors:
            per_section_results["task_contributors"] = _combine_contributor_results(
                per_section_results.get("task_contributors_org"),
                per_section_results.get("task_contributors_people"),
            )

        if has_split_collection_events:
            per_section_results["task_collection_events"] = _combine_collection_event_results(
                per_section_results.get("task_collection_events_core"),
                per_section_results.get("task_collection_events_enrichment"),
            )

        _postprocess_section_results(per_section_results, paper_text)

        # keep old flat resource row for diagnostics/order stability if needed
        flat_resource_row, flat_columns = _build_resource_row(
            label,
            pdf_path,
            {
                k: v for k, v in per_section_results.items()
                if k in {
                    "task_overview",
                    "task_design_structure",
                    "task_population",
                    "task_contributors",
                    "task_areas_of_information",
                    "task_linkage",
                    "task_access_conditions",
                    "task_information",
                }
            },
            section_orders,
            section_keys_override=resource_section_keys,
        )
        # Store flattened row inside issues only for debugging when needed
        if not any(str(v).strip() for k, v in flat_resource_row.items() if k not in {"paper", "pdf_path"}):
            run_issues.append({
                "paper": label,
                "pdf_path": pdf_path,
                "severity": "warning",
                "kind": "empty_resource_row",
                "message": "Combined resource row has no payload after extraction.",
            })

        resource_row, resource_ref = _resource_row_from_sections(label, pdf_path, per_section_results, run_issues)
        resource_rows.append(resource_row)

        overview = per_section_results.get("task_overview", {}) or {}
        internal_ids = overview.get("internal_identifiers", []) or []
        external_ids = overview.get("external_identifiers", []) or []

        for item in internal_ids:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Internal identifiers"])
            row.update({
                "resource": resource_ref,
                "identifier": _serialize_value(item.get("identifier")),
                "internal identifier type": _serialize_value(item.get("type")),
                "internal identifier type other": _serialize_value(item.get("type_other")),
            })
            if any(str(row[c]).strip() for c in row):
                internal_identifier_rows.append(row)

        for item in external_ids:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["External identifiers"])
            row.update({
                "resource": resource_ref,
                "identifier": _serialize_value(item.get("identifier")),
                "external identifier type": _serialize_value(item.get("type")),
                "external identifier type other": _serialize_value(item.get("type_other")),
            })
            if any(str(row[c]).strip() for c in row):
                external_identifier_rows.append(row)

        subpops = (per_section_results.get("task_subpopulations", {}) or {}).get("subpopulations", []) or []
        for idx, sub in enumerate(subpops, start=1):
            if not isinstance(sub, dict):
                continue
            sub_name = str(sub.get("name") or "").strip()
            if not sub_name:
                sub_name = str(sub.get("pid") or "").strip() or f"subpopulation_{idx}"
                run_issues.append({
                    "paper": label,
                    "pdf_path": pdf_path,
                    "severity": "warning",
                    "kind": "generated_subpopulation_name",
                    "message": f"Subpopulation {idx} had no explicit name; fallback name '{sub_name}' used.",
                })

            row = _blank_row(COHORT_SHEETS["Subpopulations"])
            row.update({
                "resource": resource_ref,
                "name": sub_name,
                "pid": _serialize_value(sub.get("pid")),
                "description": _serialize_value(sub.get("description")),
                "keywords": _serialize_value(sub.get("keywords")),
                "number of participants": _serialize_value(sub.get("number_of_participants")),
                "counts": _serialize_value(sub.get("counts")),
                "inclusion start": _serialize_value(sub.get("inclusion_start")),
                "inclusion end": _serialize_value(sub.get("inclusion_end")),
                "age groups": _serialize_value(sub.get("age_groups")),
                "age min": _serialize_value(sub.get("age_min")),
                "age max": _serialize_value(sub.get("age_max")),
                "main medical condition": _serialize_value(sub.get("main_medical_condition")),
                "comorbidity": _serialize_value(sub.get("comorbidity")),
                "countries": _serialize_value(sub.get("countries")),
                "regions": _serialize_value(sub.get("regions")),
                "inclusion criteria": _serialize_value(sub.get("inclusion_criteria")),
                "other inclusion criteria": _serialize_value(sub.get("other_inclusion_criteria")),
                "exclusion criteria": _serialize_value(sub.get("exclusion_criteria")),
                "other exclusion criteria": _serialize_value(sub.get("other_exclusion_criteria")),
                "issued": _serialize_value(sub.get("issued")),
                "modified": _serialize_value(sub.get("modified")),
                "theme": _serialize_value(sub.get("theme")),
                "access rights": _serialize_value(sub.get("access_rights")),
                "applicable legislation": _serialize_value(sub.get("applicable_legislation")),
            })
            subpopulation_rows.append(row)

            count_rows.extend(
                _iter_subpopulation_count_rows(
                    label,
                    pdf_path,
                    resource_ref,
                    sub_name,
                    sub.get("counts", []) or [],
                    run_issues,
                )
            )

        collection_events = (per_section_results.get("task_collection_events", {}) or {}).get("collection_events", []) or []
        for item in collection_events:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Collection events"])
            row.update({
                "resource": resource_ref,
                "name": _serialize_value(item.get("name")),
                "pid": _serialize_value(item.get("pid")),
                "description": _serialize_value(item.get("description")),
                "subpopulations": _serialize_value(item.get("subpopulations")),
                "keywords": _serialize_value(item.get("keywords")),
                "start date": _serialize_value(item.get("start_date")),
                "end date": _serialize_value(item.get("end_date")),
                "age groups": _serialize_value(item.get("age_groups")),
                "number of participants": _serialize_value(item.get("number_of_participants")),
                "areas of information": _serialize_value(item.get("areas_of_information")),
                "data categories": _serialize_value(item.get("data_categories")),
                "sample categories": _serialize_value(item.get("sample_categories")),
                "standardized tools": _serialize_value(item.get("standardized_tools")),
                "standardized tools other": _serialize_value(item.get("standardized_tools_other")),
                "core variables": _serialize_value(item.get("core_variables")),
                "issued": _serialize_value(item.get("issued")),
                "modified": _serialize_value(item.get("modified")),
                "theme": _serialize_value(item.get("theme")),
                "access rights": _serialize_value(item.get("access_rights")),
                "applicable legislation": _serialize_value(item.get("applicable_legislation")),
            })
            collection_event_rows.append(row)

        datasets = (per_section_results.get("task_datasets", {}) or {}).get("datasets", []) or []
        for item in datasets:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Datasets"])
            row.update({
                "resource": resource_ref,
                "name": _serialize_value(item.get("name")),
                "label": _serialize_value(item.get("label")),
                "dataset type": _serialize_value(item.get("dataset_type")),
                "unit of observation": _serialize_value(item.get("unit_of_observation")),
                "keywords": _serialize_value(item.get("keywords")),
                "description": _serialize_value(item.get("description")),
                "number of rows": _serialize_value(item.get("number_of_rows")),
                "since version": _serialize_value(item.get("since_version")),
                "until version": _serialize_value(item.get("until_version")),
            })
            dataset_rows.append(row)

        contributors = per_section_results.get("task_contributors", {}) or {}
        organisations = contributors.get("organisations_involved", []) or []
        people = contributors.get("people_involved", []) or []

        for item in organisations:
            if not isinstance(item, dict):
                continue
            agent_row = _blank_row(COHORT_SHEETS["Agents"])
            agent_row.update({
                "resource": resource_ref,
                "id": _serialize_value(item.get("id")),
                "type": _serialize_value(item.get("type")),
                "name": _serialize_value(item.get("name")),
                "organisation": _serialize_value(item.get("organisation")),
                "other organisation": _serialize_value(item.get("other_organisation")),
                "department": _serialize_value(item.get("department")),
                "website": _serialize_value(item.get("website")),
                "email": _serialize_value(item.get("email")),
                "logo": _serialize_value(item.get("logo")),
                "role": _serialize_value(item.get("role")),
            })
            agent_rows.append(agent_row)

            org_row = _blank_row(COHORT_SHEETS["Organisations"])
            org_row["is lead organisation"] = _serialize_value(item.get("is_lead_organisation"))
            organisation_extension_rows.append(org_row)

        for item in people:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Contacts"])
            row.update({
                "resource": resource_ref,
                "role": _serialize_value(item.get("role")),
                "first name": _serialize_value(item.get("first_name")),
                "last name": _serialize_value(item.get("last_name")),
                "prefix": _serialize_value(item.get("prefix")),
                "initials": _serialize_value(item.get("initials")),
                "title": _serialize_value(item.get("title")),
                "organisation": _serialize_value(item.get("organisation")),
                "email": _serialize_value(item.get("email")),
                "orcid": _serialize_value(item.get("orcid")),
                "homepage": _serialize_value(item.get("homepage")),
                "photo": _serialize_value(item.get("photo")),
                "expertise": _serialize_value(item.get("expertise")),
            })
            contact_rows.append(row)

        info_result = per_section_results.get("task_information", {}) or {}
        publications = info_result.get("publications", []) or []
        documentation = info_result.get("documentation", []) or []

        for item in publications:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Publications"])
            row.update({
                "resource": resource_ref,
                "doi": _serialize_value(item.get("doi")),
                "title": _serialize_value(item.get("title")),
                "is design publication": _serialize_value(item.get("is_design_publication")),
            })
            publication_rows.append(row)

        for item in documentation:
            if not isinstance(item, dict):
                continue
            row = _blank_row(COHORT_SHEETS["Documentation"])
            row.update({
                "resource": resource_ref,
                "name": _serialize_value(item.get("name")),
                "type": _serialize_value(item.get("type")),
                "description": _serialize_value(item.get("description")),
                "url": _serialize_value(item.get("url")),
                "file": _serialize_value(item.get("file")),
            })
            documentation_rows.append(row)

    log.info("--- DONE EXTRACTING ---")

    if importlib.util.find_spec("xlsxwriter") is not None:
        excel_engine = "xlsxwriter"
    elif importlib.util.find_spec("openpyxl") is not None:
        excel_engine = "openpyxl"
    else:
        raise SystemExit(
            "No Excel writer engine found. Install one in the active venv, e.g.:\n"
            "  .venv/bin/python -m pip install xlsxwriter"
        )

    log.info("Writing cohort Excel with engine=%s", excel_engine)

    frames = {
        "Resources": pd.DataFrame(resource_rows, columns=COHORT_SHEETS["Resources"]),
        "Subpopulations": pd.DataFrame(subpopulation_rows, columns=COHORT_SHEETS["Subpopulations"]),
        "Subpopulation counts": pd.DataFrame(count_rows, columns=COHORT_SHEETS["Subpopulation counts"]),
        "External identifiers": pd.DataFrame(external_identifier_rows, columns=COHORT_SHEETS["External identifiers"]),
        "Internal identifiers": pd.DataFrame(internal_identifier_rows, columns=COHORT_SHEETS["Internal identifiers"]),
        "Collection events": pd.DataFrame(collection_event_rows, columns=COHORT_SHEETS["Collection events"]),
        "Datasets": pd.DataFrame(dataset_rows, columns=COHORT_SHEETS["Datasets"]),
        "Agents": pd.DataFrame(agent_rows, columns=COHORT_SHEETS["Agents"]),
        "Organisations": pd.DataFrame(organisation_extension_rows, columns=COHORT_SHEETS["Organisations"]),
        "Contacts": pd.DataFrame(contact_rows, columns=COHORT_SHEETS["Contacts"]),
        "Publications": pd.DataFrame(publication_rows, columns=COHORT_SHEETS["Publications"]),
        "Documentation": pd.DataFrame(documentation_rows, columns=COHORT_SHEETS["Documentation"]),
    }

    with pd.ExcelWriter(args.output, engine=excel_engine) as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    countries_csv = _resolve_countries_csv()
    if countries_csv:
        try:
            llm_fallback_enabled = str(os.environ.get("COUNTRY_MAPPING_LLM_FALLBACK", "1")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            countries_issues = Path(f"{args.output}.countries_issues.json")
            map_stats = map_workbook_countries(
                workbook_path=Path(args.output),
                ontology_csv=Path(countries_csv),
                output_path=Path(args.output),
                issues_json=countries_issues,
                llm_client=client if llm_fallback_enabled else None,
            )
            run_issues.append(
                {
                    "paper": "__run__",
                    "pdf_path": "",
                    "severity": "info",
                    "kind": "countries_mapping_applied",
                    "message": (
                        f"Applied country mapping using {countries_csv}; "
                        f"mapped_cells={map_stats.get('mapped_cells', 0)}, "
                        f"issue_count={map_stats.get('issue_count', 0)}, "
                        f"llm_fallback={'on' if llm_fallback_enabled else 'off'}"
                    ),
                }
            )
            log.info("Applied country mapping using %s (mapped_cells=%s)", countries_csv, map_stats.get("mapped_cells", 0))
        except Exception as e:
            log.warning("Could not apply country mapping with %s: %s", countries_csv, e)
    else:
        log.info("No Countries.csv found; skipping country mapping.")

    regions_csv = _resolve_regions_csv()
    if regions_csv:
        try:
            llm_fallback_enabled = str(os.environ.get("REGION_MAPPING_LLM_FALLBACK", "1")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            regions_issues = Path(f"{args.output}.regions_issues.json")
            map_stats = map_workbook_regions(
                workbook_path=Path(args.output),
                ontology_csv=Path(regions_csv),
                output_path=Path(args.output),
                issues_json=regions_issues,
                llm_client=client if llm_fallback_enabled else None,
            )
            run_issues.append(
                {
                    "paper": "__run__",
                    "pdf_path": "",
                    "severity": "info",
                    "kind": "regions_mapping_applied",
                    "message": (
                        f"Applied region mapping using {regions_csv}; "
                        f"mapped_cells={map_stats.get('mapped_cells', 0)}, "
                        f"issue_count={map_stats.get('issue_count', 0)}, "
                        f"llm_fallback={'on' if llm_fallback_enabled else 'off'}"
                    ),
                }
            )
            log.info("Applied region mapping using %s (mapped_cells=%s)", regions_csv, map_stats.get("mapped_cells", 0))
        except Exception as e:
            log.warning("Could not apply region mapping with %s: %s", regions_csv, e)
    else:
        log.info("No Regions.csv found; skipping region mapping.")

    schema_xlsx = _resolve_schema_xlsx(args.schema_xlsx)
    if schema_xlsx:
        try:
            fix_workbook(
                input_path=args.output,
                schema_path=schema_xlsx,
                output_path=args.output,
            )
            log.info("Applied schema-based datatype normalization using %s", schema_xlsx)
        except Exception as e:
            log.warning("Could not normalize workbook datatypes with schema %s: %s", schema_xlsx, e)
    else:
        log.warning("No schema workbook found; skipping datatype normalization.")

    try:
        with open(issue_file, "w", encoding="utf-8") as f:
            json.dump(run_issues, f, ensure_ascii=False, indent=2)
        log.info("Wrote issues report: %s", issue_file)
    except Exception as e:
        log.warning("Could not write issues report %s: %s", issue_file, e)

    log.info("Done. Cohort workbook written to %s", args.output)


if __name__ == "__main__":
    cli()
