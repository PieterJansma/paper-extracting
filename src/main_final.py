from __future__ import annotations

import os
import json
import argparse
import logging
import importlib.util
from typing import Any, Dict, List, Iterable, Tuple

import pandas as pd

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from llm_client import OpenAICompatibleClient
from extract_pipeline import load_pdf_text, extract_fields, build_context_prefix_messages

# ==============================================================================
# Helpers
# ==============================================================================

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)


def stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _ordered_keys_from_template(template_json: str | None) -> List[str]:
    if not template_json:
        return []
    template_json = (
        template_json
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\ufffd", " ")
        .strip()
    )
    try:
        tpl = json.loads(template_json)
    except Exception:
        return []
    if not isinstance(tpl, dict):
        return []
    return list(tpl.keys())


def _ordered_item_keys_from_template(template_json: str | None, list_key: str) -> List[str]:
    """
    For templates like {"subpopulations": [{...}]}, return the nested item keys.
    """
    if not template_json:
        return []
    template_json = (
        template_json
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\ufffd", " ")
        .strip()
    )
    try:
        tpl = json.loads(template_json)
    except Exception:
        return []

    if not isinstance(tpl, dict):
        return []

    arr = tpl.get(list_key)
    if not isinstance(arr, list) or not arr:
        return []

    item0 = arr[0]
    if not isinstance(item0, dict):
        return []

    return list(item0.keys())


def _validate_list_template_keys(section_key: str, list_name: str, keys: List[str]) -> None:
    """
    Fail fast on malformed list templates to prevent silently wrong Excel columns like
    `subpopulation.subpopulations` or `collection_event.collection_events`.
    """
    if not keys:
        raise SystemExit(
            f"Template for {section_key} is invalid or unreadable: no item keys found for '{list_name}'."
        )
    if len(keys) == 1 and keys[0] == list_name:
        raise SystemExit(
            f"Template for {section_key} appears nested incorrectly: item key equals container '{list_name}'."
        )


def _serialize_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _collect_pass_result(
    client: OpenAICompatibleClient,
    paper_text: str,
    task_cfg: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    log: logging.Logger,
    name: str,
    prefix_messages: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    if not task_cfg:
        log.warning("Sectie is leeg of ontbreekt: %s", name)
        return {}

    log.info("--- Running %s ---", name)
    return extract_fields(
        client,
        paper_text,
        template_json=task_cfg.get("template_json"),
        instructions=task_cfg.get("instructions"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        prefix_messages=prefix_messages,
        cache_prompt=bool(llm_cfg.get("prompt_cache", False)),
        timeout=int(llm_cfg.get("timeout", 600)),
    )


def _build_resource_row(
    label: str,
    pdf_path: str,
    section_results: Dict[str, Dict[str, Any]],
    section_orders: Dict[str, List[str]],
    section_keys_override: Dict[str, List[str]] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    row: Dict[str, Any] = {
        "paper": label,
        "pdf_path": pdf_path,
    }

    columns = ["paper", "pdf_path"]
    for section, result in section_results.items():
        if section_keys_override and section in section_keys_override:
            ordered_keys = section_keys_override[section]
        else:
            ordered_keys = section_orders.get(section, [])
        for key in ordered_keys:
            col = f"{section}.{key}"
            columns.append(col)
            row[col] = _serialize_value(result.get(key))

    return row, columns


def _build_list_rows(
    label: str,
    items: Iterable[Dict[str, Any]],
    prefix: str,
    ordered_keys: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    columns = ["paper", f"{prefix}_index"] + [f"{prefix}.{k}" for k in ordered_keys]

    out_idx = 1
    for item in items:
        # Skip structurally empty items (all fields null/empty), which can appear
        # when model output partially matches schema but contains no actual values.
        has_payload = False
        for key in ordered_keys:
            val = item.get(key)
            if val is None:
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            if isinstance(val, list) and len(val) == 0:
                continue
            if isinstance(val, dict) and len(val) == 0:
                continue
            has_payload = True
            break
        if not has_payload:
            continue

        row: Dict[str, Any] = {
            "paper": label,
            f"{prefix}_index": out_idx,
        }
        for key in ordered_keys:
            row[f"{prefix}.{key}"] = _serialize_value(item.get(key))
        rows.append(row)
        out_idx += 1

    return rows, columns


def _combine_contributor_results(
    org_result: Dict[str, Any] | None,
    people_result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    org = org_result or {}
    ppl = people_result or {}
    return {
        "organisations_involved": org.get("organisations_involved", []),
        "publisher": org.get("publisher"),
        "creator": org.get("creator", []),
        "people_involved": ppl.get("people_involved", []),
        "contact_point_first_name": ppl.get("contact_point_first_name"),
        "contact_point_last_name": ppl.get("contact_point_last_name"),
        "contact_point_email": ppl.get("contact_point_email"),
    }


def _as_list_str(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if x is not None and str(x).strip() != ""]
    return []


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _postprocess_section_results(
    per_section_results: Dict[str, Dict[str, Any]],
    _paper_text: str,
) -> None:
    overview = per_section_results.get("task_overview")
    info = per_section_results.get("task_information")

    health_context = False
    if isinstance(overview, dict):
        types = [x.lower() for x in _as_list_str(overview.get("type"))]
        if any(x in types for x in ["clinical trial", "cohort study", "registry", "disease specific", "health records"]):
            health_context = True
    if isinstance(info, dict):
        themes = [x.lower() for x in _as_list_str(info.get("theme"))]
        if "health" in themes:
            health_context = True

    # ------------------------------------------------------------------
    # PASS X3 fallback: derive resource-level areas from collection events
    # ------------------------------------------------------------------
    aoi = per_section_results.get("task_areas_of_information")
    ce = per_section_results.get("task_collection_events")
    if isinstance(aoi, dict) and isinstance(ce, dict):
        aoi_vals = _as_list_str(aoi.get("areas_of_information"))
        if not aoi_vals:
            merged: List[str] = []
            for ev in ce.get("collection_events", []) or []:
                if isinstance(ev, dict):
                    merged.extend(_as_list_str(ev.get("areas_of_information")))
            aoi["areas_of_information"] = _dedupe_keep_order(merged)

    # ------------------------------------------------------------------
    # PASS H default: Data Governance Act for health resources
    # ------------------------------------------------------------------
    if isinstance(info, dict):
        # Remove common article-editorial text that is not resource provenance.
        prov = str(info.get("provenance_statement") or "").strip().lower()
        if prov and ("not commissioned" in prov or "peer reviewed" in prov):
            info["provenance_statement"] = None

        laws = _as_list_str(info.get("applicable_legislation"))
        if health_context and "data governance act" not in [x.lower() for x in laws]:
            info["applicable_legislation"] = _dedupe_keep_order(laws + ["Data Governance Act"])

    # Keep legislation policy consistent in list sheets when context is health.
    if health_context:
        for item in (per_section_results.get("task_subpopulations", {}) or {}).get("subpopulations", []) or []:
            if isinstance(item, dict):
                laws = _as_list_str(item.get("applicable_legislation"))
                if "data governance act" not in [x.lower() for x in laws]:
                    item["applicable_legislation"] = _dedupe_keep_order(laws + ["Data Governance Act"])
        for item in (per_section_results.get("task_collection_events", {}) or {}).get("collection_events", []) or []:
            if isinstance(item, dict):
                laws = _as_list_str(item.get("applicable_legislation"))
                if "data governance act" not in [x.lower() for x in laws]:
                    item["applicable_legislation"] = _dedupe_keep_order(laws + ["Data Governance Act"])


# ==============================================================================
# CLI
# ==============================================================================

def cli() -> None:
    parser = argparse.ArgumentParser(description="Run PDF extraction passes (final config).")
    parser.add_argument(
        "-p", "--passes",
        nargs="+",
        default=["all"],
        help="Specify passes: A, B, C, D, E, F/F1/F2, G, H, X1, X2, X3, Y or 'all'.",
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
        default="final_result.xlsx",
        help="Output Excel filename.",
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]
    if "F" in selected_passes:
        selected_passes.extend(["F1", "F2"])

    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.final.toml")
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main_final")

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

    pass_defs = [
        ("A", "PASS A: Overview", "task_overview"),
        ("B", "PASS B: Design & structure", "task_design_structure"),
        ("C", "PASS C: Subpopulations", "task_subpopulations"),
        ("D", "PASS D: Collection events", "task_collection_events"),
        ("E", "PASS E: Population", "task_population"),
        ("X1", "PASS X1: Datasets", "task_datasets"),
        ("X2", "PASS X2: Samplesets", "task_samplesets"),
        ("X3", "PASS X3: Areas of information", "task_areas_of_information"),
        ("Y", "PASS Y: Linkage", "task_linkage"),
        ("G", "PASS G: Access conditions", "task_access_conditions"),
        ("H", "PASS H: Information", "task_information"),
    ]
    if has_split_contributors:
        pass_defs.insert(5, ("F1", "PASS F1: Contributors (organisations)", "task_contributors_org"))
        pass_defs.insert(6, ("F2", "PASS F2: Contributors (people/contact)", "task_contributors_people"))
    else:
        pass_defs.insert(5, ("F", "PASS F: Contributors", "task_contributors"))

    # Preload template orders for stable column order
    section_orders: Dict[str, List[str]] = {}
    for _, _, section_key in pass_defs:
        task_cfg = cfg.get(section_key, {})
        section_orders[section_key] = _ordered_keys_from_template(task_cfg.get("template_json"))

    resource_sections = [
        "task_overview",
        "task_design_structure",
        "task_population",
        "task_contributors",
        "task_areas_of_information",
        "task_linkage",
        "task_access_conditions",
        "task_information",
    ]
    list_sections = {
        "task_subpopulations": "subpopulations",
        "task_collection_events": "collection_events",
        "task_datasets": "datasets",
        "task_samplesets": "samplesets",
    }

    list_section_orders: Dict[str, List[str]] = {}
    for section_key, list_name in list_sections.items():
        task_cfg = cfg.get(section_key, {})
        keys = _ordered_item_keys_from_template(
            task_cfg.get("template_json"),
            list_name,
        )
        _validate_list_template_keys(section_key, list_name, keys)
        list_section_orders[section_key] = keys

    resource_section_keys: Dict[str, List[str]] = {
        "task_contributors": [
            "publisher",
            "creator",
            "contact_point_first_name",
            "contact_point_last_name",
            "contact_point_email",
        ],
        "task_information": [
            "funding_statement",
            "citation_requirements",
            "acknowledgements",
            "provenance_statement",
            "supplementary_information",
            "theme",
            "applicable_legislation",
        ],
    }

    resource_rows: List[Dict[str, Any]] = []
    resource_columns: List[str] = []

    subpop_rows: List[Dict[str, Any]] = []
    ce_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    sampleset_rows: List[Dict[str, Any]] = []
    organisation_rows: List[Dict[str, Any]] = []
    people_rows: List[Dict[str, Any]] = []
    publication_rows: List[Dict[str, Any]] = []
    documentation_rows: List[Dict[str, Any]] = []

    for pdf_path, label in zip(pdf_paths, labels):
        paper_text = load_pdf_text(pdf_path, max_pages=pdf_cfg.get("max_pages"))
        log.info("PDF '%s' loaded (%d chars)", pdf_path, len(paper_text))

        prefix_messages: List[Dict[str, str]] | None = None
        if bool(llm_cfg.get("prompt_cache", False)):
            prefix_messages = build_context_prefix_messages(paper_text)
            log.info(
                "Prompt cache enabled: using reusable paper prefix (%d chars).",
                len(paper_text),
            )

        per_section_results: Dict[str, Dict[str, Any]] = {}

        for code, name, section_key in pass_defs:
            if "ALL" in selected_passes or code in selected_passes:
                task_cfg = cfg.get(section_key, {})
                result = _collect_pass_result(
                    client,
                    paper_text,
                    task_cfg,
                    llm_cfg,
                    log,
                    name,
                    prefix_messages=prefix_messages,
                )
                per_section_results[section_key] = result
            else:
                log.info("Skipping %s (not selected)", name)

        _postprocess_section_results(per_section_results, paper_text)

        # Resource-level combined row
        if has_split_contributors:
            contributors_result = _combine_contributor_results(
                per_section_results.get("task_contributors_org", {}),
                per_section_results.get("task_contributors_people", {}),
            )
        else:
            contributors_result = per_section_results.get("task_contributors", {})

        resource_result = {
            section: per_section_results.get(section, {})
            for section in resource_sections
            if section in per_section_results
        }
        if (
            ("ALL" in selected_passes)
            or ("F" in selected_passes)
            or ("F1" in selected_passes)
            or ("F2" in selected_passes)
        ):
            resource_result["task_contributors"] = contributors_result

        row, cols = _build_resource_row(
            label,
            pdf_path,
            resource_result,
            section_orders,
            section_keys_override=resource_section_keys,
        )
        resource_rows.append(row)
        if not resource_columns:
            resource_columns = cols

        # List sections
        for section_key, list_name in list_sections.items():
            section_result = per_section_results.get(section_key, {})
            items = section_result.get(list_name, []) if isinstance(section_result, dict) else []
            ordered_keys = list_section_orders.get(section_key, [])

            if section_key == "task_subpopulations":
                rows, _ = _build_list_rows(label, items, "subpopulation", ordered_keys)
                subpop_rows.extend(rows)
            elif section_key == "task_collection_events":
                rows, _ = _build_list_rows(label, items, "collection_event", ordered_keys)
                ce_rows.extend(rows)
            elif section_key == "task_datasets":
                rows, _ = _build_list_rows(label, items, "dataset", ordered_keys)
                dataset_rows.extend(rows)
            elif section_key == "task_samplesets":
                rows, _ = _build_list_rows(label, items, "sampleset", ordered_keys)
                sampleset_rows.extend(rows)

        # Contributors tables
        # Contributors tables
        if isinstance(contributors_result, dict):
            organisations = contributors_result.get("organisations_involved", [])
            people = contributors_result.get("people_involved", [])

            org_keys = [
                "id",
                "type",
                "name",
                "organisation",
                "other_organisation",
                "department",
                "website",
                "email",
                "logo",
                "role",
                "is_lead_organisation",
            ]
            rows, _ = _build_list_rows(label, organisations, "organisation", org_keys)
            organisation_rows.extend(rows)

            people_keys = [
                "role",
                "role_description",
                "first_name",
                "last_name",
                "prefix",
                "initials",
                "title",
                "organisation",
                "email",
                "orcid",
                "homepage",
                "photo",
                "expertise",
            ]
            rows, _ = _build_list_rows(label, people, "person", people_keys)
            people_rows.extend(rows)

        # Information tables
        info_result = per_section_results.get("task_information", {})
        if isinstance(info_result, dict):
            publications = info_result.get("publications", [])
            documentation = info_result.get("documentation", [])

            pub_keys = ["doi", "title", "is_design_publication", "reference"]
            rows, _ = _build_list_rows(label, publications, "publication", pub_keys)
            publication_rows.extend(rows)

            doc_keys = ["name", "type", "description", "url", "file"]
            rows, _ = _build_list_rows(label, documentation, "documentation", doc_keys)
            documentation_rows.extend(rows)

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

    log.info("Writing Excel with engine=%s", excel_engine)

    with pd.ExcelWriter(args.output, engine=excel_engine) as writer:
        # Resources sheet
        res_df = pd.DataFrame(resource_rows, columns=resource_columns)
        res_df.to_excel(writer, sheet_name="resources", index=False)

        # Subpopulations sheet
        sub_cols = ["paper", "subpopulation_index"] + [
            f"subpopulation.{k}" for k in list_section_orders.get("task_subpopulations", [])
        ]
        pd.DataFrame(subpop_rows, columns=sub_cols).to_excel(
            writer, sheet_name="subpopulations", index=False
        )

        # Collection events sheet
        ce_cols = ["paper", "collection_event_index"] + [
            f"collection_event.{k}" for k in list_section_orders.get("task_collection_events", [])
        ]
        pd.DataFrame(ce_rows, columns=ce_cols).to_excel(
            writer, sheet_name="collection_events", index=False
        )

        # Datasets sheet
        ds_cols = ["paper", "dataset_index"] + [
            f"dataset.{k}" for k in list_section_orders.get("task_datasets", [])
        ]
        pd.DataFrame(dataset_rows, columns=ds_cols).to_excel(
            writer, sheet_name="datasets", index=False
        )

        # Samplesets sheet
        ss_cols = ["paper", "sampleset_index"] + [
            f"sampleset.{k}" for k in list_section_orders.get("task_samplesets", [])
        ]
        pd.DataFrame(sampleset_rows, columns=ss_cols).to_excel(
            writer, sheet_name="samplesets", index=False
        )

        # Organisations sheet
        org_cols = ["paper", "organisation_index"] + [
            "organisation.id",
            "organisation.type",
            "organisation.name",
            "organisation.organisation",
            "organisation.other_organisation",
            "organisation.department",
            "organisation.website",
            "organisation.email",
            "organisation.logo",
            "organisation.role",
            "organisation.is_lead_organisation",
        ]
        pd.DataFrame(organisation_rows, columns=org_cols).to_excel(
            writer, sheet_name="organisations", index=False
        )

        # People involved sheet
        people_cols = ["paper", "person_index"] + [
            "person.role",
            "person.role_description",
            "person.first_name",
            "person.last_name",
            "person.prefix",
            "person.initials",
            "person.title",
            "person.organisation",
            "person.email",
            "person.orcid",
            "person.homepage",
            "person.photo",
            "person.expertise",
        ]
        pd.DataFrame(people_rows, columns=people_cols).to_excel(
            writer, sheet_name="people", index=False
        )

        # Publications sheet
        pub_cols = ["paper", "publication_index"] + [
            "publication.doi",
            "publication.title",
            "publication.is_design_publication",
            "publication.reference",
        ]
        pd.DataFrame(publication_rows, columns=pub_cols).to_excel(
            writer, sheet_name="publications", index=False
        )

        # Documentation sheet
        doc_cols = ["paper", "documentation_index"] + [
            "documentation.name",
            "documentation.type",
            "documentation.description",
            "documentation.url",
            "documentation.file",
        ]
        pd.DataFrame(documentation_rows, columns=doc_cols).to_excel(
            writer, sheet_name="documentation", index=False
        )

    log.info("Excel succesvol opgeslagen: %s", args.output)


if __name__ == "__main__":
    cli()
