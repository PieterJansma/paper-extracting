from __future__ import annotations

import os
import json
import argparse
import logging
import importlib.util
import re
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
        chunking_enabled=bool(llm_cfg.get("chunking_enabled", True)),
        long_text_threshold_chars=int(llm_cfg.get("long_text_threshold_chars", 60000)),
        chunk_size_chars=int(llm_cfg.get("chunk_size_chars", 45000)),
        chunk_overlap_chars=int(llm_cfg.get("chunk_overlap_chars", 4000)),
        max_chunks=(
            int(llm_cfg["max_chunks"])
            if llm_cfg.get("max_chunks") is not None
            else None
        ),
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


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, list) or isinstance(v, dict):
        return len(v) == 0
    return False


def _combine_collection_event_results(
    core_result: Dict[str, Any] | None,
    enrich_result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    core_items = (core_result or {}).get("collection_events", []) or []
    enrich_items = (enrich_result or {}).get("collection_events", []) or []

    merged: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}

    def _norm_name(item: Dict[str, Any]) -> str:
        raw = str(item.get("name") or "").strip().lower()
        if not raw:
            return raw

        txt = re.sub(r"[-_/]", " ", raw)
        txt = re.sub(r"\s+", " ", txt).strip()

        if txt in {"baseline", "enrolment", "enrollment", "t0", "time 0"}:
            return "m0"

        # "6 months", "6 month follow up", "month 6"
        m = re.search(r"\b(\d+)\s*(?:month|months|mo)\b", txt)
        if m:
            return f"m{int(m.group(1))}"
        m = re.search(r"\bmonth\s*(\d+)\b", txt)
        if m:
            return f"m{int(m.group(1))}"

        # "year 1", "1 year", "12-month" equivalents
        y = re.search(r"\byear\s*(\d+)\b", txt)
        if y:
            return f"m{int(y.group(1)) * 12}"
        y = re.search(r"\b(\d+)\s*year\b", txt)
        if y:
            return f"m{int(y.group(1)) * 12}"

        return txt

    for item in core_items:
        if not isinstance(item, dict):
            continue
        out = dict(item)
        merged.append(out)
        n = _norm_name(out)
        if n:
            by_name[n] = out

    for item in enrich_items:
        if not isinstance(item, dict):
            continue
        n = _norm_name(item)
        if n and n in by_name:
            target = by_name[n]
            for k, v in item.items():
                if _is_empty_value(v):
                    continue
                cur = target.get(k)
                if isinstance(cur, list) and isinstance(v, list):
                    target[k] = _dedupe_keep_order([str(x) for x in cur + v if not _is_empty_value(x)])
                elif _is_empty_value(cur):
                    target[k] = v
                else:
                    target[k] = v
        else:
            merged.append(dict(item))

    return {"collection_events": merged}


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


def _normalize_ws(text: str) -> str:
    text = (
        text
        .replace("\u00ad", "")
        .replace("\u00a0", " ")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return re.sub(r"\s+", " ", text).strip()


def _extract_first_doi(text: str) -> str | None:
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    if not m:
        return None
    return m.group(0).rstrip(".,;:)")


def _extract_participant_count_hint(text: str) -> int | None:
    m = re.search(
        r"\b(?:about|approximately|approx\.?|circa|around|at least)?\s*(\d{2,6})\s*"
        r"(?:patients|participants)\s*(?:will be included|included)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_dataset_name_hints(text: str, limit: int = 3) -> List[str]:
    names: List[str] = []
    seen = set()
    for m in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9'()/\- ]{0,60}\bdataset)\b",
        text,
        flags=re.IGNORECASE,
    ):
        cand = _normalize_ws(m.group(1))
        if len(cand) < 8:
            continue
        toks = cand.split()
        if len(toks) > 6:
            cand = " ".join(toks[-6:])
        toks = cand.split()
        while toks and toks[0].lower() in {"and", "or", "the", "a", "an", "of", "for", "in"}:
            toks = toks[1:]
        cand = " ".join(toks)
        if cand.lower() in {"dataset", "data set"}:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(cand)
        if len(names) >= limit:
            break
    return names


def _extract_sampleset_hints(text: str) -> Tuple[str | None, List[str]]:
    name = None
    for pat in [
        r"\bbiological samples?\b",
        r"\bbiomaterials?\b",
        r"\bbiobank samples?\b",
        r"\bdata-biobank\b",
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            name = _normalize_ws(m.group(0))
            break

    terms: List[Tuple[str, str]] = [
        ("bone marrow", "Bone marrow"),
        ("whole blood", "Whole blood"),
        ("blood", "Blood"),
        ("serum", "Serum"),
        ("plasma", "Plasma"),
        ("urine", "Urine"),
        ("saliva", "Saliva"),
        ("tissue", "Tissue"),
        ("feces", "Feces"),
        ("stool", "Stool"),
        ("dna", "DNA"),
        ("rna", "RNA"),
    ]
    found: List[str] = []
    for needle, label in terms:
        if re.search(rf"\b{re.escape(needle)}\b", text, flags=re.IGNORECASE):
            found.append(label)
    return name, _dedupe_keep_order(found)


def _repair_contact_point_from_people(result: Dict[str, Any]) -> None:
    people = result.get("people_involved", [])
    if not isinstance(people, list) or not people:
        return

    cp_first = str(result.get("contact_point_first_name") or "").strip()
    cp_last = str(result.get("contact_point_last_name") or "").strip()
    cp_email = str(result.get("contact_point_email") or "").strip().lower()

    if len(cp_first) > 1:
        return

    for person in people:
        if not isinstance(person, dict):
            continue

        p_first = str(person.get("first_name") or "").strip()
        p_last = str(person.get("last_name") or "").strip()
        p_email = str(person.get("email") or "").strip().lower()

        if len(p_first) <= 1:
            continue

        if cp_email and p_email and cp_email == p_email:
            result["contact_point_first_name"] = p_first
            if not cp_last and p_last:
                result["contact_point_last_name"] = p_last
            return

        if cp_last and p_last and cp_last.lower() == p_last.lower():
            result["contact_point_first_name"] = p_first
            if not cp_email and p_email:
                result["contact_point_email"] = p_email
            return


def _postprocess_section_results(
    per_section_results: Dict[str, Dict[str, Any]],
    _paper_text: str,
) -> None:
    paper_text = _normalize_ws(_paper_text)

    overview = per_section_results.get("task_overview")
    info = per_section_results.get("task_information")
    population = per_section_results.get("task_population")
    linkage = per_section_results.get("task_linkage")
    access = per_section_results.get("task_access_conditions")
    datasets = per_section_results.get("task_datasets")
    samplesets = per_section_results.get("task_samplesets")

    health_context = False
    if isinstance(overview, dict):
        types = [x.lower() for x in _as_list_str(overview.get("type"))]
        if any(x in types for x in ["clinical trial", "cohort study", "registry", "disease specific", "health records", "biobank"]):
            health_context = True
    if isinstance(info, dict):
        themes = [x.lower() for x in _as_list_str(info.get("theme"))]
        if "health" in themes:
            health_context = True

    # Repair contact point name truncation from the extracted people list.
    for contrib_key in ("task_contributors", "task_contributors_people"):
        contrib = per_section_results.get(contrib_key)
        if isinstance(contrib, dict):
            _repair_contact_point_from_people(contrib)

    # Fix obvious population contradictions and recover explicit participant counts.
    if isinstance(population, dict):
        groups = [x.lower() for x in _as_list_str(population.get("population_age_groups"))]
        age_min = population.get("age_min")
        try:
            if "adult (18+ years)" in groups and age_min is not None and float(age_min) < 18:
                population["age_min"] = None
        except Exception:
            pass

        if _is_empty_value(population.get("number_of_participants")):
            hinted_n = _extract_participant_count_hint(paper_text)
            if hinted_n is not None:
                population["number_of_participants"] = hinted_n

    # If access wording says "available upon request", map to restricted access.
    if isinstance(access, dict):
        request_pat = r"available (?:upon|on) request|upon reasonable request|made available on request"
        if _is_empty_value(access.get("access_rights")) and re.search(request_pat, paper_text, flags=re.IGNORECASE):
            access["access_rights"] = "Restricted access"
        if _is_empty_value(access.get("data_access_conditions_description")):
            m = re.search(rf"[^.]*({request_pat})[^.]*\.", paper_text, flags=re.IGNORECASE)
            if m:
                access["data_access_conditions_description"] = _normalize_ws(m.group(0))[:400]

    # Clean linkage_options: keep only source-like entries, drop outcomes.
    if isinstance(linkage, dict):
        opts = linkage.get("linkage_options")
        if isinstance(opts, str) and opts.strip():
            parts = [p.strip() for p in re.split(r"[;,]", opts) if p.strip()]
            cleaned = [
                p for p in parts
                if not re.search(r"\b(survival|mortality|outcome|response|recurrence)\b", p, flags=re.IGNORECASE)
            ]
            linkage["linkage_options"] = "; ".join(_dedupe_keep_order(cleaned)) if cleaned else None

    # Recover datasets from explicit "... dataset" phrases when extraction returns empty.
    if isinstance(datasets, dict):
        ds_items = datasets.get("datasets")
        if isinstance(ds_items, list) and not ds_items:
            names = _extract_dataset_name_hints(paper_text)
            if names:
                datasets["datasets"] = [
                    {
                        "name": nm,
                        "label": None,
                        "dataset_type": [],
                        "unit_of_observation": None,
                        "keywords": [],
                        "description": None,
                        "number_of_rows": None,
                        "since_version": None,
                        "until_version": None,
                    }
                    for nm in names
                ]

    # Recover one sampleset from explicit sample-type evidence when empty.
    if isinstance(samplesets, dict):
        ss_items = samplesets.get("samplesets")
        if isinstance(ss_items, list) and not ss_items:
            ss_name, ss_types = _extract_sampleset_hints(paper_text)
            if ss_name and ss_types:
                samplesets["samplesets"] = [
                    {
                        "name": ss_name,
                        "sample_types": ss_types,
                    }
                ]

    # Fill missing PID with the first explicit DOI (if present).
    if isinstance(overview, dict) and _is_empty_value(overview.get("pid")):
        doi = _extract_first_doi(paper_text)
        if doi:
            overview["pid"] = doi

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
        if health_context and not _as_list_str(info.get("theme")):
            info["theme"] = ["Health"]

        # Remove common article-editorial text that is not resource provenance.
        prov = str(info.get("provenance_statement") or "").strip().lower()
        if prov and ("not commissioned" in prov or "peer reviewed" in prov):
            info["provenance_statement"] = None

        laws = _as_list_str(info.get("applicable_legislation"))
        if health_context and "data governance act" not in [x.lower() for x in laws]:
            info["applicable_legislation"] = _dedupe_keep_order(laws + ["Data Governance Act"])

        # If name mirrors article title and does not look like a resource label, drop it.
        if isinstance(overview, dict):
            ov_name = str(overview.get("name") or "").strip()
            pub_titles = {
                str(p.get("title") or "").strip().lower()
                for p in (info.get("publications") or [])
                if isinstance(p, dict)
            }
            if (
                ov_name
                and ov_name.lower() in pub_titles
                and not re.search(r"\b(study|cohort|biobank|registry|trial|database|databank|network)\b", ov_name, flags=re.IGNORECASE)
            ):
                overview["name"] = None

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
        default="final_result.xlsx",
        help="Output Excel filename.",
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]
    if "F" in selected_passes:
        selected_passes.extend(["F1", "F2"])
    if "D" in selected_passes:
        selected_passes.extend(["D1", "D2"])

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
        prompt_cache_enabled = bool(llm_cfg.get("prompt_cache", False))
        chunking_enabled = bool(llm_cfg.get("chunking_enabled", True))
        long_text_threshold = int(llm_cfg.get("long_text_threshold_chars", 60000))
        if prompt_cache_enabled and not (chunking_enabled and len(paper_text) > long_text_threshold):
            prefix_messages = build_context_prefix_messages(paper_text)
            log.info(
                "Prompt cache enabled: using reusable paper prefix (%d chars).",
                len(paper_text),
            )
        elif prompt_cache_enabled:
            log.info(
                "Prompt cache disabled for this paper (%d chars > long_text_threshold_chars=%d); "
                "chunked extraction will use per-chunk context.",
                len(paper_text),
                long_text_threshold,
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

        if has_split_collection_events:
            per_section_results["task_collection_events"] = _combine_collection_event_results(
                per_section_results.get("task_collection_events_core", {}),
                per_section_results.get("task_collection_events_enrichment", {}),
            )

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
