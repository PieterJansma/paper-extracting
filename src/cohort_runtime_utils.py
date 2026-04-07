from __future__ import annotations

import os
import json
import logging
import re
from typing import Any, Dict, List, Iterable, Tuple

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from llm_client import OpenAICompatibleClient
from extract_pipeline import extract_fields

# ==============================================================================
# Helpers
# ==============================================================================

TASK_SECTION_PREFIX = "task_"
DEFAULT_PROMPTS_FILE = "prompts/prompts_cohort.toml"

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _load_toml_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            data = toml.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {path}")
    except Exception as e:
        raise SystemExit(f"Could not parse TOML file {path}: {e}")

    if not isinstance(data, dict):
        raise SystemExit(f"Invalid TOML root in {path}: expected table/dict at top-level.")
    return data


def _has_task_sections(cfg: Dict[str, Any]) -> bool:
    return any(str(k).startswith(TASK_SECTION_PREFIX) for k in cfg.keys())


def _merge_task_sections(target_cfg: Dict[str, Any], source_cfg: Dict[str, Any]) -> None:
    for k, v in source_cfg.items():
        if str(k).startswith(TASK_SECTION_PREFIX):
            target_cfg[k] = v


def _default_prompt_candidates(config_path: str) -> List[str]:
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = [
        os.path.join(cfg_dir, DEFAULT_PROMPTS_FILE),
        os.path.abspath(DEFAULT_PROMPTS_FILE),
    ]
    out: List[str] = []
    seen = set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out


def load_config(path: str, prompts_path: str | None = None) -> Dict[str, Any]:
    cfg = _load_toml_file(path)

    if prompts_path:
        prompts_cfg = _load_toml_file(prompts_path)
        _merge_task_sections(cfg, prompts_cfg)
    elif not _has_task_sections(cfg):
        for candidate in _default_prompt_candidates(path):
            if not os.path.isfile(candidate):
                continue
            prompts_cfg = _load_toml_file(candidate)
            _merge_task_sections(cfg, prompts_cfg)
            break

    if not _has_task_sections(cfg):
        raise SystemExit(
            "No task_* prompt sections found. Put prompts in config or provide prompts/prompts_cohort.toml "
            "via PDF_EXTRACT_PROMPTS."
        )
    return cfg


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


def _has_payload_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, list):
        return any(_has_payload_value(x) for x in v)
    if isinstance(v, dict):
        return any(_has_payload_value(x) for x in v.values())
    return True


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


