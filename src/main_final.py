from __future__ import annotations

import os
import json
import argparse
import logging
from typing import Any, Dict, List, Iterable, Tuple

import pandas as pd

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from llm_client import OpenAICompatibleClient
from extract_pipeline import load_pdf_text, extract_fields

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
    try:
        tpl = json.loads(template_json)
    except Exception:
        return []
    if not isinstance(tpl, dict):
        return []
    return list(tpl.keys())


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
    )


def _build_resource_row(
    label: str,
    pdf_path: str,
    section_results: Dict[str, Dict[str, Any]],
    section_orders: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], List[str]]:
    row: Dict[str, Any] = {
        "paper": label,
        "pdf_path": pdf_path,
    }

    columns = ["paper", "pdf_path"]
    for section, result in section_results.items():
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

    for idx, item in enumerate(items, start=1):
        row: Dict[str, Any] = {
            "paper": label,
            f"{prefix}_index": idx,
        }
        for key in ordered_keys:
            row[f"{prefix}.{key}"] = _serialize_value(item.get(key))
        rows.append(row)

    return rows, columns


# ==============================================================================
# CLI
# ==============================================================================

def cli() -> None:
    parser = argparse.ArgumentParser(description="Run PDF extraction passes (final config).")
    parser.add_argument(
        "-p", "--passes",
        nargs="+",
        default=["all"],
        help="Specify passes: A, B, C, D, E, X1, X2, X3, Y or 'all'.",
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
    )

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
    ]

    # Preload template orders for stable column order
    section_orders: Dict[str, List[str]] = {}
    for _, _, section_key in pass_defs:
        task_cfg = cfg.get(section_key, {})
        section_orders[section_key] = _ordered_keys_from_template(task_cfg.get("template_json"))

    resource_sections = [
        "task_overview",
        "task_design_structure",
        "task_population",
        "task_areas_of_information",
        "task_linkage",
    ]
    list_sections = {
        "task_subpopulations": "subpopulations",
        "task_collection_events": "collection_events",
        "task_datasets": "datasets",
        "task_samplesets": "samplesets",
    }

    resource_rows: List[Dict[str, Any]] = []
    resource_columns: List[str] = []

    subpop_rows: List[Dict[str, Any]] = []
    ce_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    sampleset_rows: List[Dict[str, Any]] = []

    for pdf_path, label in zip(pdf_paths, labels):
        paper_text = load_pdf_text(pdf_path, max_pages=pdf_cfg.get("max_pages"))
        log.info("PDF '%s' loaded (%d chars)", pdf_path, len(paper_text))

        per_section_results: Dict[str, Dict[str, Any]] = {}

        for code, name, section_key in pass_defs:
            if "ALL" in selected_passes or code in selected_passes:
                task_cfg = cfg.get(section_key, {})
                result = _collect_pass_result(client, paper_text, task_cfg, llm_cfg, log, name)
                per_section_results[section_key] = result
            else:
                log.info("Skipping %s (not selected)", name)

        # Resource-level combined row
        resource_result = {
            section: per_section_results.get(section, {})
            for section in resource_sections
            if section in per_section_results
        }
        row, cols = _build_resource_row(label, pdf_path, resource_result, section_orders)
        resource_rows.append(row)
        if not resource_columns:
            resource_columns = cols

        # List sections
        for section_key, list_name in list_sections.items():
            section_result = per_section_results.get(section_key, {})
            items = section_result.get(list_name, []) if isinstance(section_result, dict) else []
            ordered_keys = section_orders.get(section_key, [])

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

    log.info("--- DONE EXTRACTING ---")

    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        # Resources sheet
        res_df = pd.DataFrame(resource_rows, columns=resource_columns)
        res_df.to_excel(writer, sheet_name="resources", index=False)

        # Subpopulations sheet
        sub_cols = ["paper", "subpopulation_index"] + [
            f"subpopulation.{k}" for k in section_orders.get("task_subpopulations", [])
        ]
        pd.DataFrame(subpop_rows, columns=sub_cols).to_excel(
            writer, sheet_name="subpopulations", index=False
        )

        # Collection events sheet
        ce_cols = ["paper", "collection_event_index"] + [
            f"collection_event.{k}" for k in section_orders.get("task_collection_events", [])
        ]
        pd.DataFrame(ce_rows, columns=ce_cols).to_excel(
            writer, sheet_name="collection_events", index=False
        )

        # Datasets sheet
        ds_cols = ["paper", "dataset_index"] + [
            f"dataset.{k}" for k in section_orders.get("task_datasets", [])
        ]
        pd.DataFrame(dataset_rows, columns=ds_cols).to_excel(
            writer, sheet_name="datasets", index=False
        )

        # Samplesets sheet
        ss_cols = ["paper", "sampleset_index"] + [
            f"sampleset.{k}" for k in section_orders.get("task_samplesets", [])
        ]
        pd.DataFrame(sampleset_rows, columns=ss_cols).to_excel(
            writer, sheet_name="samplesets", index=False
        )

    log.info("Excel succesvol opgeslagen: %s", args.output)


if __name__ == "__main__":
    cli()
