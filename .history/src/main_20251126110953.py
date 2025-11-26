from __future__ import annotations
import os
import logging
import json
import argparse
from typing import Any, Dict

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from .llm_client import OpenAICompatibleClient
from .extract_pipeline import load_pdf_text, extract_fields, _merge_json_results


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)


def _print_pretty_json(data: Any) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    print(text)


def cli():
    parser = argparse.ArgumentParser(description="Run PDF extraction passes.")
    parser.add_argument(
        "-p", "--passes", 
        nargs="+", 
        default=["all"],
        help="Specify passes: A, B, C, D, E, F, G. Default: 'all'"
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]
    
    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.toml")
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main")

    llm_cfg = cfg["llm"]
    pdf_cfg = cfg["pdf"]
    
    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        api_key=llm_cfg.get("api_key", "sk-local"),
        model=llm_cfg.get("model", "numind/NuExtract-2.0-8B"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
    )

    paper_text = load_pdf_text(pdf_cfg["path"], max_pages=pdf_cfg.get("max_pages"))
    log.info("PDF loaded (%d chars)", len(paper_text))

    def run_pass(name, cfg_section_key):
        task_cfg = cfg.get(cfg_section_key, {})
        if not task_cfg:
            log.warning(f"Config section '{cfg_section_key}' is empty or missing!")
            return {}
            
        log.info(f"--- Running {name} ---")
        return extract_fields(
            client, paper_text,
            template_json=task_cfg.get("template_json"),
            instructions=task_cfg.get("instructions"),
            use_grammar=bool(llm_cfg.get("use_grammar", False)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 1024)),
        )

    # De lijst met taken, nu inclusief G
    all_tasks = [
        ("A", "Pass A (Main)", "task_main"),
        ("B", "Pass B (Criteria)", "task_criteria"),
        ("C", "Pass C (Design)", "task_design_details"),
        ("D", "Pass D (Population)", "task_population"),
        ("E", "Pass E (Access)", "task_access"),
        ("F", "Pass F (Contributors)", "task_contributors"),
        ("G", "Pass G (Data Model)", "task_datamodel"), # <--- NIEUW
    ]

    merged_results = {}

    for code, name, section in all_tasks:
        if "ALL" in selected_passes or code in selected_passes:
            res = run_pass(f"{name} [{code}]", section)
            merged_results = _merge_json_results(merged_results, res)
        else:
            log.info(f"Skipping {name} (not selected)")

    log.info("--- DONE ---")
    _print_pretty_json(merged_results)


if __name__ == "__main__":
    cli()