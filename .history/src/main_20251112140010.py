from __future__ import annotations
import os
import sys
import logging
from typing import Any, Dict

try:
    import tomllib as toml  # Python 3.11+
except ModuleNotFoundError:
    import tomli as toml  # type: ignore  # Python 3.10 fallback

from .llm_client import OpenAICompatibleClient
from .extract_pipeline import load_pdf_text, extract_n_included

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)

def cli():
    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.toml")
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main")

    llm_cfg = cfg["llm"]
    pdf_cfg = cfg["pdf"]
    task_cfg = cfg.get("task", {})

    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        api_key=llm_cfg.get("api_key", "sk-local"),
        model=llm_cfg.get("model", "numind/NuExtract-2.0-8B"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
    )

    paper_text = load_pdf_text(pdf_cfg["path"], max_pages=pdf_cfg.get("max_pages"))
    log.info("PDF loaded (%d chars)", len(paper_text))

    if task_cfg.get("name") == "n_included":
        result = extract_n_included(
            client,
            paper_text,
            use_grammar=bool(llm_cfg.get("use_grammar", False)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 256)),
            use_vllm_template=False,  # set True if you're serving with vLLM and want template via extra_body
        )
        print(result)
    else:
        log.error("Unknown task: %r (try 'n_included')", task_cfg.get("name"))
        sys.exit(2)

if __name__ == "__main__":
    cli()
