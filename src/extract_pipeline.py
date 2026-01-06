from __future__ import annotations

import logging
import re
import json
from typing import Optional, Dict, Any, List

from pypdf import PdfReader

from llm_client import OpenAICompatibleClient

try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)


# ---------- PDF Handling ----------

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    """Load text from PDF, optionally limiting pages."""
    if not path:
        return ""
    try:
        reader = PdfReader(path)
    except Exception as e:
        log.error(f"Failed to read PDF: {e}")
        return ""

    pages = reader.pages[:max_pages] if max_pages else reader.pages
    texts: List[str] = []

    for i, p in enumerate(pages):
        try:
            extracted = p.extract_text()
            if extracted:
                texts.append(extracted)
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)

    return "\n\n".join(texts)


# ---------- Helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    if not s:
        return {}
    t = s.strip()

    # Strip triple-backtick fences, if present anywhere
    # Example:
    # ```json
    # { ... }
    # ```
    if "```" in t:
        # remove leading fence
        t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", t)
        # remove trailing fence
        t = re.sub(r"\n\s*```\s*$", "", t)
        t = t.strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        log.debug("JSON Decode Error: %s. Raw output head: %r", e, t[:300])
        return {}


def _build_nuextract_prompt(
    template_json: str,
    instructions: Optional[str],
    paper_text: str,
) -> str:
    """Construct the prompt strictly following NuExtract format."""
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()

    blocks = []
    if template_json:
        blocks += ["# Template:", template_json]
    if instr:
        blocks += ["# Instructions:", instr]
    if paper_text:
        blocks += ["# Context:", paper_text]

    return "\n".join(blocks)


def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """Merge JSON results."""
    if not acc:
        return dict(cur)
    if not cur:
        return acc

    for k, v in cur.items():
        # Keep explicit nulls so schema stays complete across passes/papers.
        # Never overwrite an existing non-null value with null.
        if v is None:
            if k not in acc:
                acc[k] = None
            continue

        if k not in acc:
            acc[k] = v
            continue

        existing = acc[k]

        if isinstance(existing, list) and isinstance(v, list):
            combined = existing + v
            seen = set()
            deduped = []
            for item in combined:
                key = str(item).lower().strip() if isinstance(item, str) else str(item)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            acc[k] = deduped
        else:
            acc[k] = v

    return acc


# ---------- Main Extraction Function ----------

def extract_fields(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    template_json: Optional[str] = None,
    instructions: Optional[str] = None,
    use_grammar: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Execute a single extraction pass using the LLM."""
    if not paper_text:
        return {}

    prompt = _build_nuextract_prompt(template_json or "", instructions, paper_text)

    # Useful debug: prompt size
    log.info("Prompt size: %d chars (max_tokens=%d, temp=%.2f)", len(prompt), max_tokens, float(temperature))

    system_msg = (
        "You are NuExtract. Extract structured data as JSON only. "
        "Do not output markdown fences or explanations."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        raw_response = client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=(GRAMMAR_JSON_INT_OR_NULL if use_grammar else None),
        )
    except Exception as e:
        log.error("LLM Call failed: %s", e)
        return {}

    return _json_load_stripping_fences(raw_response)
