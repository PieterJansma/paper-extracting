from __future__ import annotations

import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple, Union

from pypdf import PdfReader

from llm_client import OpenAICompatibleClient

try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)


# ==============================================================================
# PDF Handling
# ==============================================================================

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    """Load text from PDF, optionally limiting pages."""
    if not path:
        return ""
    try:
        reader = PdfReader(path)
    except Exception as e:
        log.error("Failed to read PDF %r: %s", path, e)
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


# ==============================================================================
# JSON parsing helpers
# ==============================================================================

def _strip_markdown_fences(s: str) -> str:
    if not s:
        return ""
    t = s.strip()

    # remove leading ```lang
    if "```" in t:
        t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", t)
        # remove trailing ```
        t = re.sub(r"\n\s*```\s*$", "", t)
        t = t.strip()
    return t


def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    if not s:
        return {}
    t = _strip_markdown_fences(s)

    # Sometimes models prepend text. Try to find the first JSON object.
    # (Conservatief: alleen objecten, geen arrays.)
    if not t.startswith("{"):
        m = re.search(r"\{", t)
        if m:
            t = t[m.start():].strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        # Dit is belangrijk om te zien waarom passes "leeg" lijken.
        log.warning("JSON decode failed (%s). Raw head: %r", e, t[:400])
        return {}


# ==============================================================================
# NuExtract prompt building
# ==============================================================================

def _build_nuextract_prompt(
    template_json: str,
    instructions: Optional[str],
    paper_text: str,
) -> str:
    """Construct the prompt strictly following NuExtract format."""
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()

    blocks: List[str] = []
    if template_json:
        blocks += ["# Template:", template_json]
    if instr:
        blocks += ["# Instructions:", instr]
    if paper_text:
        blocks += ["# Context:", paper_text]

    return "\n".join(blocks)


# ==============================================================================
# Schema completion + type normalization
# ==============================================================================

SchemaSpec = Dict[str, Tuple[str, Any]]
# where Tuple is (kind, prototype) where kind in {"scalar","list","dict"}


def _parse_template_schema(template_json: Optional[str]) -> SchemaSpec:
    """
    Parse the template JSON into a shallow schema:
    - scalar -> default None
    - list   -> default []
    - dict   -> default {}
    We only enforce top-level keys (matches your use-case).
    """
    if not template_json:
        return {}

    try:
        tpl = json.loads(template_json)
    except Exception as e:
        log.warning("Template JSON is not valid JSON (%s). No schema completion applied.", e)
        return {}

    if not isinstance(tpl, dict):
        return {}

    schema: SchemaSpec = {}
    for k, v in tpl.items():
        if isinstance(v, list):
            schema[k] = ("list", v)
        elif isinstance(v, dict):
            schema[k] = ("dict", v)
        else:
            schema[k] = ("scalar", v)
    return schema


def _apply_schema_defaults(parsed: Dict[str, Any], schema: SchemaSpec) -> Dict[str, Any]:
    """
    Ensure output always contains all top-level keys from template schema.
    Missing keys get defaults:
      - scalar -> None
      - list   -> []
      - dict   -> {}
    Also preserves any extra keys returned by the model.
    """
    if not schema:
        return parsed

    out: Dict[str, Any] = {}

    # fill expected keys
    for k, (kind, _proto) in schema.items():
        if k in parsed:
            out[k] = parsed[k]
        else:
            if kind == "list":
                out[k] = []
            elif kind == "dict":
                out[k] = {}
            else:
                out[k] = None

    # keep extras
    for k, v in parsed.items():
        if k not in out:
            out[k] = v

    return out


def _normalize_values(obj: Any) -> Any:
    """
    Light normalization:
    - "" -> None
    - "null" -> None
    - "true"/"false" -> bool
    Applies recursively for dict/list.
    """
    if isinstance(obj, dict):
        return {k: _normalize_values(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_normalize_values(v) for v in obj]

    if isinstance(obj, str):
        s = obj.strip()
        if s == "":
            return None
        if s.lower() == "null":
            return None
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        return obj

    return obj


# ==============================================================================
# Merge logic
# ==============================================================================

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge JSON results.
    - Keep explicit nulls so schema stays complete across passes/papers.
    - Never overwrite an existing non-null value with null.
    - Lists are concatenated then deduped (case-insensitive for strings).
    """
    if not acc:
        return dict(cur)
    if not cur:
        return acc

    for k, v in cur.items():
        if v is None:
            # keep null only if key not present at all
            if k not in acc:
                acc[k] = None
            continue

        if k not in acc or acc[k] is None:
            acc[k] = v
            continue

        existing = acc[k]

        if isinstance(existing, list) and isinstance(v, list):
            combined = existing + v
            seen = set()
            deduped = []
            for item in combined:
                if isinstance(item, str):
                    key = item.lower().strip()
                else:
                    key = str(item)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            acc[k] = deduped
        else:
            # last write wins for scalars/dicts
            acc[k] = v

    return acc


# ==============================================================================
# Main Extraction Function
# ==============================================================================

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

    log.info(
        "Prompt size: %d chars (max_tokens=%d, temp=%.2f)",
        len(prompt),
        int(max_tokens),
        float(temperature),
    )

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
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            grammar=(GRAMMAR_JSON_INT_OR_NULL if use_grammar else None),
        )
    except Exception as e:
        log.error("LLM Call failed: %s", e)
        return {}

    parsed = _json_load_stripping_fences(raw_response)
    parsed = _normalize_values(parsed)

    # ✅ Fill missing keys based on template schema so RAW JSON is never "sparse"
    schema = _parse_template_schema(template_json)
    parsed = _apply_schema_defaults(parsed, schema)

    return parsed
